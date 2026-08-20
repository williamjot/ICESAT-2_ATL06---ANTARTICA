"""
thwaites.timeseries.dhdt
========================
Cálculo de dh/dt por ajuste espaço-temporal local ('fitsec'), por tile.

Implementação numérica parametrizada pela configuração, com leitura da elevação
`h_corr` (ou `h_elv`, com aviso), `min_points` ≥ 30 por padrão para evitar
matrizes mal condicionadas e uso do halo dos tiles como vizinhança, gerando nós
somente no núcleo.

Modelo por nó: h ≈ p0 + p1·(t-t_ref) + 0.5·p2·(t-t_ref)² + termos espaciais.
  p1 = dh/dt (m/ano);  p2 = aceleração (m/ano²).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.linalg import LinAlgWarning
from scipy.spatial import cKDTree

from thwaites.config import Config
from thwaites.grid.reproject import to_lonlat
from thwaites.grid.tiles import assign_xy, load_manifest
from thwaites.logging import get_logger


# ------------------------------------------------------------------ numérica
def _mad_std(r):
    return 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))


def _build_A(dx, dy, dt, poly_order, temp_order):
    n = len(dx)
    sx = max(np.std(dx), 1.0)
    sy = max(np.std(dy), 1.0)
    dxn, dyn = dx / sx, dy / sy
    cols = [np.ones(n), dt]
    if temp_order >= 2:
        cols.append(0.5 * dt**2)
    if poly_order >= 1:
        cols += [dxn, dyn, dxn * dyn]
    if poly_order >= 2:
        cols += [dxn**2, dyn**2]
    return np.column_stack(cols)


def _lstsq_iter(A, z, w, n_iter, n_sigma, rlim):
    n, m = A.shape
    mask = np.ones(n, dtype=bool)
    xhat = ehat = None
    rmse = np.nan

    with warnings.catch_warnings():
        # Esperado em nós de baixa densidade; tratado pelo rank-check abaixo.
        warnings.simplefilter("ignore", category=LinAlgWarning)
        for _ in range(n_iter):
            Af, zf, nf = A[mask], z[mask], int(mask.sum())
            if nf < m + 2:
                break
            if w is not None:
                wf = w[mask]
                wf = wf / (wf.mean() + 1e-30)
                W = np.sqrt(wf)
                Aw, zw = Af * W[:, None], zf * W
            else:
                Aw, zw = Af, zf
            try:
                xhat_new, _, rank, _ = linalg.lstsq(Aw, zw, check_finite=False)
            except (linalg.LinAlgError, ValueError):
                break
            if rank < m:
                break
            xhat = xhat_new
            res = zf - Af @ xhat
            rmse = float(np.std(res))
            try:
                AtA = Aw.T @ Aw
                sig2 = float(np.sum(res**2) / max(nf - m, 1))
                ehat = np.sqrt(np.abs(np.diag(linalg.pinv(AtA))) * sig2)
            except Exception:
                ehat = np.full(m, np.nan)
            mad = _mad_std(res)
            if mad < 1e-10:
                break
            bad = (np.abs(res) > n_sigma * mad) | (np.abs(res) > rlim)
            if not bad.any():
                break
            mask[np.where(mask)[0][bad]] = False

    return xhat, ehat, mask, rmse


def _fit_node(xo, yo, ho, to, so, x0, y0, d):
    """Ajuste num nó. `d` = cfg.dhdt. Retorna dict ou None."""
    n = len(ho)
    if n < d.min_points:
        return None
    dt_span = float(to.max() - to.min())
    if dt_span < d.dt_min_years:
        return None

    dx, dy, dt = xo - x0, yo - y0, to - d.t_ref
    if d.use_weights:
        sv = np.where(so <= 0,
                      np.median(so[so > 0]) if (so > 0).any() else 0.05, so)
        wc = 1.0 / (sv**2 + 1e-12)
    else:
        wc = None

    A = _build_A(dx, dy, dt, d.poly_order, d.temp_order)
    xhat, ehat, mask, rmse = _lstsq_iter(
        A, ho, wc, d.max_iter, d.n_sigma, d.resid_limit)
    if xhat is None or mask.sum() < d.min_points:
        return None

    p1v = float(xhat[1])
    if abs(p1v) > d.rate_limit:
        return None

    p2v = p2e = np.nan
    n_years = len(np.unique(np.floor(to[mask]).astype(int)))
    if d.temp_order >= 2 and len(xhat) > 2 and n_years >= d.dt_min_years_accel:
        p2v = float(xhat[2])
        p2e = float(ehat[2]) if ehat is not None else np.nan
        if abs(p2v) > d.accel_limit:
            p2v = p2e = np.nan

    return {
        "p0": float(xhat[0]), "dhdt": p1v, "accel": p2v,
        "p0_err": float(ehat[0]) if ehat is not None else np.nan,
        "dhdt_err": float(ehat[1]) if ehat is not None else np.nan,
        "accel_err": p2e,
        "rmse": float(rmse), "nobs": int(mask.sum()), "tspan": dt_span,
    }


# ------------------------------------------------------------------ por tile
def _height_column(df: pd.DataFrame, logger) -> str:
    # prioridade: h_res (slope-referenciado) > h_corr (maré/DAC) > h_elv (cru)
    if "h_res" in df.columns:
        return "h_res"
    if "h_corr" in df.columns:
        logger.warning("sem 'h_res' (slope) — usando 'h_corr'.")
        return "h_corr"
    logger.warning("sem 'h_res'/'h_corr' — usando 'h_elv' (SEM correção).")
    return "h_elv"


def compute_tile_dhdt(df: pd.DataFrame, cfg: Config,
                      x_min: float, x_max: float,
                      y_min: float, y_max: float) -> pd.DataFrame:
    """
    Calcula dh/dt nos nós do NÚCLEO [x_min,x_max]×[y_min,y_max], usando todos
    os pontos de `df` (núcleo + halo) como vizinhança. Retorna DataFrame de nós.
    """
    logger = get_logger()
    df = assign_xy(df, cfg)
    hcol = _height_column(df, logger)
    d = cfg.dhdt

    ok = ~(df["x"].isna() | df["y"].isna() | df[hcol].isna() | df["t_year"].isna())
    x = df["x"].to_numpy()[ok]
    y = df["y"].to_numpy()[ok]
    h = df[hcol].to_numpy()[ok].astype(float)
    t = df["t_year"].to_numpy()[ok]
    s = df["s_elv"].to_numpy()[ok].astype(float) if "s_elv" in df.columns else np.full(ok.sum(), 0.05)

    if len(h) < d.min_points:
        return _empty_nodes()

    step = d.node_spacing_m
    gx = np.arange(x_min + step / 2, x_max, step)
    gy = np.arange(y_min + step / 2, y_max, step)
    if len(gx) == 0 or len(gy) == 0:
        return _empty_nodes()
    GX, GY = np.meshgrid(gx, gy)
    nodes_x, nodes_y = GX.ravel(), GY.ravel()

    tree = cKDTree(np.c_[x, y])
    rows = []
    for nx, ny in zip(nodes_x, nodes_y):
        idx = tree.query_ball_point([nx, ny], r=d.search_radius_m)
        if len(idx) < d.min_points:
            continue
        idx = np.asarray(idx)
        res = _fit_node(x[idx], y[idx], h[idx], t[idx], s[idx], nx, ny, d)
        if res is None:
            continue
        res["x"], res["y"] = float(nx), float(ny)
        rows.append(res)

    if not rows:
        return _empty_nodes()

    out = pd.DataFrame(rows)
    lon, lat = to_lonlat(out["x"].to_numpy(), out["y"].to_numpy(), cfg)
    out["lon"], out["lat"] = lon, lat
    return out[_NODE_COLUMNS]


def compute_tile_dhdt_windows(
    df: pd.DataFrame,
    cfg: Config,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    windows,
    accepted_nodes=None,
):
    """Calcula várias janelas reutilizando coordenadas, árvore e vizinhanças.

    É numericamente equivalente a chamar :func:`compute_tile_dhdt` para cada
    recorte temporal, mas evita reconstruir a mesma estrutura espacial. Quando
    ``accepted_nodes`` é fornecido, avalia somente os nós do QC comum.
    """
    logger = get_logger()
    df = assign_xy(df, cfg)
    hcol = _height_column(df, logger)
    d = cfg.dhdt

    ok = ~(
        df["x"].isna()
        | df["y"].isna()
        | df[hcol].isna()
        | df["t_year"].isna()
    )
    x = df["x"].to_numpy()[ok]
    y = df["y"].to_numpy()[ok]
    h = df[hcol].to_numpy()[ok].astype(float)
    t = df["t_year"].to_numpy()[ok]
    s = (
        df["s_elv"].to_numpy()[ok].astype(float)
        if "s_elv" in df.columns
        else np.full(ok.sum(), 0.05)
    )
    empty = {window: _empty_nodes() for window in windows}
    if len(h) < d.min_points:
        return empty

    step = d.node_spacing_m
    gx = np.arange(x_min + step / 2, x_max, step)
    gy = np.arange(y_min + step / 2, y_max, step)
    if len(gx) == 0 or len(gy) == 0:
        return empty
    grid_x, grid_y = np.meshgrid(gx, gy)
    nodes_x, nodes_y = grid_x.ravel(), grid_y.ravel()

    if accepted_nodes is not None:
        accepted = set(accepted_nodes)
        keep = np.fromiter(
            (
                (int(round(nx)), int(round(ny))) in accepted
                for nx, ny in zip(nodes_x, nodes_y)
            ),
            dtype=bool,
            count=len(nodes_x),
        )
        nodes_x, nodes_y = nodes_x[keep], nodes_y[keep]

    tree = cKDTree(np.c_[x, y])
    rows = {window: [] for window in windows}
    for nx, ny in zip(nodes_x, nodes_y):
        indices = tree.query_ball_point([nx, ny], r=d.search_radius_m)
        if len(indices) < d.min_points:
            continue
        indices = np.asarray(indices)
        local_t = t[indices]
        for window in windows:
            start, end = window
            temporal = (local_t >= start) & (local_t < end)
            if temporal.sum() < d.min_points:
                continue
            selected = indices[temporal]
            result = _fit_node(
                x[selected],
                y[selected],
                h[selected],
                t[selected],
                s[selected],
                nx,
                ny,
                d,
            )
            if result is None:
                continue
            result["x"], result["y"] = float(nx), float(ny)
            rows[window].append(result)

    outputs = {}
    for window in windows:
        if not rows[window]:
            outputs[window] = _empty_nodes()
            continue
        out = pd.DataFrame(rows[window])
        lon, lat = to_lonlat(out["x"].to_numpy(), out["y"].to_numpy(), cfg)
        out["lon"], out["lat"] = lon, lat
        outputs[window] = out[_NODE_COLUMNS]
    return outputs


_NODE_COLUMNS = ["x", "y", "lon", "lat", "p0", "dhdt", "accel",
                 "p0_err", "dhdt_err", "accel_err", "rmse", "nobs", "tspan"]


def _empty_nodes() -> pd.DataFrame:
    return pd.DataFrame({c: np.array([], dtype="float64") for c in _NODE_COLUMNS})


# ------------------------------------------------------------------ pipeline
def run_dhdt(cfg: Config) -> pd.DataFrame:
    """Roda dh/dt em todos os tiles do manifesto; grava e retorna os nós."""
    logger = get_logger()
    cfg.paths.dhdt_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cfg)
    logger.info(f"dh/dt em {len(manifest)} tiles "
                f"(raio {d_km(cfg.dhdt.search_radius_m)} km, nós {d_km(cfg.dhdt.node_spacing_m)} km, "
                f"min_points {cfg.dhdt.min_points})")

    all_nodes = []
    for entry in manifest:
        tile_path = cfg.paths.tiles_dir / entry["file"]
        tdf = pd.read_parquet(tile_path, engine="pyarrow")
        nodes = compute_tile_dhdt(
            tdf, cfg, entry["x_min"], entry["x_max"], entry["y_min"], entry["y_max"])
        if len(nodes) == 0:
            continue
        nodes.to_parquet(cfg.paths.dhdt_dir / f"{entry['tile']}_dhdt.parquet",
                         index=False, engine="pyarrow", compression="snappy")
        all_nodes.append(nodes)
        logger.info(f"  {entry['tile']}: {len(nodes):,} nós  "
                    f"dh/dt médio {nodes['dhdt'].mean():+.4f} m/ano")

    if not all_nodes:
        logger.warning("Nenhum nó dh/dt válido.")
        return _empty_nodes()

    merged = pd.concat(all_nodes, ignore_index=True)
    out = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
    merged.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    logger.info(f"dh/dt total: {len(merged):,} nós -> {out} "
                f"(mediana {merged['dhdt'].median():+.4f} m/ano)")
    return merged


def d_km(m: float) -> float:
    return m / 1000.0
