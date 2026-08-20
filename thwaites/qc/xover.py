"""
thwaites.qc.xover
=================
Análise de CRUZAMENTOS (crossovers) — inspirada no `xover.py` do captoolkit.

Num cruzamento, uma trilha ascendente e uma descendente passam pelo MESMO ponto
em tempos diferentes. A diferença de elevação ali dá:

  1. **dh/dt independente**: dh/dt = (h₂ − h₁)/(t₂ − t₁), calculado sem nenhum
     ajuste de superfície. É um estimador metodologicamente INDEPENDENTE do
     `fitsec` — serve para validar (ou refutar) o dh/dt do pipeline principal.
  2. **Viés inter-feixe / inter-trilha**: em cruzamentos quase-simultâneos
     (|Δt| pequeno), diferença sistemática de elevação entre pares de feixes é
     viés instrumental, não mudança de gelo.

Método: cada trilha vira uma LineString; um índice espacial (STRtree) acha os
pares que se cruzam; em cada interseção a elevação de cada trilha é obtida por
ajuste de PLANO LOCAL aos pontos vizinhos (não vizinho-mais-próximo — um offset
de ~10 m sobre terreno inclinado introduziria erro comparável ao sinal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger

XOVER_COLUMNS = ["x", "y", "t1", "t2", "dt", "h1", "h2", "dh", "dh_err",
                 "dhdt", "dhdt_err",
                 "beam1", "beam2", "track1", "track2", "n1", "n2"]


def _height_column(df: pd.DataFrame) -> str:
    for c in ("h_res", "h_corr", "h_elv"):
        if c in df.columns:
            return c
    raise ValueError("nenhuma coluna de elevação (h_res/h_corr/h_elv)")


def classify_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Rotula cada trilha como ascendente/descendente pelo sinal de Δy."""
    g = df.groupby("track_id")
    y_first = g["y"].first().to_numpy()
    y_last = g["y"].last().to_numpy()
    return pd.DataFrame({
        "track_id": g["y"].first().index.to_numpy(),
        "ascending": (y_last - y_first) > 0,
        "n_pts": g.size().to_numpy(),
    })


def _plane_value(xs, ys, hs, x0, y0):
    """
    Ajusta h = a + b·dx + c·dy e avalia no cruzamento.

    Retorna (a, var_a) — a elevação no ponto E sua variância formal, para que a
    incerteza do dh/dt do crossover possa ser PROPAGADA (§6.3). Sem isso, o
    crossover produz um número sem barra de erro e a comparação com o ajuste
    local não pode ser normalizada.
    """
    n = len(hs)
    if n < 3:
        return (float(np.mean(hs)) if n else np.nan), np.nan
    A = np.column_stack([np.ones(n), xs - x0, ys - y0])
    try:
        coef, *_ = np.linalg.lstsq(A, hs, rcond=None)
    except np.linalg.LinAlgError:
        return float(np.median(hs)), np.nan
    resid = hs - A @ coef
    dof = max(n - 3, 1)
    sigma2 = float(np.sum(resid ** 2) / dof)
    try:
        cov00 = float(np.linalg.inv(A.T @ A)[0, 0])
        var_a = max(sigma2 * cov00, 0.0)
    except np.linalg.LinAlgError:
        var_a = np.nan
    return float(coef[0]), float(var_a)


def find_crossovers(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Encontra cruzamentos ascendente×descendente e calcula dh/dt em cada um.

    `df` precisa de: track_id, x, y, t_year, beam + coluna de elevação.
    `track_id` é criado por run_filttrack.py.
    """
    from shapely.geometry import LineString, Point
    from shapely.strtree import STRtree
    from scipy.spatial import cKDTree

    logger = get_logger()
    xo = cfg.xover
    if "track_id" not in df.columns:
        raise ValueError("xover exige a coluna 'track_id' (rode run_filttrack.py antes).")

    hcol = _height_column(df)
    d = df[["track_id", "x", "y", "t_year", "beam", hcol]].dropna().copy()
    d = d.sort_values(["track_id", "t_year"]).reset_index(drop=True)

    info = classify_tracks(d)
    info = info[info["n_pts"] >= xo.min_track_points]
    asc_ids = info.loc[info["ascending"], "track_id"].to_numpy()
    des_ids = info.loc[~info["ascending"], "track_id"].to_numpy()
    n_asc_all, n_des_all = len(asc_ids), len(des_ids)

    # Amostragem ALEATÓRIA de trilhas (não "as mais longas", que enviesaria
    # espacialmente para trilhas que cruzam a ROI inteira). Limita o custo
    # O(n_asc × n_desc) mantendo a amostra representativa.
    cap = xo.max_tracks_per_direction
    if cap is not None:
        rng = np.random.default_rng(xo.seed)
        if len(asc_ids) > cap:
            asc_ids = rng.choice(asc_ids, cap, replace=False)
        if len(des_ids) > cap:
            des_ids = rng.choice(des_ids, cap, replace=False)
        # restringe também os pontos, para não carregar trilhas não usadas
        keep = np.concatenate([asc_ids, des_ids])
        d = d[d["track_id"].isin(set(keep.tolist()))].reset_index(drop=True)

    logger.info(f"xover: {len(asc_ids)}/{n_asc_all} ascendentes, "
                f"{len(des_ids)}/{n_des_all} descendentes "
                f"(≥{xo.min_track_points} pts; amostra aleatória seed={xo.seed})")
    if len(asc_ids) == 0 or len(des_ids) == 0:
        return pd.DataFrame({c: [] for c in XOVER_COLUMNS})

    # geometria simplificada por trilha (subamostrada — basta p/ localizar a interseção)
    usable = set(info["track_id"].tolist())      # set: lookup O(1) no laço
    lines: dict[int, "LineString"] = {}
    for tid, g in d.groupby("track_id"):
        if tid not in usable:
            continue
        pts = g[["x", "y"]].to_numpy()[:: xo.subsample]
        if len(pts) >= 2:
            lines[int(tid)] = LineString(pts)

    des_list = [(t, lines[t]) for t in des_ids if t in lines]
    if not des_list:
        return pd.DataFrame({c: [] for c in XOVER_COLUMNS})
    tree = STRtree([g for _, g in des_list])
    des_ids_arr = np.array([t for t, _ in des_list])

    # KDTree global para o ajuste local de plano
    xy = d[["x", "y"]].to_numpy()
    kdt = cKDTree(xy)
    tid_arr = d["track_id"].to_numpy()
    t_arr = d["t_year"].to_numpy()
    h_arr = d[hcol].to_numpy(dtype=float)
    beam_arr = d["beam"].to_numpy()

    rows = []
    for ta in asc_ids:
        la = lines.get(int(ta))
        if la is None:
            continue
        for j in tree.query(la):                    # candidatos por bbox
            lb = des_list[int(j)][1]
            inter = la.intersection(lb)
            if inter.is_empty:
                continue
            tb = int(des_ids_arr[int(j)])
            geoms = [inter] if isinstance(inter, Point) else list(getattr(inter, "geoms", []))
            for gpt in geoms:
                if not isinstance(gpt, Point):
                    continue
                x0, y0 = float(gpt.x), float(gpt.y)
                idx = np.asarray(kdt.query_ball_point([x0, y0], r=xo.fit_radius_m), dtype=int)
                if idx.size == 0:
                    continue
                ia = idx[tid_arr[idx] == ta]
                ib = idx[tid_arr[idx] == tb]
                if len(ia) < xo.min_points or len(ib) < xo.min_points:
                    continue
                h1, v1 = _plane_value(xy[ia, 0], xy[ia, 1], h_arr[ia], x0, y0)
                h2, v2 = _plane_value(xy[ib, 0], xy[ib, 1], h_arr[ib], x0, y0)
                t1 = float(np.median(t_arr[ia]))
                t2 = float(np.median(t_arr[ib]))
                if not (np.isfinite(h1) and np.isfinite(h2)):
                    continue
                dt = t2 - t1
                dh = h2 - h1
                # propagação da covariância dos dois ajustes locais (§6.3):
                #   var(dh) = var(h1) + var(h2)   (ajustes independentes)
                #   var(dh/dt) = var(dh) / dt²
                var_dh = (v1 + v2) if (np.isfinite(v1) and np.isfinite(v2)) else np.nan
                if abs(dt) >= xo.min_dt_years:
                    rate = dh / dt
                    rate_err = float(np.sqrt(var_dh) / abs(dt)) if np.isfinite(var_dh) else np.nan
                else:
                    rate = rate_err = np.nan
                rows.append((x0, y0, t1, t2, dt, h1, h2, dh,
                             float(np.sqrt(var_dh)) if np.isfinite(var_dh) else np.nan,
                             rate, rate_err,
                             int(np.median(beam_arr[ia])), int(np.median(beam_arr[ib])),
                             int(ta), tb, len(ia), len(ib)))

    if not rows:
        logger.warning("xover: nenhum cruzamento encontrado.")
        return pd.DataFrame({c: [] for c in XOVER_COLUMNS})

    out = pd.DataFrame(rows, columns=XOVER_COLUMNS)
    # rejeita taxas fisicamente implausíveis (mesmo limite do pipeline principal)
    out.loc[out["dhdt"].abs() > cfg.dhdt.rate_limit, "dhdt"] = np.nan
    valid = out["dhdt"].notna()
    logger.info(
        f"xover: {len(out):,} cruzamentos | {int(valid.sum()):,} com |Δt| ≥ "
        f"{xo.min_dt_years} ano | dh/dt mediana {out.loc[valid,'dhdt'].median():+.3f} m/ano")
    return out


def interbeam_bias(xovers: pd.DataFrame, max_dt_years: float = 0.25,
                   expected_dhdt: float | None = None) -> pd.DataFrame:
    """
    Viés entre pares de feixes usando cruzamentos quase-simultâneos.

    ATENÇÃO (§6.3): "a mudança real de elevação durante o intervalo NÃO deve ser
    assumida como zero em uma geleira dinâmica". Com dh/dt ≈ −0,5 m/ano e
    max_dt_years = 0,25, a mudança real esperada é ~0,125 m — **maior** que os
    ~0,03 m de viés tipicamente medido. Ignorar isso contamina a estimativa.

    Por isso `expected_dhdt` permite REMOVER a mudança esperada antes de estimar
    o viés:  Δh_corrigido = Δh − dh/dt_esperado · Δt.
    Passe a mediana de dh/dt da região (ou por ponto, se disponível).

    Sempre reporte junto a sensibilidade a `max_dt_years` (ver
    `interbeam_bias_sensitivity`).
    """
    near = xovers[xovers["dt"].abs() <= max_dt_years].copy()
    cols = ["beam1", "beam2", "n", "bias_m", "mad_m", "max_dt_years",
            "expected_change_removed"]
    if near.empty:
        return pd.DataFrame(columns=cols)

    if expected_dhdt is not None:
        near["dh_adj"] = near["dh"] - expected_dhdt * near["dt"]
    else:
        near["dh_adj"] = near["dh"]

    g = near.groupby(["beam1", "beam2"])["dh_adj"]
    out = pd.DataFrame({
        "n": g.size(),
        "bias_m": g.median(),
        "mad_m": g.apply(lambda s: 1.4826 * np.median(np.abs(s - s.median()))),
    }).reset_index()
    out["max_dt_years"] = max_dt_years
    out["expected_change_removed"] = (expected_dhdt is not None)
    return out


def interbeam_bias_sensitivity(xovers: pd.DataFrame,
                               windows=(0.05, 0.1, 0.25, 0.5),
                               expected_dhdt: float | None = None) -> pd.DataFrame:
    """
    Sensibilidade do viés inter-feixe à janela temporal (§6.3).

    Se o "viés" cresce com a janela, ele não é instrumental — é mudança real de
    elevação sendo mal atribuída.
    """
    rows = []
    for w in windows:
        b = interbeam_bias(xovers, max_dt_years=w, expected_dhdt=expected_dhdt)
        if b.empty:
            continue
        rows.append({"max_dt_years": w, "n_pairs": int(b["n"].sum()),
                     "max_abs_bias_m": float(b["bias_m"].abs().max()),
                     "median_abs_bias_m": float(b["bias_m"].abs().median())})
    return pd.DataFrame(rows)
