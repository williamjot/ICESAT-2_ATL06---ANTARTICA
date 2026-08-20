"""
thwaites.timeseries.build
=========================
Constrói a série temporal de elevação por nó, além da taxa agregada.

Para cada nó da grade e cada ANO do período, seleciona os pontos daquele ano
dentro do raio de busca e ajusta um plano local `h = a + b·dx + c·dy`; a
elevação representativa do nó naquele ano é `a` (o plano avaliado no centro),
com incerteza formal. O plano remove a inclinação local da célula SEM precisar
de REMA (usa os próprios dados) — coerente com "sem slope/REMA por agora".

Saída (formato longo, tidy): uma linha por (nó, ano) com elevação e incerteza.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.grid.reproject import to_lonlat
from thwaites.grid.tiles import assign_xy
from thwaites.logging import get_logger

SERIES_COLUMNS = ["node_x", "node_y", "lon", "lat", "year", "month",
                  "h_node", "sigma", "n_obs"]


def _month_from_decimal_year(t_year: np.ndarray) -> np.ndarray:
    """Converte anos decimais ATL06 em meses calendáricos (1..12).

    O arquivo leve preserva ``t_year``, não o timestamp bruto. A conversão
    respeita a duração de cada ano, evitando ``floor(frac * 12)``, que desloca
    observações perto das fronteiras dos meses.
    """
    import calendar

    t = np.asarray(t_year, dtype=float)
    years = np.floor(t).astype(int)
    frac = t - years
    out = np.empty(len(t), dtype=np.int8)
    for yr in np.unique(years):
        sel = years == yr
        days = 366 if calendar.isleap(int(yr)) else 365
        doy = np.clip(np.floor(frac[sel] * days).astype(int), 0, days - 1)
        lengths = np.array([31, 29 if calendar.isleap(int(yr)) else 28, 31, 30,
                            31, 30, 31, 31, 30, 31, 30, 31])
        out[sel] = np.searchsorted(np.cumsum(lengths), doy, side="right") + 1
    return out


def _fit_plane_at_node(dx, dy, h):
    """
    Ajusta h = a + b·dx + c·dy; retorna (a, sigma_a).
    `a` é a elevação no nó (dx=dy=0). sigma_a é o erro formal de `a`.
    """
    A = np.column_stack([np.ones_like(dx), dx, dy])
    coef, resid, rank, _ = np.linalg.lstsq(A, h, rcond=None)
    if rank < 3:
        return np.nan, np.nan
    fitted = A @ coef
    dof = max(len(h) - 3, 1)
    sig2 = float(np.sum((h - fitted) ** 2) / dof)
    try:
        cov00 = float(np.linalg.inv(A.T @ A)[0, 0])
    except np.linalg.LinAlgError:
        return float(coef[0]), np.nan
    return float(coef[0]), float(np.sqrt(max(sig2 * cov00, 0.0)))


def build_node_series(df: pd.DataFrame, cfg: Config,
                      x_min: float, x_max: float,
                      y_min: float, y_max: float) -> pd.DataFrame:
    """
    Constrói a série temporal (nó × ano) para o núcleo do tile.
    Usa `h_corr` se presente (cai para `h_elv` com aviso).
    """
    from scipy.spatial import cKDTree

    logger = get_logger()
    ts = cfg.timeseries
    df = assign_xy(df, cfg)
    # prioridade: h_res (slope) > h_corr (maré/DAC) > h_elv (cru)
    hcol = next((c for c in ("h_res", "h_corr", "h_elv") if c in df.columns), "h_elv")
    if hcol == "h_elv":
        logger.warning("sem 'h_res'/'h_corr' — série usando 'h_elv' (sem correção).")

    ok = ~(df["x"].isna() | df["y"].isna() | df[hcol].isna() | df["t_year"].isna())
    x = df["x"].to_numpy()[ok]
    y = df["y"].to_numpy()[ok]
    h = df[hcol].to_numpy()[ok].astype(float)
    t_year = df["t_year"].to_numpy()[ok].astype(float)
    year = np.floor(t_year).astype(int)
    # O MK sazonal exige comparar o mesmo mês entre anos. Agregar antes por
    # ano destruiria essa estrutura e tornaria o teste estatisticamente inválido.
    seasonal = cfg.trend.mk_variant == "seasonal"
    month = (_month_from_decimal_year(t_year) if seasonal
             else np.zeros(len(year), dtype=np.int8))

    if len(h) < ts.min_points_per_epoch:
        return _empty_series()

    step = ts.node_spacing_m
    gx = np.arange(x_min + step / 2, x_max, step)
    gy = np.arange(y_min + step / 2, y_max, step)
    if len(gx) == 0 or len(gy) == 0:
        return _empty_series()
    GX, GY = np.meshgrid(gx, gy)
    nodes_x, nodes_y = GX.ravel(), GY.ravel()

    tree = cKDTree(np.c_[x, y])
    years_all = np.arange(cfg.temporal.year_start, cfg.temporal.year_end + 1)

    rows = []
    for nx, ny in zip(nodes_x, nodes_y):
        idx = np.asarray(tree.query_ball_point([nx, ny], r=ts.search_radius_m), dtype=int)
        if idx.size < ts.min_points_per_epoch:
            continue
        yr_local = year[idx]
        for yr in years_all:
            idx_year = idx[yr_local == yr]
            periods = range(1, 13) if seasonal else (0,)
            for mon in periods:
                sel = idx_year[month[idx_year] == mon] if seasonal else idx_year
                if sel.size < ts.min_points_per_epoch:
                    continue
                a, sig = _fit_plane_at_node(x[sel] - nx, y[sel] - ny, h[sel])
                if np.isnan(a):
                    continue
                rows.append((float(nx), float(ny), int(yr), int(mon), a, sig,
                             int(sel.size)))

    if not rows:
        return _empty_series()

    arr = np.array(rows, dtype=object)
    out = pd.DataFrame({
        "node_x": arr[:, 0].astype(float),
        "node_y": arr[:, 1].astype(float),
        "year":   arr[:, 2].astype(int),
        "month":  arr[:, 3].astype(int),
        "h_node": arr[:, 4].astype(float),
        "sigma":  arr[:, 5].astype(float),
        "n_obs":  arr[:, 6].astype(int),
    })
    lon, lat = to_lonlat(out["node_x"].to_numpy(), out["node_y"].to_numpy(), cfg)
    out["lon"], out["lat"] = lon, lat
    return out[SERIES_COLUMNS]


def _empty_series() -> pd.DataFrame:
    dtypes = {"node_x": float, "node_y": float, "lon": float, "lat": float,
              "year": int, "month": int, "h_node": float, "sigma": float,
              "n_obs": int}
    return pd.DataFrame({c: np.array([], dtype=t) for c, t in dtypes.items()})[SERIES_COLUMNS]
