"""
thwaites.timeseries.uncertainty
===============================
Incerteza DEFENSÁVEL do dh/dt por nó.

O PROBLEMA MEDIDO
-----------------
O erro formal do mínimos quadrados trata as ~10⁵ observações de um nó como
independentes, e devolve `dhdt_err` da ordem de 10⁻³ m/ano — num campo cuja
variabilidade real é ~0,9 m/ano. Duas medições independentes mostraram que essa
incerteza é otimista por uma ordem de magnitude:

  - crossovers × ajuste local (Prioridade 5): |z| mediano ≈ 11,7; só 12% das
    diferenças dentro de 2σ (esperado ~95%);
  - validação temporal sem vazamento (Prioridade 4): z-std ≈ 6,2;
    cobertura de 68% observada = 0,31 (esperado 0,68).

A CAUSA: observações do mesmo sobrevoo/ano não são amostras independentes da
TAXA. A amostra efetiva de uma tendência temporal é o número de épocas
independentes (~7 invernos), não o número de segmentos ATL06.

A CORREÇÃO: **jackknife sobre anos**. Remove-se um ano inteiro por vez, refaz-se
o ajuste, e a dispersão entre as estimativas dá a incerteza:

    var_jack = (k−1)/k · Σᵢ (θᵢ − θ̄)²      com k = nº de anos

É o estimador padrão para dados agrupados e responde exactamente à pergunta
certa: "se os anos observados fossem outros, quanto a taxa mudaria?".

Também é reportado o FATOR DE INFLAÇÃO (err_jack / err_formal), que quantifica
o quanto a incerteza formal subestimava — sem esconder a correção.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.logging import get_logger


def jackknife_rate_uncertainty(t, h, dx, dy, t_ref, weights=None,
                               poly_order: int = 2, min_years: int = 3,
                               min_points: int = 10):
    """
    Incerteza da taxa por jackknife sobre ANOS.

    Retorna (dhdt, err_jackknife, n_years_used, err_formal).
    Devolve NaN em err_jackknife se houver anos insuficientes.
    """
    t = np.asarray(t, float)
    h = np.asarray(h, float)
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    w = None if weights is None else np.asarray(weights, float)

    def _fit(mask):
        n = int(mask.sum())
        if n < min_points:
            return np.nan, np.nan
        d = t[mask] - t_ref
        sx = max(float(np.std(dx[mask])), 1.0)
        sy = max(float(np.std(dy[mask])), 1.0)
        xn, yn = dx[mask] / sx, dy[mask] / sy
        cols = [np.ones(n), d]
        if poly_order >= 1:
            cols += [xn, yn, xn * yn]
        if poly_order >= 2:
            cols += [xn ** 2, yn ** 2]
        A = np.column_stack(cols)
        hh = h[mask]
        if w is not None:
            ww = w[mask]
            ww = ww / (ww.mean() + 1e-30)
            s = np.sqrt(ww)
            Aw, hw = A * s[:, None], hh * s
        else:
            Aw, hw = A, hh
        try:
            coef, *_ = np.linalg.lstsq(Aw, hw, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        resid = hh - A @ coef
        dof = max(n - A.shape[1], 1)
        sig2 = float(np.sum(resid ** 2) / dof)
        try:
            cov = np.linalg.inv(Aw.T @ Aw) * sig2
            err = float(np.sqrt(max(cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            err = np.nan
        return float(coef[1]), err

    all_mask = np.ones(t.size, dtype=bool)
    rate_full, err_formal = _fit(all_mask)
    if not np.isfinite(rate_full):
        return np.nan, np.nan, 0, np.nan

    years = np.unique(np.floor(t))
    k = years.size
    if k < min_years:
        return rate_full, np.nan, k, err_formal

    est = []
    for y in years:
        m = np.floor(t) != y
        r, _ = _fit(m)
        if np.isfinite(r):
            est.append(r)
    if len(est) < min_years:
        return rate_full, np.nan, k, err_formal

    est = np.asarray(est)
    kk = est.size
    var_jack = (kk - 1) / kk * float(np.sum((est - est.mean()) ** 2))
    return rate_full, float(np.sqrt(max(var_jack, 0.0))), int(k), err_formal


def add_jackknife_uncertainty(points: pd.DataFrame, nodes: pd.DataFrame, cfg,
                              height_col: str | None = None) -> pd.DataFrame:
    """
    Recalcula a incerteza dos nós existentes por jackknife sobre anos.

    Não recalcula a TAXA (que já está validada e com viés ~zero) — apenas a
    incerteza, que é o produto defeituoso. Adiciona:
      `dhdt_err_jack`, `dhdt_err_formal`, `err_inflation`, `n_years_node`.
    """
    from scipy.spatial import cKDTree
    from thwaites.grid.tiles import assign_xy

    logger = get_logger()
    d = cfg.dhdt
    points = assign_xy(points, cfg)
    hcol = height_col or next(
        (c for c in ("h_res", "h_corr", "h_elv") if c in points.columns), None)
    if hcol is None:
        raise ValueError("nenhuma coluna de elevação disponível")

    ok = ~(points["x"].isna() | points["y"].isna() |
           points[hcol].isna() | points["t_year"].isna())
    px = points["x"].to_numpy()[ok]
    py = points["y"].to_numpy()[ok]
    ph = points[hcol].to_numpy()[ok].astype(float)
    pt = points["t_year"].to_numpy()[ok]
    if "s_elv" in points.columns:
        s = points["s_elv"].to_numpy()[ok].astype(float)
        med = np.median(s[s > 0]) if np.any(s > 0) else 0.05
        pw = 1.0 / (np.where(s > 0, s, med) ** 2 + 1e-12)
    else:
        pw = None

    tree = cKDTree(np.c_[px, py])
    out = nodes.copy()
    n = len(out)
    err_j = np.full(n, np.nan)
    err_f = np.full(n, np.nan)
    nyr = np.zeros(n, dtype=int)

    nx = out["x"].to_numpy()
    ny = out["y"].to_numpy()
    for i in range(n):
        idx = np.asarray(tree.query_ball_point([nx[i], ny[i]], r=d.search_radius_m),
                         dtype=int)
        if idx.size < d.min_points:
            continue
        _, ej, ky, ef = jackknife_rate_uncertainty(
            pt[idx], ph[idx], px[idx] - nx[i], py[idx] - ny[i], d.t_ref,
            weights=None if pw is None else pw[idx],
            poly_order=d.poly_order, min_points=d.min_points)
        err_j[i], err_f[i], nyr[i] = ej, ef, ky

    out["dhdt_err_jack"] = err_j
    out["dhdt_err_formal"] = err_f
    out["n_years_node"] = nyr
    with np.errstate(invalid="ignore", divide="ignore"):
        out["err_inflation"] = np.where(err_f > 0, err_j / err_f, np.nan)

    good = np.isfinite(err_j) & np.isfinite(err_f) & (err_f > 0)
    if good.any():
        infl = out.loc[good, "err_inflation"]
        logger.info(
            f"incerteza por jackknife: {int(good.sum()):,}/{n:,} nós | "
            f"formal mediano {np.nanmedian(err_f[good]):.5f} -> "
            f"jackknife {np.nanmedian(err_j[good]):.5f} m/ano | "
            f"FATOR DE INFLAÇÃO mediano {infl.median():.1f}× "
            f"(p90 {infl.quantile(0.9):.1f}×)")
    return out
