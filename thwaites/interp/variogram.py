"""
thwaites.interp.variogram
=========================
Variograma empírico e ajuste de modelo — DATA-DRIVEN.

Fornece o alcance (range), patamar (sill) e efeito pepita (nugget) a partir
dos próprios dados, em vez de um lag/modelo fixado por convenção. O modelo
(esférico ou exponencial) é escolhido pelo menor erro de ajuste.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def _spherical(h, nugget, sill, rng):
    h = np.asarray(h, dtype=float)
    out = nugget + (sill - nugget) * (1.5 * h / rng - 0.5 * (h / rng) ** 3)
    out = np.where(h >= rng, sill, out)
    return np.where(h == 0, 0.0, out)


def _exponential(h, nugget, sill, rng):
    h = np.asarray(h, dtype=float)
    out = nugget + (sill - nugget) * (1.0 - np.exp(-h / rng))
    return np.where(h == 0, 0.0, out)


_MODELS = {"spherical": _spherical, "exponential": _exponential}


def empirical_variogram(x, y, values, n_lags=15, max_lag=None, max_points=3000, seed=0):
    """
    Variograma empírico (método dos momentos, binado por lag).

    Subamostra para no máximo `max_points` pontos (pares são O(n²)).
    Retorna (lag_centers, gamma, counts).
    """
    x = np.asarray(x, float); y = np.asarray(y, float); v = np.asarray(values, float)
    n = len(x)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_points, replace=False)
        x, y, v = x[idx], y[idx], v[idx]

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    semiv = 0.5 * (v[:, None] - v[None, :]) ** 2

    iu = np.triu_indices(len(x), k=1)
    d = dist[iu]; g = semiv[iu]

    if max_lag is None:
        max_lag = np.percentile(d, 90)
    edges = np.linspace(0, max_lag, n_lags + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gamma = np.full(n_lags, np.nan)
    counts = np.zeros(n_lags, dtype=int)
    for i in range(n_lags):
        m = (d >= edges[i]) & (d < edges[i + 1])
        counts[i] = int(m.sum())
        if counts[i] > 0:
            gamma[i] = float(np.mean(g[m]))
    ok = ~np.isnan(gamma)
    return centers[ok], gamma[ok], counts[ok]


def fit_variogram(x, y, values, models=("spherical", "exponential"),
                  n_lags=15, max_lag=None, seed=0) -> dict:
    """
    Ajusta modelos ao variograma empírico e retorna o de melhor ajuste (SSE).

    Retorna dict: {model, nugget, sill, range_m, sse, lags, gamma}.
    """
    centers, gamma, counts = empirical_variogram(
        x, y, values, n_lags=n_lags, max_lag=max_lag, seed=seed)
    if len(centers) < 3:
        raise ValueError("variograma empírico com pontos insuficientes para ajuste.")

    sill0 = float(np.nanvar(values)) or float(np.max(gamma))
    rng0 = float(centers[-1] / 2) or 1.0
    best = None
    for name in models:
        f = _MODELS[name]
        try:
            popt, _ = curve_fit(
                f, centers, gamma, p0=[0.0, sill0, rng0],
                bounds=([0, 0, centers[1] if len(centers) > 1 else 1],
                        [sill0 * 3 + 1e-9, sill0 * 5 + 1e-9, centers[-1] * 3]),
                maxfev=10000,
            )
        except Exception:
            continue
        sse = float(np.sum((f(centers, *popt) - gamma) ** 2))
        if best is None or sse < best["sse"]:
            best = {"model": name, "nugget": float(popt[0]), "sill": float(popt[1]),
                    "range_m": float(popt[2]), "sse": sse,
                    "lags": centers, "gamma": gamma}
    if best is None:
        raise RuntimeError("nenhum modelo de variograma convergiu.")
    return best


def make_gamma(params: dict):
    """Retorna a função γ(h) do modelo ajustado (semivariância)."""
    f = _MODELS[params["model"]]
    nugget, sill, rng = params["nugget"], params["sill"], params["range_m"]
    return lambda h: f(h, nugget, sill, rng)
