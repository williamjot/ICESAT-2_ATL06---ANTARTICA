"""
Testes da correção de incerteza por jackknife sobre anos.

O teste central: com variabilidade interanual real, o jackknife tem de dar uma
incerteza muito maior que o erro formal quando há dependência temporal.
"""

import numpy as np
import pandas as pd

from thwaites.timeseries.uncertainty import (
    jackknife_rate_uncertainty, add_jackknife_uncertainty,
)

T_REF = 2022.0


def _node_obs(dhdt=-0.5, year_offsets=None, n_per_year=200, noise=0.05, seed=0):
    """
    Observações de um nó. `year_offsets` injeta variabilidade INTERANUAL:
    cada ano ganha um deslocamento comum a todas as suas observações — que é
    exactamente o que o erro formal ignora.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2019, 2026)
    offs = year_offsets if year_offsets is not None else np.zeros(len(years))
    t, h, dx, dy = [], [], [], []
    for y, off in zip(years, offs):
        tt = y + 0.6 + rng.normal(0, 0.01, n_per_year)
        t.append(tt)
        h.append(500 + dhdt * (tt - T_REF) + off + rng.normal(0, noise, n_per_year))
        dx.append(rng.uniform(-1e4, 1e4, n_per_year))
        dy.append(rng.uniform(-1e4, 1e4, n_per_year))
    return (np.concatenate(t), np.concatenate(h),
            np.concatenate(dx), np.concatenate(dy))


def test_recovers_rate_and_returns_both_errors():
    t, h, dx, dy = _node_obs(dhdt=-0.5, seed=1)
    rate, err_j, k, err_f = jackknife_rate_uncertainty(t, h, dx, dy, T_REF)
    assert np.isclose(rate, -0.5, atol=0.02)
    assert k == 7
    assert np.isfinite(err_j) and np.isfinite(err_f)


def test_jackknife_much_larger_with_interannual_variability():
    """
    Com deslocamento anual comum (variabilidade interanual), o erro formal
    permanece minúsculo, mas o jackknife cresce e representa a dependência de
    forma mais realista.
    """
    rng = np.random.default_rng(2)
    offs = rng.normal(0, 0.30, 7)          # 30 cm de variabilidade entre anos
    t, h, dx, dy = _node_obs(dhdt=-0.5, year_offsets=offs, seed=2)
    rate, err_j, k, err_f = jackknife_rate_uncertainty(t, h, dx, dy, T_REF)
    assert err_j > 5 * err_f, f"inflação insuficiente: {err_j/err_f:.1f}x"


def test_no_interannual_variability_gives_small_inflation():
    """Sem variabilidade interanual, jackknife e formal ficam comparáveis."""
    t, h, dx, dy = _node_obs(dhdt=-0.5, noise=0.05, seed=3)
    rate, err_j, k, err_f = jackknife_rate_uncertainty(t, h, dx, dy, T_REF)
    assert err_j / err_f < 5.0


def test_too_few_years_returns_nan_uncertainty():
    rng = np.random.default_rng(4)
    t = np.repeat([2020.6, 2021.6], 100) + rng.normal(0, 0.01, 200)
    h = 500 - 0.5 * (t - T_REF) + rng.normal(0, 0.05, 200)
    dx = rng.uniform(-1e4, 1e4, 200); dy = rng.uniform(-1e4, 1e4, 200)
    rate, err_j, k, err_f = jackknife_rate_uncertainty(t, h, dx, dy, T_REF,
                                                      min_years=3)
    assert np.isfinite(rate)
    assert np.isnan(err_j)          # não inventa incerteza sem anos suficientes
    assert k == 2


def test_add_jackknife_uncertainty_on_frame(cfg):
    cfg.dhdt.search_radius_m = 30_000.0
    cfg.dhdt.min_points = 20
    rng = np.random.default_rng(5)
    n = 4000
    x = rng.uniform(0, 40_000, n)
    y = rng.uniform(0, 40_000, n)
    year = rng.choice(np.arange(2019, 2026), n)
    t = year + 0.6
    offs = {y_: o for y_, o in zip(np.arange(2019, 2026), rng.normal(0, 0.25, 7))}
    h = 500 - 0.5 * (t - T_REF) + np.array([offs[int(v)] for v in year]) \
        + rng.normal(0, 0.05, n)
    points = pd.DataFrame({"x": x, "y": y, "t_year": t.astype(float),
                           "h_res": h.astype("float32"),
                           "s_elv": np.full(n, 0.1, dtype="float32")})
    nodes = pd.DataFrame({"x": [20_000.0], "y": [20_000.0],
                          "dhdt": [-0.5], "dhdt_err": [0.001]})
    out = add_jackknife_uncertainty(points, nodes, cfg)
    assert "dhdt_err_jack" in out.columns and "err_inflation" in out.columns
    assert np.isfinite(out.iloc[0]["dhdt_err_jack"])
    assert out.iloc[0]["err_inflation"] > 1.0
