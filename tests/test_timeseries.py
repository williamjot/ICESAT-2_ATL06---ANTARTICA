"""
Testes da série temporal por nó e do teste de tendência formal.
Inclui corretude: recuperar elevações por ano e uma tendência conhecida,
e checar que o FDR distingue sinal real de ruído.
"""

import numpy as np
import pandas as pd

from thwaites.timeseries.build import build_node_series, SERIES_COLUMNS
from thwaites.timeseries.trend import (
    mann_kendall_sen, seasonal_mann_kendall_sen, compute_trends,
)


# --------------------------------------------------------------- build série
def _synthetic_points(slope=-0.4, seed=0, n=6000):
    """Pontos num tile 50x50 km, h = 400 + slope*(ano-2022) + gradiente + ruído."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 50_000, n)
    y = rng.uniform(0, 50_000, n)
    years = rng.integers(2019, 2026, n)
    spatial = 2e-4 * (x - 25_000)
    h = 400 + slope * (years - 2022) + spatial + rng.normal(0, 0.05, n)
    return pd.DataFrame({
        "x": x, "y": y, "h_corr": h.astype("float32"),
        "t_year": (years + 0.6).astype("float64"),
        "s_elv": np.full(n, 0.1, dtype="float32"),
    })


def test_build_series_shape_and_years(cfg):
    s = build_node_series(_synthetic_points(), cfg, 0.0, 50_000.0, 0.0, 50_000.0)
    assert len(s) > 0
    assert list(s.columns) == SERIES_COLUMNS
    # anos dentro do período configurado
    assert s["year"].min() >= cfg.temporal.year_start
    assert s["year"].max() <= cfg.temporal.year_end
    # cada nó tem no máximo um registro por ano
    per = s.groupby(["node_x", "node_y", "year", "month"]).size()
    assert (per == 1).all()


def test_series_recovers_yearly_elevation(cfg):
    # sinal sem gradiente: elevação de cada ano ~ 400 + slope*(ano-2022)
    df = _synthetic_points(slope=-0.4, seed=1)
    df["h_corr"] = (400 - 0.4 * (np.floor(df["t_year"]) - 2022)).astype("float32")
    s = build_node_series(df, cfg, 0.0, 50_000.0, 0.0, 50_000.0)
    # pega um nó e confere a série
    node = s[(s["node_x"] == s["node_x"].iloc[0]) & (s["node_y"] == s["node_y"].iloc[0])]
    expected = 400 - 0.4 * (node["year"].to_numpy() - 2022)
    assert np.allclose(node["h_node"].to_numpy(), expected, atol=1e-3)


# --------------------------------------------------------------- tendência
def test_mann_kendall_sen_recovers_slope():
    years = np.arange(2019, 2026)
    values = 100 - 0.5 * (years - 2022) + np.array([0.01, -0.01, 0, 0.01, -0.01, 0, 0.01])
    r = mann_kendall_sen(years, values, alpha=0.05)
    assert np.isclose(r["sens_slope"], -0.5, atol=0.05)
    assert r["p_value"] < 0.05
    assert r["trend"] == "decrescente"


def test_seasonal_mann_kendall_removes_monthly_cycle():
    years = np.repeat(np.arange(2019, 2026), 12)
    months = np.tile(np.arange(1, 13), 7)
    values = 100 - 0.35 * (years - 2022) + 4.0 * np.sin(2 * np.pi * months / 12)
    r = seasonal_mann_kendall_sen(years, months, values, alpha=0.05)
    assert np.isclose(r["sens_slope"], -0.35, atol=1e-8)
    assert r["p_value"] < 0.05
    assert r["trend"] == "decrescente"


def test_compute_trends_annual_profile_uses_month_strata(cfg_anual):
    years = np.repeat(np.arange(2019, 2026), 12)
    months = np.tile(np.arange(1, 13), 7)
    values = 50 - 0.25 * (years - 2022) + 3 * np.cos(2 * np.pi * months / 12)
    series = pd.DataFrame({
        "node_x": 0.0, "node_y": 0.0, "lon": -100.0, "lat": -75.0,
        "year": years, "month": months, "h_node": values,
        "sigma": 0.1, "n_obs": 20,
    })[SERIES_COLUMNS]
    out = compute_trends(series, cfg_anual)
    assert len(out) == 1
    assert np.isclose(out.iloc[0]["sens_slope"], -0.25, atol=1e-8)
    assert bool(out.iloc[0]["significant"]) is True


def test_compute_trends_fdr_separates_signal_from_noise(cfg):
    # nó A: tendência forte; nó B: ruído puro
    yrs = np.arange(2019, 2026)
    rng = np.random.default_rng(0)
    rowsA = [{"node_x": 0.0, "node_y": 0.0, "lon": -100.0, "lat": -75.0,
              "year": y, "month": 0,
              "h_node": 100 - 0.6 * (y - 2022) + rng.normal(0, 0.02),
              "sigma": 0.1, "n_obs": 20} for y in yrs]
    rowsB = [{"node_x": 1.0, "node_y": 1.0, "lon": -100.1, "lat": -75.1,
              "year": y, "month": 0, "h_node": 50 + rng.normal(0, 0.5),
              "sigma": 0.1, "n_obs": 20} for y in yrs]
    series = pd.DataFrame(rowsA + rowsB)[SERIES_COLUMNS]

    trends = compute_trends(series, cfg)
    assert len(trends) == 2
    a = trends[(trends["node_x"] == 0.0)].iloc[0]
    b = trends[(trends["node_x"] == 1.0)].iloc[0]
    assert np.isclose(a["sens_slope"], -0.6, atol=0.1)
    assert bool(a["significant"]) is True
    assert bool(b["significant"]) is False
    assert "p_fdr" in trends.columns
