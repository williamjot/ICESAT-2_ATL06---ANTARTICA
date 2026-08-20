"""
Teste de corretude numérica do dh/dt: gera pontos com uma taxa conhecida e
verifica que o ajuste a recupera.
"""

import numpy as np
import pandas as pd

from thwaites.timeseries.dhdt import compute_tile_dhdt


def _synthetic_points(slope=-0.5, t_ref=2022.0, seed=0, n=3000):
    """
    Pontos num tile de 50x50 km com sinal h = h0 + slope*(t-t_ref) + gradiente
    espacial leve + ruído pequeno. Anos 2019..2025 (amostra de inverno).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 50_000, n)
    y = rng.uniform(0, 50_000, n)
    years = rng.integers(2019, 2026, n)          # 2019..2025
    t = years + 0.6                              # ~inverno austral
    h0 = 500.0
    spatial = 1e-4 * (x - 25_000) + 5e-5 * (y - 25_000)   # gradiente leve
    noise = rng.normal(0, 0.05, n)
    h = h0 + slope * (t - t_ref) + spatial + noise
    return pd.DataFrame({
        "x": x, "y": y, "h_corr": h.astype("float32"),
        "t_year": t.astype("float64"), "s_elv": np.full(n, 0.1, dtype="float32"),
    })


def test_recovers_known_dhdt(cfg):
    df = _synthetic_points(slope=-0.5)
    nodes = compute_tile_dhdt(df, cfg, 0.0, 50_000.0, 0.0, 50_000.0)
    assert len(nodes) > 0
    # a mediana dos nós deve recuperar a taxa imposta
    assert np.isclose(nodes["dhdt"].median(), -0.5, atol=0.05)
    # aceleração ~0 (sinal puramente linear)
    acc = nodes["accel"].to_numpy()
    acc = acc[~np.isnan(acc)]
    if len(acc):
        assert np.abs(np.median(acc)) < 0.1
    # respeitou o mínimo de observações por nó
    assert (nodes["nobs"] >= cfg.dhdt.min_points).all()


def test_recovers_positive_dhdt(cfg):
    df = _synthetic_points(slope=0.3, seed=7)
    nodes = compute_tile_dhdt(df, cfg, 0.0, 50_000.0, 0.0, 50_000.0)
    assert len(nodes) > 0
    assert np.isclose(nodes["dhdt"].median(), 0.3, atol=0.05)


def test_insufficient_points_returns_empty(cfg):
    df = _synthetic_points(n=10)          # < min_points
    nodes = compute_tile_dhdt(df, cfg, 0.0, 50_000.0, 0.0, 50_000.0)
    assert len(nodes) == 0
    assert list(nodes.columns)[:4] == ["x", "y", "lon", "lat"]
