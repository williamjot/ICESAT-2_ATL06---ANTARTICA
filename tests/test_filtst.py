"""Testes do filtro espaço-temporal (binagem)."""

import numpy as np
import pandas as pd

from thwaites.qc.filtst import filter_space_time


def _cloud(n=4000, seed=0, extent=20_000.0):
    """Nuvem de pontos com elevação suave em 4 anos."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.uniform(0, extent, n),
        "y": rng.uniform(0, extent, n),
        "t_year": rng.choice([2020.5, 2021.5, 2022.5, 2023.5], n),
        "h_res": rng.normal(0.0, 0.05, n).astype("float32"),
    })


def test_filtst_disabled_is_noop(cfg):
    df = _cloud()
    out = filter_space_time(df, cfg)          # enabled=False por padrão
    assert len(out) == len(df)


def test_filtst_removes_spikes(cfg):
    cfg.filtst.enabled = True
    df = _cloud(seed=1)
    # injeta 20 outliers grosseiros
    idx = np.arange(0, 20)
    df.loc[idx, "h_res"] = np.float32(15.0)
    out = filter_space_time(df, cfg)
    assert len(out) < len(df)
    assert out["h_res"].max() < 5.0           # os spikes sumiram


def test_filtst_preserves_clean_data(cfg):
    cfg.filtst.enabled = True
    df = _cloud(n=6000, seed=2)
    out = filter_space_time(df, cfg)
    assert len(out) >= 0.98 * len(df)         # sem super-rejeição


def test_filtst_requires_xy(cfg):
    cfg.filtst.enabled = True
    df = _cloud().drop(columns=["x"])
    import pytest
    with pytest.raises(ValueError):
        filter_space_time(df, cfg)


def test_filtst_sparse_cells_not_filtered(cfg):
    """Células com poucos pontos não têm estatística robusta -> não filtram."""
    cfg.filtst.enabled = True
    cfg.filtst.min_count = 1000               # nenhuma célula atinge
    df = _cloud(seed=3)
    df.loc[0, "h_res"] = np.float32(50.0)
    out = filter_space_time(df, cfg)
    assert len(out) == len(df)
