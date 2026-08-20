"""
Testes da Prioridade 4 (§5): validação sem vazamento de observações.

O teste central (§9): "comprovar que a validação não compartilha observações".
"""

import numpy as np
import pandas as pd
import pytest

from thwaites.validation.folds import (
    Fold, spatial_buffer_folds, track_folds, temporal_folds, verify_no_leakage,
    default_buffer_m,
)
from thwaites.validation.evaluate import (
    fold_metrics, fit_nodes_from_observations, evaluate_fold, summarize_by_method,
)


def _obs(n=4000, extent=120_000.0, seed=0, n_tracks=20):
    """Observações sintéticas com trilhas, anos e um sinal linear conhecido."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, extent, n)
    y = rng.uniform(0, extent, n)
    year = rng.choice(np.arange(2019, 2026), n)
    t = year + 0.6
    h = 500.0 - 0.5 * (t - 2022.0) + 1e-4 * x + rng.normal(0, 0.05, n)
    return pd.DataFrame({
        "x": x, "y": y, "t_year": t.astype(float),
        "track_id": rng.integers(0, n_tracks, n),
        "beam": rng.integers(1, 7, n).astype("int8"),
        "s_elv": np.full(n, 0.1, dtype="float32"),
        "h_res": h.astype("float32"),
    })


# ------------------------------------------------------ ausência de vazamento
def test_spatial_folds_have_no_shared_observations():
    df = _obs()
    folds = spatial_buffer_folds(df["x"], df["y"], block_m=30_000,
                                 n_folds=4, buffer_m=10_000)
    assert len(folds) >= 2
    for f in folds:
        assert verify_no_leakage(f) is True
        assert f.n_train > 0 and f.n_test > 0


def test_buffer_actually_excludes_observations():
    """O buffer precisa RETIRAR observações do treino — senão não é buffer."""
    df = _obs()
    no_buf = spatial_buffer_folds(df["x"], df["y"], 30_000, 4, buffer_m=0.0)
    with_buf = spatial_buffer_folds(df["x"], df["y"], 30_000, 4, buffer_m=15_000)
    assert with_buf[0].info["n_buffer_excluded"] > 0
    assert with_buf[0].n_train < no_buf[0].n_train


def test_buffer_creates_minimum_separation():
    """Nenhuma observação de treino pode estar a menos de buffer do teste."""
    from scipy.spatial import cKDTree
    df = _obs()
    buf = 12_000.0
    folds = spatial_buffer_folds(df["x"], df["y"], 30_000, 4, buffer_m=buf)
    f = folds[0]
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    tree = cKDTree(np.c_[x[f.test], y[f.test]])
    d, _ = tree.query(np.c_[x[f.train], y[f.train]], k=1)
    assert d.min() >= buf - 1e-6


def test_track_folds_hold_out_entire_tracks():
    """Uma trilha retida não pode aparecer no treino (§5.5)."""
    df = _obs()
    folds = track_folds(df["track_id"], n_folds=5, seed=0)
    tid = df["track_id"].to_numpy()
    for f in folds:
        verify_no_leakage(f)
        test_tracks = set(np.unique(tid[f.test]).tolist())
        train_tracks = set(np.unique(tid[f.train]).tolist())
        assert test_tracks.isdisjoint(train_tracks)


def test_temporal_folds_hold_out_entire_years():
    df = _obs()
    folds = temporal_folds(df["t_year"])
    yr = np.floor(df["t_year"].to_numpy())
    for f in folds:
        verify_no_leakage(f)
        held = f.info["year_held_out"]
        assert set(np.unique(yr[f.test]).tolist()) == {held}
        assert held not in set(np.unique(yr[f.train]).tolist())


def test_verify_no_leakage_detects_overlap():
    n = 100
    train = np.ones(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    test[:10] = True                     # sobreposição proposital
    with pytest.raises(AssertionError):
        verify_no_leakage(Fold("ruim", 0, train, test))


def test_folds_cover_all_observations_as_test():
    """Cada observação deve ser testada exatamente uma vez nos folds temporais."""
    df = _obs()
    folds = temporal_folds(df["t_year"])
    counts = np.zeros(len(df), dtype=int)
    for f in folds:
        counts += f.test.astype(int)
    assert np.all(counts == 1)


# ------------------------------------------------------------------ métricas
def test_fold_metrics_basic():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.5, 2000)
    v = np.full(2000, 0.25)              # var = 0.25 -> sigma = 0.5 (calibrado)
    m = fold_metrics(r, v)
    assert np.isclose(m["rmse"], 0.5, atol=0.05)
    assert abs(m["bias"]) < 0.05
    assert 0.6 < m["coverage_68"] < 0.75  # ideal 0.683
    assert 0.9 < m["z_std"] < 1.1


def test_fold_metrics_detects_miscalibration():
    """Variância subestimada -> cobertura baixa e z_std alto."""
    rng = np.random.default_rng(1)
    r = rng.normal(0, 1.0, 2000)
    v = np.full(2000, 0.01)              # sigma declarado 0.1, real 1.0
    m = fold_metrics(r, v)
    assert m["coverage_68"] < 0.2
    assert m["z_std"] > 5


def test_default_buffer_at_least_search_radius(cfg):
    b = default_buffer_m(cfg)
    assert b >= cfg.dhdt.search_radius_m
    b2 = default_buffer_m(cfg, variogram_range_m=cfg.dhdt.search_radius_m * 3)
    assert b2 > b


# --------------------------------------------------------------- integração
def test_evaluate_fold_end_to_end(cfg):
    """Fluxo completo: ajusta no treino, prevê observações retidas."""
    cfg.dhdt.min_points = 10
    cfg.dhdt.node_spacing_m = 20_000.0
    cfg.dhdt.search_radius_m = 40_000.0
    cfg.dhdt.dt_min_years = 2.0
    df = _obs(n=6000, extent=120_000.0, seed=7)
    folds = temporal_folds(df["t_year"])
    r = evaluate_fold(df, folds[2], cfg, method="idw")
    assert r["status"] == "ok"
    assert r["n"] > 0
    # o sinal é limpo (ruído 0.05 m) -> erro deve ser pequeno
    assert r["rmse"] < 2.0


def test_summarize_preserves_fold_dispersion():
    rows = [
        {"strategy": "temporal", "method": "idw", "status": "ok", "rmse": 0.4,
         "mae": 0.3, "bias": 0.01, "coverage_68": 0.7, "coverage_95": 0.95, "z_std": 1.0},
        {"strategy": "temporal", "method": "idw", "status": "ok", "rmse": 0.6,
         "mae": 0.4, "bias": -0.01, "coverage_68": 0.66, "coverage_95": 0.94, "z_std": 1.1},
    ]
    s = summarize_by_method(pd.DataFrame(rows))
    assert s.iloc[0]["n_folds"] == 2
    assert s.iloc[0]["rmse_std"] > 0        # dispersão preservada (§5.5)
    assert "cov68_dev" in s.columns
