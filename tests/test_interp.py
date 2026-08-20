"""
Testes da Fase 5: variograma data-driven, os três interpoladores, blocos
espaciais de CV e a seleção do vencedor.
"""

import numpy as np
import pandas as pd
import pytest

from thwaites.interp.variogram import empirical_variogram, fit_variogram, make_gamma
from thwaites.interp.methods import (
    idw_predict, oi_markov_predict, ordinary_kriging_predict,
)
from thwaites.interp.select import (
    spatial_block_folds, cross_validate, select_interpolator,
)


def _smooth_field(n=900, seed=0):
    """Campo suave (tendência plana + ondulação leve) sobre 50x50 km."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 50_000, n)
    y = rng.uniform(0, 50_000, n)
    v = (1e-4 * x - 5e-5 * y
         + 0.3 * np.sin(x / 8000.0) + 0.2 * np.cos(y / 9000.0)
         + rng.normal(0, 0.02, n))
    sig = np.full(n, 0.1)
    return x, y, v, sig


# ------------------------------------------------------------------ variograma
def test_variogram_fit_reasonable():
    x, y, v, _ = _smooth_field()
    p = fit_variogram(x, y, v)
    assert p["model"] in ("spherical", "exponential")
    assert p["range_m"] > 0
    assert p["sill"] > 0
    # gamma cresce de ~0 em direção ao patamar
    g = make_gamma(p)
    assert g(1.0) < g(p["range_m"] * 2)


# ------------------------------------------------------------------ métodos
@pytest.mark.parametrize("predictor", ["idw", "oi", "ok", "gauss", "median"])
def test_predictor_recovers_smooth_field(predictor):
    from thwaites.interp.methods import gaussian_kernel_predict, median_kernel_predict
    x, y, v, sig = _smooth_field(seed=1)
    # treina em 800, prediz 100 retidos
    tr = slice(0, 800); te = slice(800, 900)
    if predictor == "gauss":
        pred, var = gaussian_kernel_predict(x[tr], y[tr], v[tr], x[te], y[te],
                                            sigma=4000.0, k=32)
    elif predictor == "median":
        pred, var = median_kernel_predict(x[tr], y[tr], v[tr], x[te], y[te],
                                          radius=8000.0, k=32)
    elif predictor == "idw":
        pred, var = idw_predict(x[tr], y[tr], v[tr], x[te], y[te], power=2.0, k=32)
    elif predictor == "oi":
        p = fit_variogram(x[tr], y[tr], v[tr])
        pred, var = oi_markov_predict(x[tr], y[tr], v[tr], sig[tr], x[te], y[te],
                                      corr_len=p["range_m"], sill=p["sill"], k=32)
    else:
        p = fit_variogram(x[tr], y[tr], v[tr])
        pred, var = ordinary_kriging_predict(x[tr], y[tr], v[tr], x[te], y[te],
                                             gamma=make_gamma(p), k=32)
    rmse = np.sqrt(np.mean((pred - v[te]) ** 2))
    # muito melhor que o preditor trivial (média)
    assert rmse < 0.5 * np.std(v)
    assert np.all(var >= 0)


# ------------------------------------------------------------------ blocos CV
def test_spatial_block_folds_partition():
    x, y, _, _ = _smooth_field()
    fold = spatial_block_folds(x, y, block_m=10_000, n_folds=5, seed=0)
    assert fold.min() >= 0 and fold.max() <= 4
    assert len(np.unique(fold)) == 5           # todos os folds usados
    # pontos do mesmo bloco caem no mesmo fold
    bi = (np.floor(x / 10_000).astype(int) * 100003 + np.floor(y / 10_000).astype(int))
    for b in np.unique(bi)[:20]:
        assert len(np.unique(fold[bi == b])) == 1


# ------------------------------------------------------------------ seleção
def test_cross_validate_and_select(cfg):
    # campo sintético tem 50 km; usa blocos de 10 km p/ ter vários blocos
    cfg.interpolation.cv.block_km = 10.0
    x, y, v, sig = _smooth_field(n=1200, seed=2)
    table, winner = cross_validate(x, y, v, sig, cfg)
    assert winner in cfg.interpolation.candidates
    assert set(["method", "rmse", "cal_frac_1sigma"]).issubset(table.columns)
    # todos os candidatos batem o preditor trivial (RMSE < std)
    assert (table["rmse"] < np.std(v)).all()


def test_select_interpolator_endtoend(cfg):
    cfg.interpolation.cv.block_km = 10.0
    x, y, v, sig = _smooth_field(n=1000, seed=3)
    nodes = pd.DataFrame({"x": x, "y": y, "dhdt": v, "dhdt_err": sig})
    res = select_interpolator(nodes, cfg)
    assert res["winner"] in cfg.interpolation.candidates
    assert res["variogram"]["range_m"] > 0
    assert len(res["metrics"]) == len(cfg.interpolation.candidates)
