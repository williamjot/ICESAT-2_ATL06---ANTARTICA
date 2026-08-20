"""
Testes da Fase 6: valores de balanço de massa (unidades/sinal) e a
propagação de incerteza correlacionada vs. independente.
"""

import math

import numpy as np
import pandas as pd

from thwaites.uncertainty.mass_balance import (
    compute_mass_balance, apply_coverage_mask, gt_per_mm_sle,
)


def _uniform_grid(cfg, n=100, dhdt=-1.0, var=0.04):
    """Grade de n células (res do config) com dh/dt e variância uniformes."""
    res = cfg.interpolation.grid_res_m
    xs = (np.arange(n) % 10) * res
    ys = (np.arange(n) // 10) * res
    return pd.DataFrame({"x": xs.astype(float), "y": ys.astype(float),
                         "pred": np.full(n, dhdt), "var": np.full(n, var)})


def test_gt_per_mm_constant(cfg):
    # área oceânica 3.618e14 m² -> 361.8 Gt/mm
    assert math.isclose(gt_per_mm_sle(cfg.mass_balance.ocean_area_m2), 361.8, rel_tol=1e-6)


def test_mass_balance_values_and_sign(cfg):
    grid = _uniform_grid(cfg, n=100, dhdt=-1.0, var=0.04)
    r = compute_mass_balance(grid, cfg, correlation_length_m=20_000.0)
    a = cfg.interpolation.grid_res_m ** 2
    expected_dVdt = -1.0 * 100 * a
    assert math.isclose(r["dVdt_m3_yr"], expected_dVdt, rel_tol=1e-9)
    expected_dMdt = 917.0 * expected_dVdt / 1e12
    assert math.isclose(r["dMdt_Gt_yr"], expected_dMdt, rel_tol=1e-9)
    # perda de massa (dM negativo) -> subida do nível do mar (SLE positivo)
    assert r["dMdt_Gt_yr"] < 0
    assert r["sle_mm_yr"] > 0


def test_correlated_uncertainty_larger_than_independent(cfg):
    grid = _uniform_grid(cfg, n=200, dhdt=-0.5, var=0.04)  # sigma=0.2
    L = 20_000.0
    r = compute_mass_balance(grid, cfg, correlation_length_m=L)
    # correlacionado > independente
    assert r["sigma_dMdt_Gt_yr_correlated"] > r["sigma_dMdt_Gt_yr_independent"]
    # fator de inflação = sqrt(A_corr / A_celula)
    a = cfg.interpolation.grid_res_m ** 2
    expected = math.sqrt(math.pi * L**2 / a)
    assert math.isclose(r["inflation_factor"], expected, rel_tol=1e-6)


def test_coverage_mask(cfg):
    res = cfg.interpolation.grid_res_m
    grid = pd.DataFrame({"x": [0.0, 10 * res], "y": [0.0, 0.0],
                         "pred": [-1.0, -1.0], "var": [0.04, 0.04]})
    nodes = pd.DataFrame({"x": [0.0], "y": [0.0]})   # nó só perto da 1ª célula
    covered = apply_coverage_mask(grid, nodes, max_dist_m=cfg.mass_balance.coverage_dist_m)
    assert len(covered) == 1
    assert covered.iloc[0]["x"] == 0.0
