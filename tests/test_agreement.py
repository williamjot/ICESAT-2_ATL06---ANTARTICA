"""
Testes da Prioridade 5 (§6): concordância espacial ajuste local × crossovers.

O teste central: a análise precisa DETECTAR discordância regional que a
mediana global esconde — é a pergunta do §6.1.
"""

import numpy as np
import pandas as pd
import pytest

from thwaites.validation.agreement import (
    match_xovers_to_nodes, paired_differences, robust_regression,
    spatial_structure_of_differences, find_hotspots, assess_agreement,
    INDEPENDENCE_CAVEAT,
)
from thwaites.qc.xover import interbeam_bias, interbeam_bias_sensitivity


def _nodes(n=40, extent=200_000.0, dhdt=-0.5, err=0.1, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, extent, n)
    X, Y = np.meshgrid(x, x)
    return pd.DataFrame({
        "x": X.ravel(), "y": Y.ravel(),
        "dhdt": np.full(X.size, dhdt) + rng.normal(0, 0.01, X.size),
        "dhdt_err": np.full(X.size, err),
    })


def _xovers_from(nodes, offset=0.0, err=0.1, seed=1, regional=False):
    """Crossovers colocados sobre os nós, com deslocamento controlado."""
    rng = np.random.default_rng(seed)
    x = nodes["x"].to_numpy(); y = nodes["y"].to_numpy()
    off = np.full(x.size, offset)
    if regional:
        # discordância que troca de sinal entre metades -> mediana global ~0
        off = np.where(x < np.median(x), +0.30, -0.30)
    return pd.DataFrame({
        "x": x, "y": y,
        "dhdt": nodes["dhdt"].to_numpy() + off + rng.normal(0, 0.01, x.size),
        "dhdt_err": np.full(x.size, err),
        "dt": rng.uniform(1.0, 5.0, x.size),
    })


# ------------------------------------------------------------- pareamento
def test_match_discards_distant_xovers():
    nodes = _nodes(n=10, extent=100_000.0)
    xo = _xovers_from(nodes)
    xo.loc[0, "x"] += 50_000.0                 # afasta um crossover
    m = match_xovers_to_nodes(xo, nodes, max_dist_m=3000.0)
    assert len(m) == len(xo) - 1
    assert (m["dist_to_node_m"] <= 3000.0).all()


def test_paired_differences_normalizes_by_combined_uncertainty():
    nodes = _nodes(n=8)
    xo = _xovers_from(nodes, offset=0.2, err=0.1)
    m = paired_differences(match_xovers_to_nodes(xo, nodes))
    assert np.isclose(np.median(m["diff"]), 0.2, atol=0.02)
    # σ combinada = sqrt(0.1² + 0.1²) ≈ 0.141 -> z ≈ 1.4
    assert np.isclose(np.median(m["sigma_combined"]), np.sqrt(0.02), atol=1e-3)
    assert np.isclose(np.median(np.abs(m["z"])), 0.2 / np.sqrt(0.02), atol=0.2)


# ------------------------------------------------------------- regressão
def test_robust_regression_detects_consistency():
    nodes = _nodes(n=15)
    xo = _xovers_from(nodes, offset=0.0)
    m = paired_differences(match_xovers_to_nodes(xo, nodes))
    r = robust_regression(m["local_dhdt"], m["dhdt"])
    assert r["n"] > 100
    assert abs(r["intercept"]) < 0.1


# --------------------------------- O TESTE CENTRAL DO §6.1 ------------------
def test_detects_regional_disagreement_hidden_by_global_median():
    """
    Discordância que troca de sinal entre setores: a MEDIANA GLOBAL fica ~0,
    mas a análise espacial precisa flagrar a discordância regional.
    """
    nodes = _nodes(n=30, extent=200_000.0)
    xo = _xovers_from(nodes, regional=True)
    m = paired_differences(match_xovers_to_nodes(xo, nodes))
    # 1) a mediana global esconde o problema
    assert abs(np.median(m["diff"])) < 0.05
    # 2) mas a estrutura espacial o revela
    sp = spatial_structure_of_differences(m, cfg=None)
    assert sp["status"] == "ok"
    assert sp["spatially_structured"] is True
    # 3) e os hotspots o localizam
    hs = find_hotspots(m, cell_km=25.0, min_count=5)
    assert hs["hotspot"].sum() > 0


def test_no_disagreement_gives_no_spatial_structure():
    nodes = _nodes(n=30, extent=200_000.0, seed=3)
    xo = _xovers_from(nodes, offset=0.0, seed=4)
    m = paired_differences(match_xovers_to_nodes(xo, nodes))
    hs = find_hotspots(m, cell_km=25.0, min_count=5)
    # sem discordância imposta, praticamente nenhum hotspot
    assert hs["hotspot"].sum() <= max(1, int(0.05 * len(hs)))


# ------------------------------------------------- viés inter-feixe (§6.3)
def _near_simultaneous(n=400, dhdt_real=-0.5, bias=0.03, seed=0):
    """Crossovers quase simultâneos com mudança REAL + viés instrumental."""
    rng = np.random.default_rng(seed)
    dt = rng.uniform(-0.25, 0.25, n)
    dh = dhdt_real * dt + bias + rng.normal(0, 0.01, n)
    return pd.DataFrame({"dt": dt, "dh": dh,
                         "beam1": np.ones(n, int), "beam2": np.full(n, 4)})


def test_interbeam_bias_is_contaminated_without_correction():
    """Sem remover a mudança real, o 'viés' sai enviesado (§6.3)."""
    xo = _near_simultaneous(bias=0.03, dhdt_real=-2.0)
    raw = interbeam_bias(xo, max_dt_years=0.25)
    corr = interbeam_bias(xo, max_dt_years=0.25, expected_dhdt=-2.0)
    assert np.isclose(corr.iloc[0]["bias_m"], 0.03, atol=0.01)
    assert bool(corr.iloc[0]["expected_change_removed"]) is True
    # O pareamento espacial recupera o viés real com menor dispersão.
    assert corr.iloc[0]["mad_m"] < raw.iloc[0]["mad_m"]


def test_interbeam_sensitivity_to_window():
    xo = _near_simultaneous(bias=0.03, dhdt_real=-2.0)
    s = interbeam_bias_sensitivity(xo, windows=(0.05, 0.25))
    assert len(s) == 2
    assert (s["n_pairs"] > 0).all()


def test_independence_caveat_is_explicit():
    """§6.5: crossovers não podem ser vendidos como validação independente."""
    assert "NÃO" in INDEPENDENCE_CAVEAT
    assert "ATL06" in INDEPENDENCE_CAVEAT
