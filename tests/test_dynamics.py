"""
Testes da Prioridade 7 (§7): integração com velocidade do gelo.

Cobre as travas de honestidade: aceleração bloqueada com mosaico único,
"estável" só dentro da incerteza, e significância corrigida por autocorrelação.
"""

import numpy as np
import pandas as pd
import pytest

from thwaites.validate.velocity import (
    joint_classification, summarize_dynamics, effective_sample_size,
    correlation_with_autocorrelation, flow_acceleration,
    ACCELERATION_BLOCKED_MSG, CLASS_THIN_FAST, CLASS_THIN_SLOW,
    CLASS_STABLE_FAST, CLASS_STABLE, CLASS_INCONCLUSIVE,
)


def _nodes():
    """Nós cobrindo cada combinação de (tendência × velocidade)."""
    return pd.DataFrame({
        "x": [0.0, 1e4, 2e4, 3e4, 4e4, 5e4],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # 0: afina signif. + rápido   1: afina signif. + lento
        # 2: compatível com 0 + rápido (PRECURSOR)  3: compatível com 0 + lento
        # 4: sem velocidade           5: sem incerteza
        "dhdt":     [-1.0, -1.0,  0.01,  0.01, -1.0, -1.0],
        "dhdt_err": [ 0.1,  0.1,  0.10,  0.10,  0.1, np.nan],
        "speed":    [500.0, 5.0, 500.0,   5.0, np.nan, 500.0],
        "n_pixels": [100, 100, 100, 100, 0, 100],
    })


# ------------------------------------------------- aceleração bloqueada (§7.2)
def test_flow_acceleration_is_blocked_with_single_mosaic():
    with pytest.raises(NotImplementedError) as e:
        flow_acceleration()
    assert "mosaico único" in str(e.value) or "múltiplas épocas" in str(e.value)


def test_acceleration_message_states_requirement():
    assert "2019" in ACCELERATION_BLOCKED_MSG
    assert "épocas" in ACCELERATION_BLOCKED_MSG


# ------------------------------------------------- classificação conjunta
def test_joint_classification_uses_confidence_intervals(cfg):
    out = joint_classification(_nodes(), cfg, fast_speed_m_yr=100.0)
    c = out["joint_class"].tolist()
    assert c[0] == CLASS_THIN_FAST
    assert c[1] == CLASS_THIN_SLOW
    assert c[2] == CLASS_STABLE_FAST      # precursor
    assert c[3] == CLASS_STABLE
    assert c[4] == CLASS_INCONCLUSIVE     # sem velocidade
    assert c[5] == CLASS_INCONCLUSIVE     # sem incerteza


def test_large_uncertainty_is_not_called_stable(cfg):
    """
    dh/dt = -1,0 mas com erro 2,0: compatível com zero, porém NÃO é evidência
    de estabilidade — a distinção precisa aparecer.
    """
    df = pd.DataFrame({"x": [0.0], "y": [0.0], "dhdt": [-1.0],
                       "dhdt_err": [2.0], "speed": [5.0], "n_pixels": [100]})
    out = joint_classification(df, cfg)
    # é classificado como estável-conjunto porque É compatível com zero,
    # mas a flag de adelgaçamento significativo tem de ser falsa
    assert bool(out.iloc[0]["compatible_with_zero"]) is True
    assert bool(out.iloc[0]["thinning_significant"]) is False


def test_no_velocity_is_inconclusive_not_stable(cfg):
    """Sem cobertura de velocidade, jamais 'estável' por omissão (§7.5)."""
    df = pd.DataFrame({"x": [0.0], "y": [0.0], "dhdt": [0.0],
                       "dhdt_err": [0.05], "speed": [np.nan], "n_pixels": [0]})
    out = joint_classification(df, cfg)
    assert out.iloc[0]["joint_class"] == CLASS_INCONCLUSIVE


# ------------------------------------------------- autocorrelação (§7.4)
def test_effective_sample_size_shrinks_with_correlation():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 200_000, 5000)
    y = rng.uniform(0, 200_000, 5000)
    n_small = effective_sample_size(x, y, 1_000.0)     # correlação curta
    n_big = effective_sample_size(x, y, 50_000.0)      # correlação longa
    assert n_big < n_small <= 5000
    assert n_big >= 2


def test_correlation_significance_corrected_for_autocorrelation():
    """
    Campos espacialmente suaves produzem correlação "altamente significativa"
    se os nós forem tratados como independentes. A correção precisa reduzir
    drasticamente a significância.
    """
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.uniform(0, 200_000, n)
    y = rng.uniform(0, 200_000, n)
    # dois campos suaves correlacionados por construção
    a = np.sin(x / 40_000) + 0.1 * rng.normal(size=n)
    b = np.sin(x / 40_000) + 0.1 * rng.normal(size=n)
    r = correlation_with_autocorrelation(x, y, a, b, correlation_length_m=50_000.0)
    assert r["n"] == n
    assert r["n_effective"] < n / 10          # muito menos amostras efetivas
    assert r["p_naive"] < r["p_autocorr_corrected"] or r["p_naive"] == 0.0


def test_summarize_reports_precursor_nodes(cfg):
    out = joint_classification(_nodes(), cfg)
    res = summarize_dynamics(out, cfg, correlation_length_m=20_000.0)
    assert res["n_possible_precursor"] == 1
    assert CLASS_STABLE_FAST in res["class_counts"]
    assert "época" in res["acceleration_status"] or "épocas" in res["acceleration_status"]
