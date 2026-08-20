"""
Testes da Prioridade 3 (recorte de aceleração) — §4.6 e §9.

Cobre os dois lados que importam:
  - recuperar uma aceleração CONHECIDA quando ela existe;
  - NÃO reportar aceleração quando ela não existe (evitar falso positivo),
    que é o risco real com ~7 invernos.
"""

import numpy as np
import pandas as pd
import pytest

from thwaites.timeseries.model import (
    build_design_matrix, fit_model, check_identifiability,
    seasonal_phase_coverage, leave_one_year_out, compare_jja_annual,
)
from thwaites.timeseries.acceleration import (
    assess_acceleration, AccelCriteria, LINEAR_TERMS, ACCEL_TERMS,
)

T_REF = 2022.0


def _series(dhdt=-0.5, accel=0.0, n_per_year=180, years=range(2019, 2026),
            noise=0.05, seed=0, month=0.6):
    """Série sintética JJA-like: h = dhdt·dt + ½·accel·dt² + ruído."""
    rng = np.random.default_rng(seed)
    t, h, dx, dy = [], [], [], []
    for y in years:
        tt = y + month + rng.normal(0, 0.02, n_per_year)
        dt = tt - T_REF
        xx = rng.uniform(-10000, 10000, n_per_year)
        yy = rng.uniform(-10000, 10000, n_per_year)
        hh = 500 + dhdt * dt + 0.5 * accel * dt**2 + rng.normal(0, noise, n_per_year)
        t.append(tt); h.append(hh); dx.append(xx); dy.append(yy)
    return (np.concatenate(h), np.concatenate(t),
            np.concatenate(dx), np.concatenate(dy))


# ------------------------------------------------- convenção e matriz
def test_acceleration_convention_is_second_derivative():
    """½·β2·dt² => β2 = d²h/dt² diretamente (§4.3)."""
    h, t, dx, dy = _series(dhdt=-0.4, accel=-0.12, noise=0.001, seed=1)
    fit = fit_model(h, t, dx, dy, ACCEL_TERMS, T_REF)
    assert fit is not None
    assert np.isclose(fit.value("accel"), -0.12, atol=0.02)
    assert np.isclose(fit.value("dhdt"), -0.4, atol=0.02)


def test_design_matrix_has_expected_columns():
    h, t, dx, dy = _series()
    A, names = build_design_matrix(t, dx, dy, ACCEL_TERMS, T_REF)
    assert "dhdt" in names and "accel" in names and "constant" in names
    assert A.shape[0] == t.size and A.shape[1] == len(names)


# ------------------------------------------------- identificabilidade
def test_jja_only_blocks_seasonal():
    """Dados só de inverno: sazonalidade NÃO identificável (§4.2)."""
    _, t, _, _ = _series(month=0.6)                 # sempre ~ mesmo mês
    ident = check_identifiability(t, ("constant", "linear", "seasonal"))
    assert "seasonal" not in ident["allowed_terms"]
    assert "seasonal" in ident["reasons"]
    assert seasonal_phase_coverage(t) < 0.35


def test_full_year_sampling_allows_seasonal():
    rng = np.random.default_rng(0)
    t = 2019 + rng.uniform(0, 6, 2000)              # espalhado no ano todo
    ident = check_identifiability(t, ("constant", "linear", "seasonal"))
    assert "seasonal" in ident["allowed_terms"]
    assert seasonal_phase_coverage(t) > 0.5


def test_short_period_blocks_acceleration():
    _, t, _, _ = _series(years=range(2019, 2022))   # só 3 anos
    ident = check_identifiability(t, ACCEL_TERMS, min_years_for_accel=5)
    assert "acceleration" not in ident["allowed_terms"]


# ------------------------------------------------- recuperação do sinal
def test_recovers_strong_acceleration():
    """Aceleração forte e limpa deve ser detectada e reportada."""
    h, t, dx, dy = _series(dhdt=-0.3, accel=-0.25, noise=0.03, seed=2)
    r = assess_acceleration(h, t, dx, dy, T_REF,
                            criteria=AccelCriteria(boot_iters=60))
    assert r["accel_supported"] is True, r["reason"]
    assert np.isclose(r["accel"], -0.25, atol=0.05)
    assert r["delta_aicc"] > 2.0


# ------------------------------------------------- FALSO POSITIVO (§9)
def test_no_acceleration_is_not_reported():
    """Sinal puramente linear: NÃO deve reportar aceleração."""
    h, t, dx, dy = _series(dhdt=-0.5, accel=0.0, noise=0.05, seed=3)
    r = assess_acceleration(h, t, dx, dy, T_REF,
                            criteria=AccelCriteria(boot_iters=60))
    assert r["accel_supported"] is False
    assert np.isnan(r["accel"])
    # a taxa linear continua sendo estimada e correta
    assert np.isclose(r["dhdt"], -0.5, atol=0.05)


def _noise_case(seed, n_per_year=150, sigma=0.5):
    rng = np.random.default_rng(seed)
    t = np.repeat(np.arange(2019, 2026) + 0.6, n_per_year) + \
        rng.normal(0, 0.02, 7 * n_per_year)
    h = rng.normal(500, sigma, t.size)
    return h, t, rng.uniform(-1e4, 1e4, t.size), rng.uniform(-1e4, 1e4, t.size)


def test_false_positive_rate_on_pure_noise():
    """
    TAXA de falso positivo em ruído puro — não um sorteio isolado.

    Com α=0,05, esperar que UMA realização específica de ruído nunca passe é
    estatisticamente incorreto: ~5% delas passam por definição. O que precisa
    ser controlado é a TAXA ao longo de muitas realizações.
    (Medido em 40 realizações: 2,5%.)
    """
    crit = AccelCriteria(boot_iters=60)
    n = 20
    fp = sum(bool(assess_acceleration(*_noise_case(s), T_REF, criteria=crit,
                                      seed=s)["accel_supported"]) for s in range(n))
    assert fp <= 3, f"taxa de falso positivo alta demais: {fp}/{n}"


def test_detects_real_acceleration_reliably():
    """Poder: aceleração real deve ser detectada de forma consistente."""
    crit = AccelCriteria(boot_iters=60)
    hits = 0
    n = 10
    for s in range(n):
        rng = np.random.default_rng(1000 + s)
        t = np.repeat(np.arange(2019, 2026) + 0.6, 150) + rng.normal(0, 0.02, 7 * 150)
        dt = t - T_REF
        h = 500 - 0.3 * dt + 0.5 * (-0.25) * dt ** 2 + rng.normal(0, 0.5, t.size)
        dx = rng.uniform(-1e4, 1e4, t.size); dy = rng.uniform(-1e4, 1e4, t.size)
        hits += bool(assess_acceleration(h, t, dx, dy, T_REF, criteria=crit,
                                         seed=s)["accel_supported"])
    assert hits >= 8, f"poder baixo demais: {hits}/{n}"


def test_short_series_refuses_and_explains():
    h, t, dx, dy = _series(years=range(2019, 2022), seed=5)
    r = assess_acceleration(h, t, dx, dy, T_REF)
    assert r["accel_supported"] is False
    assert "curto" in r["reason"] or "anos" in r["reason"]
    assert np.isfinite(r["dhdt"])          # ainda entrega a tendência linear


# ------------------------------------------------- validação fora da amostra
def test_leave_one_year_out_prefers_correct_model():
    """Com aceleração real, o modelo com aceleração prevê melhor o ano retido."""
    h, t, dx, dy = _series(dhdt=-0.3, accel=-0.3, noise=0.03, seed=6)
    oos_lin = leave_one_year_out(h, t, dx, dy, LINEAR_TERMS, T_REF)
    oos_acc = leave_one_year_out(h, t, dx, dy, ACCEL_TERMS, T_REF)
    assert np.isfinite(oos_lin) and np.isfinite(oos_acc)
    assert oos_acc < oos_lin


# ------------------------------------------------- JJA vs anual (§4.5)
def test_jja_annual_comparison_tests_not_assumes():
    """A conclusão 'JJA é conservador' precisa sair dos dados, não de premissa."""
    n = 300
    x = np.arange(n) * 1000.0
    jja = pd.DataFrame({"x": x, "y": np.zeros(n), "dhdt": np.full(n, -0.40)})
    ann = pd.DataFrame({"x": x, "y": np.zeros(n), "dhdt": np.full(n, -0.60)})
    r = compare_jja_annual(jja, ann, bootstrap=100)
    # JJA menos negativo que o anual => subestima a perda => conservador
    assert r["jja_is_conservative"] is True
    assert np.isclose(r["median_diff_jja_minus_annual"], 0.20, atol=1e-6)

    # caso oposto: JJA MAIS negativo => NÃO é conservador
    ann2 = pd.DataFrame({"x": x, "y": np.zeros(n), "dhdt": np.full(n, -0.20)})
    r2 = compare_jja_annual(jja, ann2, bootstrap=100)
    assert r2["jja_is_conservative"] is False
