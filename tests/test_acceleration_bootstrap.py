"""
Equivalência do bootstrap de aceleração entre a implementação por equações
normais pré-acumuladas e o ajuste completo por reamostra.

A versão rápida evita reconstruir a matriz de projeto e chamar `np.linalg.cond`
(uma SVD) a cada uma das 200 reamostras. O ganho só é legítimo se a distribuição
bootstrap resultante for a MESMA — é isso que estes testes verificam.
"""

import numpy as np
import pytest

from thwaites.timeseries.acceleration import _bootstrap_accel, ACCEL_TERMS
from thwaites.timeseries.model import build_design_matrix, fit_model


def _reference_bootstrap(h, t, dx, dy, t_ref, weights, iters, seed):
    """Implementação ANTIGA: refaz o ajuste completo a cada reamostra."""
    rng = np.random.default_rng(seed)
    yr = np.floor(t)
    years = np.unique(yr)
    if years.size < 4:
        return np.array([])
    idx_by_year = {y: np.flatnonzero(yr == y) for y in years}
    vals = []
    for _ in range(iters):
        pick = rng.choice(years, size=years.size, replace=True)
        if np.unique(pick).size < 3:
            continue
        idx = np.concatenate([idx_by_year[y] for y in pick])
        f = fit_model(h[idx], t[idx], dx[idx], dy[idx], ACCEL_TERMS, t_ref,
                      weights=None if weights is None else weights[idx])
        if f is None:
            continue
        a = f.value("accel")
        if np.isfinite(a):
            vals.append(a)
    return np.asarray(vals, dtype=float)


def _synthetic(n_years=7, n_per_year=120, accel=-0.08, dhdt=-0.4, noise=0.10,
               seed=0):
    rng = np.random.default_rng(seed)
    t, dx, dy = [], [], []
    for i in range(n_years):
        y = 2019 + i
        t.append(y + rng.uniform(0.42, 0.67, n_per_year))   # janela JJA
        dx.append(rng.uniform(-7000, 7000, n_per_year))
        dy.append(rng.uniform(-7000, 7000, n_per_year))
    t = np.concatenate(t)
    dx = np.concatenate(dx)
    dy = np.concatenate(dy)
    t_ref = 2022.5
    dt = t - t_ref
    h = (100.0 + dhdt * dt + 0.5 * accel * dt ** 2
         + 1e-4 * dx + 5e-5 * dy + rng.normal(0, noise, t.size))
    return h, t, dx, dy, t_ref


def test_bootstrap_matches_full_fit_unweighted():
    """Sem pesos: as duas implementações devem dar a MESMA distribuição."""
    h, t, dx, dy, t_ref = _synthetic()
    fast = _bootstrap_accel(h, t, dx, dy, t_ref, None, iters=120, seed=42)
    ref = _reference_bootstrap(h, t, dx, dy, t_ref, None, iters=120, seed=42)

    assert fast.size == ref.size, "nº de reamostras aceitas divergiu"
    # mesma sequência de sorteio (mesma seed) => comparação ponto a ponto
    assert np.allclose(fast, ref, rtol=1e-6, atol=1e-9), (
        f"máx |diff| = {np.max(np.abs(fast - ref)):.3e}")


def test_bootstrap_matches_full_fit_weighted():
    """Com pesos 1/σ²: a equivalência precisa valer no caso ponderado."""
    h, t, dx, dy, t_ref = _synthetic(seed=3)
    rng = np.random.default_rng(7)
    sig = rng.uniform(0.03, 0.25, t.size)
    w = 1.0 / sig ** 2

    fast = _bootstrap_accel(h, t, dx, dy, t_ref, w, iters=120, seed=11)
    ref = _reference_bootstrap(h, t, dx, dy, t_ref, w, iters=120, seed=11)

    assert fast.size == ref.size
    assert np.allclose(fast, ref, rtol=1e-6, atol=1e-9), (
        f"máx |diff| = {np.max(np.abs(fast - ref)):.3e}")


def test_bootstrap_recovers_known_acceleration():
    """
    A mediana bootstrap deve estar próxima da aceleração verdadeira.

    Testa VIÉS, não cobertura: exigir que o IC95 de UMA realização contenha o
    valor verdadeiro é estatisticamente inválido — um IC de 95% falha em 5% das
    realizações por construção, então esse teste seria intermitente por desenho.
    A cobertura, se for de interesse, tem de ser medida sobre muitas
    realizações (ver `test_bootstrap_coverage_rate`).
    """
    true_accel = -0.12
    h, t, dx, dy, t_ref = _synthetic(accel=true_accel, noise=0.05, seed=5)
    b = _bootstrap_accel(h, t, dx, dy, t_ref, None, iters=300, seed=1)
    assert b.size > 200
    med = float(np.median(b))
    assert abs(med - true_accel) < 0.05 * abs(true_accel), (
        f"viés de {100*abs(med-true_accel)/abs(true_accel):.1f}% "
        f"(mediana {med:.4f} vs verdadeiro {true_accel})")


def test_bootstrap_coverage_rate():
    """
    Taxa de cobertura do IC95 sobre MUITAS realizações de ruído.

    É assim que se testa um intervalo de confiança: a fração de realizações em
    que ele contém o valor verdadeiro deve se aproximar do nominal. Com 40
    realizações a variação amostral é grande, então a faixa aceita é larga —
    o objetivo é detectar um IC grosseiramente mal calibrado, não certificar
    95,0%.
    """
    true_accel = -0.10
    hits = 0
    n_real = 40
    for s in range(n_real):
        h, t, dx, dy, t_ref = _synthetic(accel=true_accel, noise=0.08, seed=100 + s)
        b = _bootstrap_accel(h, t, dx, dy, t_ref, None, iters=200, seed=s)
        if b.size < 50:
            continue
        lo, hi = np.percentile(b, [2.5, 97.5])
        hits += int(lo <= true_accel <= hi)
    rate = hits / n_real
    assert 0.6 <= rate <= 1.0, f"cobertura observada {rate:.2f} (nominal 0,95)"


def test_bootstrap_needs_four_years():
    """Com menos de 4 anos não há reamostragem de blocos possível."""
    h, t, dx, dy, t_ref = _synthetic(n_years=3)
    assert _bootstrap_accel(h, t, dx, dy, t_ref, None, iters=50, seed=0).size == 0


def test_column_rescaling_does_not_change_accel():
    """
    Justifica montar a matriz de projeto UMA vez: `build_design_matrix`
    normaliza as colunas espaciais pelo desvio da amostra, mas reescalar
    colunas não muda o espaço-coluna — o coeficiente de aceleração é invariante.
    """
    h, t, dx, dy, t_ref = _synthetic(seed=9)
    A, names = build_design_matrix(t, dx, dy, ACCEL_TERMS, t_ref)
    ia = names.index("accel")
    c1, *_ = np.linalg.lstsq(A, h, rcond=None)

    scale = np.ones(A.shape[1])
    for j, nm in enumerate(names):
        if nm not in ("constant", "dhdt", "accel"):
            scale[j] = 3.7                     # reescala só as espaciais
    c2, *_ = np.linalg.lstsq(A * scale, h, rcond=None)

    assert np.isclose(c1[ia], c2[ia], rtol=1e-8), (
        f"accel mudou com reescala: {c1[ia]:.6e} vs {c2[ia]:.6e}")
