"""
Testes da divergência de fluxo e do derretimento basal.

Usa casos ANALÍTICOS: campos onde ∇·(H·v) tem solução fechada, para verificar
que o cálculo numérico a recupera.
"""

import numpy as np
import pytest

from thwaites.glaciology.flux import (
    flux_divergence, basal_melt_rate, hydrostatic_thickness_rate,
    hydrostatic_amplification,
)


def _grid(res=1000.0, n=80):
    x = np.arange(0, n * res, res)
    y = np.arange(0, n * res, res)
    return x, y, *np.meshgrid(x, y)


def test_uniform_flux_has_zero_divergence(cfg):
    """H e v constantes -> ∇·(H·v) = 0."""
    cfg.flux.smooth_km = 0.0
    x, y, X, Y = _grid()
    H = np.full(X.shape, 500.0)
    vx = np.full(X.shape, 300.0)
    vy = np.zeros_like(X)
    div = flux_divergence(x, y, H, vx, vy, cfg)
    inner = div[5:-5, 5:-5]
    assert np.nanmax(np.abs(inner)) < 1e-6


def test_linear_flux_divergence_analytic(cfg):
    """
    vx = a·x (aceleração linear), H constante -> ∇·(H·v) = H·a.
    Com H=500 e a=0.001 /ano -> divergência = 0.5 m/ano.
    """
    cfg.flux.smooth_km = 0.0
    x, y, X, Y = _grid()
    H = np.full(X.shape, 500.0)
    a = 1e-3
    vx = a * X
    vy = np.zeros_like(X)
    div = flux_divergence(x, y, H, vx, vy, cfg)
    inner = div[5:-5, 5:-5]
    assert np.allclose(inner, 500.0 * a, atol=1e-6)


def test_thinning_ice_gives_divergence(cfg):
    """H decrescente em x com v constante -> divergência negativa."""
    cfg.flux.smooth_km = 0.0
    x, y, X, Y = _grid()
    H = 800.0 - 1e-3 * X            # afina 1 m por km
    vx = np.full(X.shape, 200.0)
    vy = np.zeros_like(X)
    div = flux_divergence(x, y, H, vx, vy, cfg)
    inner = div[5:-5, 5:-5]
    assert np.allclose(inner, 200.0 * (-1e-3), atol=1e-6)


def test_min_thickness_masks_out(cfg):
    cfg.flux.smooth_km = 0.0
    cfg.flux.min_thickness_m = 100.0
    x, y, X, Y = _grid()
    H = np.full(X.shape, 50.0)      # abaixo do mínimo
    v = np.full(X.shape, 100.0)
    div = flux_divergence(x, y, H, v, np.zeros_like(v), cfg)
    assert np.all(np.isnan(div))


def test_hydrostatic_amplification(cfg):
    """∂H/∂t = dh/dt · ρw/(ρw−ρi); com 1027/917 o fator é ~9.34."""
    factor = cfg.flux.water_density / (cfg.flux.water_density - cfg.mass_balance.ice_density)
    with pytest.warns(RuntimeWarning, match="sem máscara"):
        dHdt = hydrostatic_thickness_rate(np.array([-1.0]), cfg)
    assert np.isclose(dHdt[0], -factor, rtol=1e-9)
    assert 8.0 < factor < 11.0            # sanidade física


def test_hydrostatic_gating_by_floating_mask(cfg):
    """
    A amplificação hidrostática vale SÓ sobre gelo flutuante; sobre gelo
    aterrado ∂H/∂t = dh/dt (fator 1).

    A máscara deve impedir que o fator ~9,3× seja aplicado à grade
    inteira, superestimando ∂H/∂t em quase uma ordem de grandeza sobre todo o
    gelo aterrado — que domina a área de qualquer domínio continental.
    """
    amp = hydrostatic_amplification(cfg)
    dhdt = np.full((3, 3), -1.0)
    floating = np.zeros((3, 3), dtype=bool)
    floating[0, :] = True                      # só a primeira linha flutua

    out = hydrostatic_thickness_rate(dhdt, cfg, floating=floating)
    assert np.allclose(out[floating], -amp)
    assert np.allclose(out[~floating], -1.0)
    # o erro que o gating evita é de quase uma ordem de grandeza
    assert amp > 8.0


def test_basal_melt_is_nan_outside_floating(cfg):
    """
    Fora do gelo flutuante ṁ_b não é definido (o resíduo mistura deformação
    interna, derretimento subglacial e erro do dado) — deve sair NaN, para que
    nenhuma estatística agregada o inclua silenciosamente.
    """
    dHdt = np.full((2, 2), -5.0)
    div = np.zeros((2, 2))
    floating = np.array([[True, False], [False, True]])

    m = basal_melt_rate(dHdt, div, None, floating=floating)
    assert np.isfinite(m[floating]).all()
    assert np.isnan(m[~floating]).all()
    # a mediana passa a refletir só a plataforma
    assert np.isclose(np.nanmedian(m), 5.0)


def test_basal_melt_sign_convention(cfg):
    """
    ṁ_b = SMB − ∂H/∂t − ∇·(H·v). Gelo afinando (∂H/∂t<0) sem divergência
    => derretimento positivo.
    """
    m = basal_melt_rate(np.array([-5.0]), np.array([0.0]))
    assert m[0] > 0 and np.isclose(m[0], 5.0)
    # com SMB de acumulação, o derretimento inferido aumenta
    m2 = basal_melt_rate(np.array([-5.0]), np.array([0.0]), smb=np.array([0.3]))
    assert np.isclose(m2[0], 5.3)
