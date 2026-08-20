"""Testes da aplicação de correções geofísicas (gating, fill, sinal)."""

import numpy as np
import pandas as pd
import pytest

from thwaites.corrections import apply_corrections


def _df():
    # r0 flutuante (3), r1 aterrado (2), r2 flutuante com tide fill(NaN)
    return pd.DataFrame({
        "h_elv":      np.array([100.0, 100.0, 100.0], dtype="float32"),
        "tide_ocean": np.array([0.5, 0.5, np.nan],   dtype="float32"),
        "dac":        np.array([0.1, 0.1, 0.1],       dtype="float32"),
        "geoid":      np.array([-30.0, -30.0, -30.0], dtype="float32"),
        "mask_class": np.array([3, 2, 3],             dtype="int32"),
    })


def test_gating_and_sign(cfg):
    # cfg default: apply=[tide_ocean,dac], gate_to_floating=True, floating_class=3
    out = apply_corrections(_df(), cfg)
    h = out["h_corr"].to_numpy()
    # r0 flutuante: 100 - 0.5 - 0.1 = 99.4
    assert np.isclose(h[0], 99.4, atol=1e-4)
    # r1 aterrado: sem correção (gating) = 100.0
    assert np.isclose(h[1], 100.0, atol=1e-4)
    # r2 flutuante, tide NaN->0, só dac: 100 - 0.1 = 99.9
    assert np.isclose(h[2], 99.9, atol=1e-4)


def test_gate_requires_mask_class(cfg):
    df = _df().drop(columns=["mask_class"])
    with pytest.raises(ValueError):
        apply_corrections(df, cfg)


def test_no_gating_applies_everywhere(cfg):
    cfg.corrections.gate_to_floating = False
    out = apply_corrections(_df(), cfg)
    h = out["h_corr"].to_numpy()
    # sem gating, o ponto aterrado também é corrigido: 100 - 0.6 = 99.4
    assert np.isclose(h[1], 99.4, atol=1e-4)


def test_missing_correction_column_errors(cfg):
    df = _df().drop(columns=["dac"])
    with pytest.raises(ValueError):
        apply_corrections(df, cfg)


def test_equilibrium_tide_requires_explicit_nonduplication(cfg):
    cfg.corrections.equilibrium_tide_mode = "apply_atl06"
    # CATS ativo, mas sem declaração de constituintes: falha antes de calcular.
    cfg.cats.enabled = True
    cfg.cats.equilibrium_tide_included = None
    with pytest.raises(ValueError, match="equilibrium_tide_included"):
        apply_corrections(_df(), cfg)


def test_equilibrium_tide_is_applied_only_when_declared(cfg):
    cfg.corrections.equilibrium_tide_mode = "apply_atl06"
    cfg.cats.enabled = True
    cfg.cats.equilibrium_tide_included = False
    df = _df().assign(tide_equilibrium=np.array([0.02, 0.02, 0.02], dtype="float32"))
    out = apply_corrections(df, cfg)
    # gelo flutuante: 100 - 0.5 - 0.02 - 0.1
    assert np.isclose(out.loc[0, "h_corr"], 99.38, atol=1e-4)
    # gelo aterrado continua protegido pelo gating.
    assert np.isclose(out.loc[1, "h_corr"], 100.0, atol=1e-4)
