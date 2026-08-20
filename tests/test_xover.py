"""
Testes de crossovers: classificação asc/desc, detecção do cruzamento e
recuperação de um dh/dt conhecido imposto entre as duas passagens.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shapely")

from thwaites.qc.xover import find_crossovers, classify_tracks, interbeam_bias


def _cross_pair(dhdt=-0.6, t1=2020.5, t2=2024.5, n=200, h0=500.0, seed=0):
    """
    Duas trilhas que se cruzam em (0,0): uma ascendente (sobe em y), outra
    descendente (desce em y), com uma diferença de elevação = dhdt*(t2-t1).
    """
    rng = np.random.default_rng(seed)
    s = np.linspace(-5000, 5000, n)          # 10 km de trilha
    noise = 0.01

    # ascendente: y cresce; passa por (0,0)
    a = pd.DataFrame({
        "track_id": 1, "x": s * 0.2, "y": s,
        "t_year": t1 + np.arange(n) * 1e-9,
        "beam": np.int8(1),
        "h_res": (h0 + rng.normal(0, noise, n)).astype("float32"),
    })
    # descendente: y decresce; cruza a primeira em ~(0,0) num ÂNGULO real
    # (x com o mesmo sinal de s e y invertido -> retas de inclinação +5 e -5)
    b = pd.DataFrame({
        "track_id": 2, "x": s * 0.2, "y": -s,
        "t_year": t2 + np.arange(n) * 1e-9,
        "beam": np.int8(4),
        "h_res": (h0 + dhdt * (t2 - t1) + rng.normal(0, noise, n)).astype("float32"),
    })
    return pd.concat([a, b], ignore_index=True)


def test_classify_asc_desc():
    df = _cross_pair()
    info = classify_tracks(df).set_index("track_id")
    assert bool(info.loc[1, "ascending"]) is True
    assert bool(info.loc[2, "ascending"]) is False


def test_crossover_recovers_known_dhdt(cfg):
    df = _cross_pair(dhdt=-0.6, t1=2020.5, t2=2024.5)
    xo = find_crossovers(df, cfg)
    assert len(xo) >= 1
    r = xo.iloc[0]
    # cruzamento perto da origem
    assert abs(r["x"]) < 200 and abs(r["y"]) < 200
    # Δt e dh/dt corretos
    assert np.isclose(r["dt"], 4.0, atol=0.01)
    assert np.isclose(r["dhdt"], -0.6, atol=0.05)
    # feixes registrados
    assert {int(r["beam1"]), int(r["beam2"])} == {1, 4}


def test_crossover_requires_track_id(cfg):
    df = _cross_pair().drop(columns=["track_id"])
    with pytest.raises(ValueError):
        find_crossovers(df, cfg)


def test_interbeam_bias_detects_offset(cfg):
    """Cruzamento quase-simultâneo com offset conhecido -> viés detectado."""
    df = _cross_pair(dhdt=0.0, t1=2021.0, t2=2021.05)   # ~18 dias
    df.loc[df["track_id"] == 2, "h_res"] += np.float32(0.25)   # viés de 25 cm
    xo = find_crossovers(df, cfg)
    bias = interbeam_bias(xo, max_dt_years=0.25)
    assert not bias.empty
    assert np.isclose(bias.iloc[0]["bias_m"], 0.25, atol=0.05)
