"""
Testes da filtragem along-track: reconstrução de trilhas, rejeição de blunder
(spike) e preservação de dados limpos (sem super-rejeição).
"""

import numpy as np
import pandas as pd

from thwaites.qc.filttrack import filter_along_track, track_ids


def _track(n=400, beam=1, t0=2021.5, h0=500.0, slope=1e-4, noise=0.05, seed=0):
    """Uma trilha: tempo monotônico e elevação suave + ruído pequeno."""
    rng = np.random.default_rng(seed)
    dt = 1e-8                                  # passo de tempo entre segmentos
    t = t0 + np.arange(n) * dt
    h = h0 + slope * np.arange(n) + rng.normal(0, noise, n)
    return pd.DataFrame({
        "beam": np.full(n, beam, dtype="int8"),
        "t_year": t,
        "h_res": h.astype("float32"),
        "lon": np.full(n, -107.0), "lat": np.full(n, -75.0),
    })


# ------------------------------------------------------- reconstrução de trilha
def test_track_ids_splits_on_beam_and_gap():
    beam = np.array([1, 1, 1, 2, 2], dtype="int8")
    t = np.array([0.0, 1e-9, 2e-9, 3e-9, 4e-9])
    # muda de feixe no índice 3 -> duas trilhas
    assert track_ids(beam, t, gap_year=1e-4).tolist() == [0, 0, 0, 1, 1]

    beam2 = np.ones(4, dtype="int8")
    t2 = np.array([0.0, 1e-9, 1.0, 1.0 + 1e-9])   # gap grande no meio
    assert track_ids(beam2, t2, gap_year=1e-4).tolist() == [0, 0, 1, 1]


def test_filttrack_assigns_track_id_per_beam(cfg):
    df = pd.concat([_track(beam=1, seed=1), _track(beam=3, seed=2)], ignore_index=True)
    out = filter_along_track(df, cfg)
    assert "track_id" in out.columns
    assert out["track_id"].nunique() == 2          # um por feixe


# ------------------------------------------------------------ rejeição de spike
def test_filttrack_removes_spike(cfg):
    df = _track(n=400, seed=3)
    spike_at = 200
    df.loc[spike_at, "h_res"] = np.float32(df.loc[spike_at, "h_res"] + 25.0)
    out = filter_along_track(df, cfg)
    # o spike foi removido (nenhuma elevação absurda sobrou)
    assert len(out) < len(df)
    assert out["h_res"].max() < df["h_res"].max()
    # e a remoção foi cirúrgica (não varreu a trilha)
    assert len(out) >= len(df) - 5


def test_filttrack_preserves_clean_data(cfg):
    """Dados limpos: rejeição deve ser mínima (evita super-rejeição)."""
    df = _track(n=1000, noise=0.05, seed=4)
    out = filter_along_track(df, cfg)
    assert len(out) >= 0.99 * len(df)


def test_filttrack_edges_not_filtered(cfg):
    """Pontos de borda (janela incompleta) são mantidos, não descartados."""
    df = _track(n=100, seed=5)
    # spike na 2ª amostra (dentro de half-window da borda) -> não filtrado
    df.loc[1, "h_res"] = np.float32(df.loc[1, "h_res"] + 50.0)
    out = filter_along_track(df, cfg)
    assert len(out) == len(df)


def test_filttrack_disabled_is_noop(cfg):
    cfg.filttrack.enabled = False
    df = _track(n=50, seed=6)
    out = filter_along_track(df, cfg)
    assert len(out) == len(df)
