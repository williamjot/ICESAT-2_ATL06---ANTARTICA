"""
thwaites.qc.filttrack
=====================
Filtragem ALONG-TRACK (inspirada no `filttrack.py` do captoolkit, NASA JPL).

Remove blunders coerentes ao longo da trilha orbital — nuvem/neblina, neve
soprada, erro do DEM de referência em margem de cisalhamento — que NÃO são
pegos por:
  - `atl06_quality_summary` (flag do produto, conservador demais);
  - a rejeição robusta dentro do ajuste por nó (que vê uma vizinhança 2D
    misturando várias trilhas, diluindo o blunder).

Método: dentro de cada trilha, ajusta uma mediana móvel à elevação e rejeita
pontos cujo resíduo excede `n_sigma` × MAD da própria trilha (MAD local, com
piso `mad_floor_m` para não super-rejeitar trilhas muito lisas).

RECONSTRUÇÃO DA ORDEM ALONG-TRACK: a ordem original é perdida na consolidação,
mas é recuperável — dentro de um feixe, ordenar por `t_year` reproduz a
sequência along-track (o satélite avança monotonicamente no tempo). Trilhas
distintas são separadas por um gap de tempo > `gap_year` (granulos diferentes).
Isso permite filtrar os dados JÁ baixados, sem re-download.

Nos pontos a menos de metade da janela da borda de uma trilha o filtro não é
aplicado (janela incompleta contaminaria com a trilha vizinha) — são mantidos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger


def _height_column(df: pd.DataFrame) -> str:
    """Prioridade: h_res (slope-referenciado) > h_corr > h_elv."""
    for c in ("h_res", "h_corr", "h_elv"):
        if c in df.columns:
            return c
    raise ValueError("nenhuma coluna de elevação encontrada (h_res/h_corr/h_elv)")


def track_ids(beam: np.ndarray, t_year: np.ndarray, gap_year: float) -> np.ndarray:
    """
    Rótulo de trilha para dados JÁ ORDENADOS por (beam, t_year).

    Nova trilha quando o feixe muda ou o gap de tempo excede `gap_year`.
    """
    if len(beam) == 0:
        return np.array([], dtype=np.int64)
    new = np.empty(len(beam), dtype=bool)
    new[0] = True
    new[1:] = (beam[1:] != beam[:-1]) | (np.diff(t_year) > gap_year)
    return np.cumsum(new) - 1


def compute_along_track_mask(beam, t_year, h, cfg: Config):
    """
    Calcula, sobre ARRAYS, a máscara de rejeição along-track e o `track_id`.

    POR QUE ARRAYS E NÃO DataFrame: `df.iloc[order]` criaria uma cópia ordenada
    da tabela inteira (~1 GB em 20 M linhas × 17 colunas). Numa máquina com
    ~3 GB livres isso pode levar a swap e travamento. Somente três
    vetores são ordenados; a saída é uma máscara booleana (20 MB) na ordem
    ORIGINAL, que o chamador aplica como quiser — inclusive em streaming.

    Retorna (bad, track_id, stats) todos na ordem original de entrada.
    """
    from scipy.ndimage import median_filter

    logger = get_logger()
    ft = cfg.filttrack
    beam = np.asarray(beam)
    t = np.asarray(t_year, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    n = t.size

    win = int(ft.window) + (1 - int(ft.window) % 2)   # garante ímpar
    half = win // 2

    order = np.lexsort((t, beam))                     # ordem along-track
    b_s = beam[order]
    t_s = t[order]
    h_s = h[order]

    tid_s = track_ids(b_s, t_s, ft.gap_year)
    n_tracks = int(tid_s[-1] + 1) if n else 0

    hw = np.where(np.isfinite(h_s), h_s,
                  np.nanmedian(h_s[np.isfinite(h_s)]) if np.isfinite(h_s).any() else 0.0)
    smooth = median_filter(hw, size=win, mode="nearest")
    resid = h_s - smooth
    del hw, smooth

    # distância à borda da própria trilha
    idx = np.arange(n)
    first = np.flatnonzero(np.r_[True, tid_s[1:] != tid_s[:-1]])
    last = np.flatnonzero(np.r_[tid_s[1:] != tid_s[:-1], True])
    starts = np.zeros(n_tracks, dtype=np.int64)
    ends = np.zeros(n_tracks, dtype=np.int64)
    starts[tid_s[first]] = first
    ends[tid_s[last]] = last
    interior = ((idx - starts[tid_s]) >= half) & ((ends[tid_s] - idx) >= half)
    del starts, ends, first, last, idx

    # MAD LOCAL móvel (ver nota no topo do módulo)
    mad_win = max(win * ft.mad_window_factor, win)
    mad_win += 1 - mad_win % 2
    abs_res = np.where(np.isfinite(resid), np.abs(resid), 0.0)
    mad = 1.4826 * median_filter(abs_res, size=int(mad_win), mode="nearest")
    del abs_res
    mad = np.maximum(np.where(np.isfinite(mad), mad, ft.mad_floor_m), ft.mad_floor_m)

    bad_s = interior & np.isfinite(resid) & (np.abs(resid) > ft.n_sigma * mad)
    n_bad = int(bad_s.sum())
    n_edge = int((~interior).sum())
    del resid, mad, interior

    # volta para a ordem ORIGINAL
    bad = np.empty(n, dtype=bool)
    tid = np.empty(n, dtype=np.int64)
    bad[order] = bad_s
    tid[order] = tid_s
    del order, bad_s, tid_s

    stats = {"n_tracks": n_tracks, "n_bad": n_bad, "n_edge": n_edge,
             "window": win, "mad_window": int(mad_win)}
    logger.info(
        f"filttrack: {n_tracks:,} trilhas | janela {win}, {ft.n_sigma:.1f}×MAD local "
        f"(piso {ft.mad_floor_m} m) | rejeitados {n_bad:,}/{n:,} "
        f"({100*n_bad/max(n,1):.3f}%) | {n_edge:,} de borda não filtrados")
    return bad, tid, stats


def filter_along_track(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Marca e remove blunders along-track.

    Retorna uma cópia de `df` SEM os pontos rejeitados (índice reindexado),
    com a coluna `track_id` adicionada.

    NOTA DE MEMÓRIA: esta é a versão conveniente, que materializa o resultado.
    Para o conjunto completo (20 M linhas) use
    `compute_along_track_mask` + aplicação em streaming (ver
    pipelines/run_filttrack.py) — materializar entrada e saída ao mesmo tempo
    não cabe em ~3 GB livres.
    """
    logger = get_logger()
    if not cfg.filttrack.enabled:
        logger.info("filttrack desabilitado — nada a fazer.")
        return df
    if "beam" not in df.columns or "t_year" not in df.columns:
        raise ValueError("filttrack exige as colunas 'beam' e 't_year'.")

    hcol = _height_column(df)
    bad, tid, _ = compute_along_track_mask(
        df["beam"].to_numpy(), df["t_year"].to_numpy(),
        df[hcol].to_numpy(dtype=np.float64), cfg)
    out = df.loc[~bad].copy()
    out["track_id"] = tid[~bad]
    return out.reset_index(drop=True)
