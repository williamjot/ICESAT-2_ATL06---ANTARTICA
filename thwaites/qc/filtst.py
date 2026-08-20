"""
thwaites.qc.filtst
==================
Filtragem ESPAÇO-TEMPORAL (inspirada no `filtst.py` do captoolkit).

Complementa o `filttrack`: aquele olha ao longo de UMA trilha; este olha para
a vizinhança 2D + tempo, pegando pontos que só se revelam anômalos quando
comparados a passagens vizinhas do MESMO período (ex.: uma trilha inteira
deslocada por nuvem, que o filtro along-track não detecta porque o viés é
coerente ao longo dela).

DESIGN — binagem em vez de janela móvel por ponto:
o captoolkit usa uma janela móvel; com ~20 M de pontos, uma consulta KDTree por
ponto é inviável. Aqui os pontos são binados em células espaço-temporais
(`cell_km` × `cell_km` × `dt_years`) e a estatística robusta (mediana/MAD) é
calculada por célula com um `groupby` — O(n), memória previsível e resultado
equivalente para o propósito de detectar outliers. A diferença é que a
vizinhança é uma célula fixa, não centrada no ponto; para reduzir o efeito de
borda, use `passes: 2` (a 2ª passagem usa a grade deslocada de meia célula).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger


def _height_column(df: pd.DataFrame) -> str:
    for c in ("h_res", "h_corr", "h_elv"):
        if c in df.columns:
            return c
    raise ValueError("nenhuma coluna de elevação (h_res/h_corr/h_elv)")


def _flag_pass(x, y, t, h, cell_m, dt_years, n_sigma, min_count,
               mad_floor, offset=0.0):
    """Marca outliers numa passagem de binagem. Retorna máscara booleana `bad`."""
    ix = np.floor((x + offset * cell_m) / cell_m).astype(np.int64)
    iy = np.floor((y + offset * cell_m) / cell_m).astype(np.int64)
    it = np.floor((t + offset * dt_years) / dt_years).astype(np.int64)

    # chave de célula compacta (evita tuplas; primos grandes reduzem colisão)
    key = (ix * 73856093) ^ (iy * 19349663) ^ (it * 83492791)

    s = pd.Series(h)
    g = s.groupby(key)
    med = g.transform("median").to_numpy()
    cnt = g.transform("size").to_numpy()
    resid = h - med
    mad = 1.4826 * pd.Series(np.abs(resid)).groupby(key).transform("median").to_numpy()
    mad = np.maximum(np.where(np.isfinite(mad), mad, mad_floor), mad_floor)

    # células com poucos pontos não permitem estatística robusta -> não filtra
    return (cnt >= min_count) & np.isfinite(resid) & (np.abs(resid) > n_sigma * mad)


def filter_space_time(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Remove outliers espaço-temporais. Exige colunas x, y, t_year e elevação.

    Retorna cópia sem os pontos rejeitados (índice reindexado).
    """
    logger = get_logger()
    fs = cfg.filtst
    if not fs.enabled:
        logger.info("filtst desabilitado — nada a fazer.")
        return df

    for c in ("x", "y", "t_year"):
        if c not in df.columns:
            raise ValueError(f"filtst exige a coluna '{c}' (rode assign_xy/tiles antes).")

    hcol = _height_column(df)
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    t = df["t_year"].to_numpy(dtype=np.float64)
    h = df[hcol].to_numpy(dtype=np.float64)

    cell_m = fs.cell_km * 1000.0
    bad = np.zeros(len(df), dtype=bool)
    for p in range(int(fs.passes)):
        offset = 0.0 if p == 0 else 0.5      # 2ª passagem: grade deslocada
        keep = ~bad
        flag = np.zeros(len(df), dtype=bool)
        flag[keep] = _flag_pass(
            x[keep], y[keep], t[keep], h[keep],
            cell_m, fs.dt_years, fs.n_sigma, fs.min_count, fs.mad_floor_m, offset)
        n_new = int(flag.sum())
        bad |= flag
        logger.info(f"filtst passagem {p+1}/{fs.passes} "
                    f"(offset {offset:.1f} célula): +{n_new:,} rejeitados")

    n_bad = int(bad.sum())
    out = df.loc[~bad].reset_index(drop=True)
    logger.info(
        f"filtst: células {fs.cell_km} km × {fs.dt_years} ano, {fs.n_sigma:.1f}×MAD "
        f"(mín {fs.min_count} pts/célula) | rejeitados {n_bad:,}/{len(df):,} "
        f"({100*n_bad/max(len(df),1):.3f}%)")
    return out
