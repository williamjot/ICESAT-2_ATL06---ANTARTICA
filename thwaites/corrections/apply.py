"""
thwaites.corrections.apply
==========================
Aplica correções geofísicas explicitamente sobre a elevação bruta `h_elv`,
produzindo `h_corr`. As correções (tide_ocean, dac) vêm como colunas
separadas da extração — aqui elas são subtraídas, de forma configurável e
rastreável.

CONVENÇÃO DE SINAL  [VERIFICADO]:
    h_corr = h_elv - Σ(correções aplicadas)
As variáveis do grupo `geophysical` são SUBTRAÍDAS para remover o sinal de
maré/atmosfera. Fontes primárias:
  - ATL06 ATBD r004, p.45: o grupo geophysical contém correções "that may be
    added to or removed from h_li" -> tide_ocean/dac NÃO vêm aplicadas em h_li.
  - Data Comparison User's Guide v006, p.17: já aplicadas em h_li (herdadas do
    ATL03): ocean loading, solid earth pole tide, ocean pole tide, solid earth
    tide -> NÃO reaplicar.
  - Idem, p.6: o detide é uma SUBTRAÇÃO ("a subtraction is made ...").

GATING (gelo flutuante):
    Maré oceânica e DAC atuam sobre a superfície do OCEANO — só afetam gelo
    flutuante. Com `corrections.gate_to_floating=True`, as correções só são
    aplicadas onde `mask_class == mask.floating_class`; sobre gelo aterrado
    ficam zeradas (gelo aterrado não sobe com a maré). Isso exige que o passo
    de máscara já tenha rodado (coluna `mask_class`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger


def apply_corrections(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Retorna uma cópia de `df` com a coluna `h_corr` (float32) adicionada.

    Levanta ValueError se uma correção pedida não estiver no DataFrame, ou se
    o gating por gelo flutuante for pedido sem a coluna `mask_class`.
    """
    logger = get_logger()
    selected = list(cfg.corrections.apply)
    eq_mode = cfg.corrections.equilibrium_tide_mode
    has_eq = "tide_equilibrium" in selected
    if eq_mode == "apply_atl06":
        if cfg.cats.enabled and cfg.cats.equilibrium_tide_included is not False:
            raise ValueError(
                "equilibrium_tide_mode='apply_atl06' exige declarar "
                "cats.equilibrium_tide_included=false; sem isso há risco de "
                "duplicar a maré de longo período na substituição CATS.")
        if not has_eq:
            selected.append("tide_equilibrium")
    elif has_eq:
        raise ValueError(
            "'tide_equilibrium' em corrections.apply exige "
            "equilibrium_tide_mode='apply_atl06'. Escolha explicitamente se "
            "o modelo de maré substituto já inclui esse componente.")
    gate = cfg.corrections.gate_to_floating

    if gate and "mask_class" not in df.columns:
        raise ValueError(
            "corrections.gate_to_floating=True exige a coluna 'mask_class'. "
            "Rode o passo de máscara (thwaites.qc.mask) antes das correções."
        )

    n = len(df)
    total = np.zeros(n, dtype=np.float64)

    if gate:
        floating = (df["mask_class"].to_numpy() == cfg.mask.floating_class)
    else:
        floating = np.ones(n, dtype=bool)

    for name in selected:
        if name not in df.columns:
            raise ValueError(f"correção '{name}' não está no DataFrame (colunas: {list(df.columns)})")
        vals = df[name].to_numpy(dtype=np.float64)
        # fill/NaN -> sem correção (0), não descarta o ponto
        contrib = np.where(np.isnan(vals), 0.0, vals)
        # gating: fora do gelo flutuante, contribuição zero
        contrib = np.where(floating, contrib, 0.0)

        n_applied = int(np.count_nonzero(contrib))
        logger.info(f"correção '{name}': aplicada em {n_applied:,}/{n:,} pontos "
                    f"(média {np.nanmean(np.where(contrib != 0, contrib, np.nan)) if n_applied else 0:.4f} m)")
        total += contrib

    out = df.copy()
    out["h_corr"] = (out["h_elv"].to_numpy(dtype=np.float64) - total).astype(np.float32)
    return out
