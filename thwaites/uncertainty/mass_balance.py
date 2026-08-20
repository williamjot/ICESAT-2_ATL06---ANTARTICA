"""
thwaites.uncertainty.mass_balance
=================================
Propaga o campo de dh/dt para taxa de perda de massa (Gt/ano) e contribuição
ao nível do mar (mm/ano), COM incerteza espacialmente correlacionada.

Cadeia:
    dV/dt = Σ (dh/dt_i · A_célula)                      [m³/ano]
    dM/dt = ρ_gelo · dV/dt                              [kg/ano -> Gt/ano]
    SLE   = -dM/dt / (área_oceânica → Gt por mm)        [mm/ano]  (perda -> subida)

PROPAGAÇÃO DE INCERTEZA:
Erros altimétricos são ESPACIALMENTE CORRELACIONADOS. Somar σ das células como
independentes subestima a incerteza. Com células de área a, N células, desvio
médio σ̄ e comprimento de correlação L:

    Var(ΣV) ≈ a²·σ̄²·ΣΣρ_ij ≈ σ̄²·A_total·A_corr,   A_corr = π·L²
    =>  σ_V(correlacionado) = σ̄·√(A_total · A_corr)

que é √(A_corr/A_célula) MAIOR que a soma ingênua σ̄·√(A_total·A_célula).
Reportamos AMBOS para deixar a diferença explícita e honesta.

PREMISSA DECLARADA: ρ_gelo = 917 kg/m³ (adelgaçamento dinâmico). A densidade é
um erro SISTEMÁTICO não propagado aqui — declarada, não escondida.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger

KG_PER_GT = 1e12


def two_component_mass_sigma(sigma_white: float, L_white: float,
                             sigma_corr: float, L_corr: float,
                             area_total_m2: float, cell_area_m2: float,
                             ice_density: float) -> dict:
    """
    Incerteza de massa com DUAS componentes de erro de comprimentos de
    correlação diferentes.

    A fórmula de componente única obriga a escolher um L para tudo, e é dessa
    escolha que vinha a indeterminação de fator 4,5 na barra de erro (σ entre
    3,25 e 14,74 Gt/ano conforme L fosse 34,1 ou 154,4 km). Mas os dois erros
    não têm a mesma estrutura espacial:

      * `sigma_white`  — dispersão residual do ajuste, essencialmente branca;
                         a área de correlação é a da própria célula.
      * `sigma_corr`   — erro de apontamento projetado na declividade, que o
                         ATBD ATL14/ATL15 §3.4.4 declara correlacionado em
                         "dezenas de km".

    Somando as variâncias com a área de correlação de CADA uma:

        Var(V) = σ_w²·A_total·A_célula + σ_c²·A_total·(π·L_corr²)

    Isso substitui uma escolha arbitrária por uma atribuição física. O L da
    componente branca não é livre: é o tamanho da célula, por construção.
    """
    Aw = cell_area_m2 if L_white is None else math.pi * L_white ** 2
    Ac = math.pi * L_corr ** 2
    var_V = (sigma_white ** 2 * area_total_m2 * Aw
             + sigma_corr ** 2 * area_total_m2 * Ac)
    sigma_V = math.sqrt(max(var_V, 0.0))
    to_gt = ice_density / KG_PER_GT
    return {
        "sigma_V_m3_yr": sigma_V,
        "sigma_dMdt_Gt_yr": sigma_V * to_gt,
        "contrib_branca_Gt_yr": (math.sqrt(sigma_white ** 2 * area_total_m2 * Aw)
                                 * to_gt),
        "contrib_correlacionada_Gt_yr": (
            math.sqrt(sigma_corr ** 2 * area_total_m2 * Ac) * to_gt),
        "L_branca_m": L_white,
        "L_correlacionada_m": L_corr,
    }


def gt_per_mm_sle(ocean_area_m2: float) -> float:
    """Gt de gelo equivalentes a 1 mm de subida do nível do mar."""
    # 1 mm sobre a área oceânica = area*1e-3 m³ de água = *1000 kg -> /1e12 Gt
    return ocean_area_m2 * 1e-3 * 1000.0 / KG_PER_GT   # = area_m2 * 1e-12


def apply_coverage_mask(grid_df: pd.DataFrame, nodes_df: pd.DataFrame,
                        max_dist_m: float) -> pd.DataFrame:
    """Mantém só as células a <= max_dist_m de algum nó dh/dt real."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[nodes_df["x"].to_numpy(), nodes_df["y"].to_numpy()])
    d, _ = tree.query(np.c_[grid_df["x"].to_numpy(), grid_df["y"].to_numpy()], k=1)
    return grid_df.loc[d <= max_dist_m].copy()


def compute_mass_balance(grid_df: pd.DataFrame, cfg: Config,
                         correlation_length_m: float,
                         value_col: str = "pred", var_col: str = "var") -> dict:
    """
    Calcula dV/dt, dM/dt (Gt/ano) e SLE (mm/ano) com incerteza correlacionada.

    `grid_df` deve ser a grade regular já restrita à cobertura (ver
    apply_coverage_mask). `correlation_length_m` = alcance do variograma (Fase 5).
    """
    logger = get_logger()
    a = cfg.interpolation.grid_res_m ** 2                 # área da célula (m²)
    mb = cfg.mass_balance

    val = grid_df[value_col].to_numpy(dtype=float)
    ok = ~np.isnan(val)
    val = val[ok]
    if val.size == 0:
        raise ValueError("Nenhuma célula válida para o balanço de massa.")
    n = val.size
    A_total = n * a

    # ---- valor central --------------------------------------------------
    dVdt = float(np.sum(val) * a)                         # m³/ano
    dMdt_Gt = mb.ice_density * dVdt / KG_PER_GT           # Gt/ano
    gpm = gt_per_mm_sle(mb.ocean_area_m2)
    sle_mm = -dMdt_Gt / gpm                               # perda -> subida (+)

    # ---- incerteza ------------------------------------------------------
    if var_col in grid_df.columns:
        var = grid_df[var_col].to_numpy(dtype=float)[ok]
        var = var[np.isfinite(var) & (var >= 0)]
        sigma_bar = float(np.sqrt(np.mean(var))) if var.size else 0.0
    else:
        sigma_bar = 0.0
        logger.warning("sem coluna de variância — incerteza do dh/dt = 0.")

    # decomposição da incerteza por célula, quando disponível: quanto vem do
    # erro dos NÓS (herdado) e quanto do erro de PREDIÇÃO espacial. Sem isso
    # não se sabe qual termo dominar para melhorar o resultado.
    parts = {}
    for _name, _col in (("input_nodes", "sigma_input"), ("interp", "var_interp")):
        if _col in grid_df.columns:
            _arr = grid_df[_col].to_numpy(dtype=float)[ok]
            _arr = _arr[np.isfinite(_arr)]
            if _arr.size:
                # sigma_input já é σ; var_interp é variância
                _s = (float(np.sqrt(np.mean(_arr ** 2))) if _col == "sigma_input"
                      else float(np.sqrt(np.mean(np.abs(_arr)))))
                parts[f"sigma_from_{_name}_m_yr"] = _s

    A_corr = math.pi * correlation_length_m ** 2
    sigma_V_indep = sigma_bar * math.sqrt(A_total * a)
    sigma_V_corr = sigma_bar * math.sqrt(A_total * A_corr)

    def to_mass_sle(sigma_V):
        sM = mb.ice_density * sigma_V / KG_PER_GT
        return sM, sM / gpm

    sM_indep, sSLE_indep = to_mass_sle(sigma_V_indep)
    sM_corr, sSLE_corr = to_mass_sle(sigma_V_corr)

    result = {
        "n_cells": n,
        "cell_area_km2": a / 1e6,
        "area_total_km2": A_total / 1e6,
        "correlation_length_m": correlation_length_m,
        "dhdt_mean_m_yr": float(np.mean(val)),
        "dVdt_m3_yr": dVdt,
        "dMdt_Gt_yr": dMdt_Gt,
        "sle_mm_yr": sle_mm,
        "ice_density": mb.ice_density,
        # incerteza (1σ)
        "sigma_dhdt_bar_m_yr": sigma_bar,
        "sigma_dMdt_Gt_yr_independent": sM_indep,
        "sigma_dMdt_Gt_yr_correlated": sM_corr,
        "sigma_sle_mm_yr_independent": sSLE_indep,
        "sigma_sle_mm_yr_correlated": sSLE_corr,
        "inflation_factor": (sigma_V_corr / sigma_V_indep) if sigma_V_indep > 0 else float("nan"),
        **parts,
    }
    logger.info(
        f"Balanço de massa: dM/dt = {dMdt_Gt:+.2f} ± {sM_corr:.2f} Gt/ano (correlacionado) "
        f"| SLE = {sle_mm:+.4f} ± {sSLE_corr:.4f} mm/ano | "
        f"fator de inflação da incerteza x{result['inflation_factor']:.1f} "
        f"(vs soma independente)")
    return result
