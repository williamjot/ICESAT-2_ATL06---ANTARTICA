"""
pipelines/run_gia.py
====================
Aplica a correção de movimento vertical do embasamento (GIA) à grade de dh/dt e
recalcula o balanço de massa, reportando ANTES e DEPOIS.

    data/interim/dhdt_grid.parquet + data/gia/GIA_maps_Caron_et_al_2018
        -> outputs/experiments/<nome>/
             dhdt_grid_gia.parquet     (grade com dhdt_ice_gia e vlm)
             mass_balance_gia.json     (antes/depois + decomposição da incerteza)

Por que um pipeline separado e não uma flag em run_mass_balance
---------------------------------------------------------------
O produto sem GIA já existe, está citado, e tem de continuar reproduzível. O
valor científico aqui está justamente na COMPARAÇÃO — sobrescrever o produto
antigo destruiria a evidência do tamanho do viés.

Uso: python pipelines/run_gia.py --name gia_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.experiments.manifest import Manifest
from thwaites.corrections.gia import (GIAField, correct_elevation_rate,
                                      systematic_mass_uncertainty)
from thwaites.uncertainty.mass_balance import (apply_coverage_mask,
                                               compute_mass_balance)

CARON_DEFAULT = "gia/GIA_maps_Caron_et_al_2018"

# GPS de Barletta et al. (2018) no ASE. NÃO é um campo — é um valor de pico
# num punhado de estações. Serve só para dimensionar o quanto o Caron pode
# estar subestimando, jamais como correção a aplicar.
BARLETTA_ASE_MM_YR = 41.0


def _bracket(covered, cfg, antes, log) -> dict:
    """
    Quanto valeria a correção se o soerguimento real fosse o de Barletta?

    Teste de ESCALA, não correção alternativa. O ponto é mostrar que a
    incerteza ESTRUTURAL do GIA (qual modelo) domina de longe a incerteza
    INTERNA do Caron (σ do ensemble), e que reportar apenas a segunda daria uma
    falsa sensação de controle.
    """
    rho = cfg.mass_balance.ice_density
    area_m2 = antes["area_total_km2"] * 1e6
    vlm_caron = float(np.nanmedian(covered["vlm"]))
    ef_caron = -vlm_caron * area_m2 * rho / 1e12
    ef_barletta = -(BARLETTA_ASE_MM_YR * 1e-3) * area_m2 * rho / 1e12
    log.warning(
        f"ESTRUTURAL: Caron mediano {1e3*vlm_caron:+.2f} mm/ano vs GPS de "
        f"Barletta {BARLETTA_ASE_MM_YR:+.0f} mm/ano no ASE — fator "
        f"{BARLETTA_ASE_MM_YR/(1e3*vlm_caron):.0f}x. Efeito na massa: "
        f"{ef_caron:+.2f} vs {ef_barletta:+.2f} Gt/ano.")
    return {
        "pergunta": ("o σ do ensemble do Caron cobre a discrepância com o GPS "
                     "do ASE?"),
        "vlm_caron_mediano_mm_ano": 1e3 * vlm_caron,
        "vlm_caron_sigma_mediano_mm_ano": float(1e3 * np.nanmedian(covered["vlm_sigma"])),
        "vlm_barletta_gps_ase_mm_ano": BARLETTA_ASE_MM_YR,
        "razao": BARLETTA_ASE_MM_YR / (1e3 * vlm_caron),
        "efeito_na_massa_caron_Gt_ano": ef_caron,
        "efeito_na_massa_escala_barletta_Gt_ano": ef_barletta,
        "conclusao": (
            "NÃO. A discrepância entre o Caron e o GPS é de mais de uma ordem "
            "de grandeza, enquanto o σ do ensemble do Caron é de décimos de "
            "mm/ano. A incerteza do GIA nesta região é ESTRUTURAL (escolha de "
            "modelo/reologia), não interna. Reportar só o σ do Caron subestima "
            "a incerteza real do GIA por um fator grande, e a correção "
            "aplicada aqui deve ser lida como PISO, não como valor definitivo. "
            "Barletta é valor de pico pontual e não é aplicável como campo — "
            "a faixa entre os dois é o que está genuinamente em aberto."),
    }


def main():
    ap = argparse.ArgumentParser(description="Correção GIA + balanço de massa.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--grid", default="dhdt_grid.parquet")
    ap.add_argument("--nodes", default="dhdt_nodes_qc.parquet")
    ap.add_argument("--value-col", default="pred")
    ap.add_argument("--gia-table", default=CARON_DEFAULT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"gia_{args.name}")

    grid_path = cfg.paths.interim / args.grid
    nodes_path = cfg.paths.dhdt_dir / args.nodes
    if not nodes_path.exists():
        nodes_path = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
    gia_path = cfg.paths.data_dir / args.gia_table
    for p in (grid_path, nodes_path, gia_path):
        if not p.exists():
            raise FileNotFoundError(p)

    grid = pd.read_parquet(grid_path, engine="pyarrow")
    nodes = pd.read_parquet(nodes_path, engine="pyarrow")
    if args.value_col not in grid.columns:
        raise ValueError(f"'{args.value_col}' ausente em {grid_path.name}")
    if not {"lon", "lat"} <= set(grid.columns):
        raise ValueError(f"{grid_path.name} sem lon/lat — o GIA é amostrado em "
                         f"coordenadas geográficas, não polares.")

    field = GIAField.from_caron_table(gia_path)
    vlm, vlm_sig = field.sample(grid["lon"].to_numpy(), grid["lat"].to_numpy())
    grid["vlm"] = vlm
    grid["vlm_sigma"] = vlm_sig
    grid["dhdt_gia"] = correct_elevation_rate(grid[args.value_col].to_numpy(), vlm)

    log.info(f"VLM na ROI: mediana {1e3*np.nanmedian(vlm):+.2f} mm/ano "
             f"(p10 {1e3*np.nanpercentile(vlm,10):+.2f}, "
             f"p90 {1e3*np.nanpercentile(vlm,90):+.2f}) | "
             f"σ mediano {1e3*np.nanmedian(vlm_sig):.2f} mm/ano")

    man = Manifest(cfg, args.name,
                   purpose="Correção GIA (Caron et al. 2018) do balanço de massa",
                   overwrite=args.overwrite, seed=0)
    man.add_input(grid_path, columns=[args.value_col, "lon", "lat", "var"])
    man.add_input(gia_path)

    L = cfg.mass_balance.correlation_length_m
    covered = apply_coverage_mask(grid, nodes, cfg.mass_balance.coverage_dist_m)
    log.info(f"{len(covered):,} células dentro da cobertura "
             f"(de {len(grid):,}) | L = {L/1000:.1f} km")

    antes = compute_mass_balance(covered, cfg, correlation_length_m=L,
                                 value_col=args.value_col)
    depois = compute_mass_balance(covered, cfg, correlation_length_m=L,
                                  value_col="dhdt_gia")

    area_m2 = antes["area_total_km2"] * 1e6
    sig_gia = systematic_mass_uncertainty(covered["vlm_sigma"].to_numpy(),
                                          area_m2, cfg.mass_balance.ice_density)
    sig_alt = depois["sigma_dMdt_Gt_yr_correlated"]
    sig_tot = float(np.hypot(sig_alt, sig_gia))

    delta = depois["dMdt_Gt_yr"] - antes["dMdt_Gt_yr"]

    rep = {
        "fonte_gia": field.source,
        "referencia": ("Caron, Ivins, Larour, Adhikari, Nilsson, Blewitt (2018), "
                       "GRL 45, doi:10.1002/2017GL076644"),
        "vlm_na_roi_mm_ano": {
            "mediana": float(1e3 * np.nanmedian(covered["vlm"])),
            "p10": float(1e3 * np.nanpercentile(covered["vlm"], 10)),
            "p90": float(1e3 * np.nanpercentile(covered["vlm"], 90)),
            "sigma_mediano": float(1e3 * np.nanmedian(covered["vlm_sigma"])),
        },
        "antes_sem_gia": {k: antes[k] for k in
                          ("dMdt_Gt_yr", "sle_mm_yr", "dhdt_mean_m_yr",
                           "sigma_dMdt_Gt_yr_correlated", "area_total_km2")},
        "depois_com_gia": {k: depois[k] for k in
                           ("dMdt_Gt_yr", "sle_mm_yr", "dhdt_mean_m_yr",
                            "sigma_dMdt_Gt_yr_correlated", "area_total_km2")},
        "efeito_da_correcao_Gt_ano": float(delta),
        "incerteza": {
            "sigma_altimetria_Gt_ano": float(sig_alt),
            "sigma_gia_Gt_ano": float(sig_gia),
            "sigma_total_Gt_ano": sig_tot,
            "nota_sigma_gia": ("tratada como TOTALMENTE CORRELACIONADA sobre a "
                               "ROI — o σ do Caron vem de dispersão de ensemble "
                               "sobre parâmetros globais, então células vizinhas "
                               "erram JUNTAS; dividir por sqrt(N) seria erro "
                               "grosseiro"),
            "comprimento_correlacao_altimetria_m": L,
        },
        "resultado": {
            "dMdt_Gt_ano": depois["dMdt_Gt_yr"],
            "sigma_Gt_ano": sig_tot,
            "sle_mm_ano": depois["sle_mm_yr"],
        },
        "sensibilidade_estrutural": _bracket(covered, cfg, antes, log),
        "limitacoes": [
            "Caron é grau 89 (1°, ~28 km em lon e ~111 km em lat a 75°S) e "
            "vinculado globalmente: NÃO resolve a resposta de baixa viscosidade "
            "que Barletta et al. (2018) mediram no ASE (+41 mm/ano por GPS, "
            "manto 4e18 Pa·s). A correção aqui é portanto conservadora e a "
            "perda corrigida é LIMITE INFERIOR da perda real.",
            "a correção é aplicada à grade interpolada, não aos pontos ATL06; "
            "isso é exato em primeira ordem porque dB/dt é suave em 1°, mas "
            "significa que o produto de NÓS não carrega a correção",
            "a densidade do gelo (917 kg/m³) continua sendo erro sistemático "
            "não propagado",
        ],
    }

    out_grid = man.path_for("dhdt_grid_gia.parquet")
    grid.to_parquet(out_grid, index=False, engine="pyarrow")
    man.add_output(out_grid)
    out_json = man.path_for("mass_balance_gia.json")
    out_json.write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    man.add_output(out_json)
    man.set("mass_balance_gia", rep["resultado"])
    man.write()

    log.info("=" * 62)
    log.info(f"  sem GIA : {antes['dMdt_Gt_yr']:+8.2f} Gt/ano  "
             f"(SLE {antes['sle_mm_yr']:+.4f} mm/ano)")
    log.info(f"  com GIA : {depois['dMdt_Gt_yr']:+8.2f} Gt/ano  "
             f"(SLE {depois['sle_mm_yr']:+.4f} mm/ano)")
    log.info(f"  efeito  : {delta:+8.2f} Gt/ano")
    log.info(f"  σ: altimetria {sig_alt:.2f} + GIA {sig_gia:.2f} "
             f"-> total {sig_tot:.2f} Gt/ano")
    log.info("=" * 62)
    log.info(f"Experimento em {man.dir}")


if __name__ == "__main__":
    main()
