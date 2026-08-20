"""
pipelines/run_firn.py
=====================
Correção de firn: separa mudança de ALTURA de mudança de MASSA.

Faz DUAS coisas, e a primeira não depende de download nenhum:

  1. **Sensibilidade** — quanto uma faixa plausível de dFAC/dt muda o balanço
     de massa. Converte um viés não quantificado numa faixa quantificada, que é
     declarável no artigo. Roda sempre.
  2. **Correção** — se `firn.enabled` e o recorte do GSFC-FDM existir, calcula
     dFAC/dt por célula, produz `dhdt_ice = dh/dt − dFAC/dt` e refaz o balanço
     de massa em altura gelo-equivalente.

    data/interim/dhdt_grid.parquet [+ data/firn_thwaites.nc]
        -> outputs/experiments/<nome>/
             firn_sensitivity.csv
             firn_corrected_grid.parquet     (só se houver o FDM)
             mass_balance_firn.json          (só se houver o FDM)
             manifest.json

Uso:
    python pipelines/run_firn.py --name firn_v1
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
from thwaites.corrections.firn import (
    firn_sensitivity, apply_firn_correction, resolve_firn_path,
)
from thwaites.uncertainty.mass_balance import compute_mass_balance, apply_coverage_mask


def main():
    ap = argparse.ArgumentParser(description="Correção e sensibilidade de firn.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--nodes", default="dhdt_nodes_qc.parquet",
                    help="nós em data/dhdt/. Default = os que passaram no QC de "
                         "posição. A máscara de cobertura é distância-aos-nós: "
                         "usar o conjunto não filtrado estende a cobertura para "
                         "células sobre oceano e plataforma e infla o dM/dt.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                       run_name=f"firn_{args.name}")

    grid_p = cfg.paths.interim / "dhdt_grid.parquet"
    nodes_p = cfg.paths.dhdt_dir / args.nodes
    if not nodes_p.exists():
        nodes_p = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
    if not grid_p.exists():
        raise FileNotFoundError(f"{grid_p} não existe (rode run_interpolation.py).")

    man = Manifest(cfg, args.name,
                   purpose="Correção de firn (GSFC-FDM) + sensibilidade",
                   overwrite=args.overwrite, seed=0)
    man.set("physics", {
        "decomposition": "dh/dt = dh_gelo/dt + dFAC/dt",
        "mass": "dM/dt = rho_ice * (dh/dt - dFAC/dt) * A",
        "why": ("a densificação do firn baixa a superfície SEM perder massa; "
                "rho=917 sobre o dh/dt bruto atribui isso a perda de gelo"),
        "reference": "IMBIE; Smith et al. 2020; Medley et al. 2022 (GSFC-FDM)",
    })
    man.set("firn_epoch_note", cfg.firn.epoch_note)
    man.add_input(grid_p)

    grid = pd.read_parquet(grid_p, engine="pyarrow")
    nodes = pd.read_parquet(nodes_p, engine="pyarrow") if nodes_p.exists() else None
    if nodes is not None:
        grid = apply_coverage_mask(grid, nodes, cfg.mass_balance.coverage_dist_m)
    log.info(f"{len(grid):,} células cobertas")

    # ---------------- 1. sensibilidade (sempre) ------------------------------
    log.info("Sensibilidade a dFAC/dt (não depende do FDM)...")
    sens = firn_sensitivity(grid, cfg)
    sp = man.path_for("firn_sensitivity.csv")
    sens.to_csv(sp, index=False)
    man.add_output(sp)
    log.info("\n" + sens.to_string(index=False))
    span = float(sens["dMdt_Gt_yr"].max() - sens["dMdt_Gt_yr"].min())
    man.set("sensitivity_span_Gt_yr", span)
    log.info(f"Faixa de dM/dt no intervalo testado de dFAC/dt: {span:.1f} Gt/ano")

    # ---------------- 2. correção (se houver o FDM) --------------------------
    fpath = resolve_firn_path(cfg)
    if not cfg.firn.enabled:
        log.warning("firn.enabled=false — correção não aplicada. "
                    "Rode pipelines/fetch_firn.py e ative na config.")
    elif not fpath.exists():
        log.warning(f"{fpath} não existe — correção não aplicada. "
                    f"Rode pipelines/fetch_firn.py.")
    else:
        man.add_input(fpath)
        log.info("Aplicando correção de firn...")
        corrected, info = apply_firn_correction(grid, cfg)
        cp = man.path_for("firn_corrected_grid.parquet")
        corrected.to_parquet(cp, index=False, engine="pyarrow", compression="snappy")
        man.add_output(cp)
        man.set("firn_info", info)

        L = cfg.mass_balance.correlation_length_m
        if L is None:
            sel = cfg.paths.tables / "interp_selection.json"
            L = (json.loads(sel.read_text())["variogram"]["range_m"]
                 if sel.exists() else 20_000.0)

        raw = compute_mass_balance(grid, cfg, correlation_length_m=L)
        ice = compute_mass_balance(corrected, cfg, correlation_length_m=L,
                                   value_col="dhdt_ice")
        delta = ice["dMdt_Gt_yr"] - raw["dMdt_Gt_yr"]
        out = {"raw": raw, "firn_corrected": ice,
               "delta_Gt_yr": delta,
               "delta_percent": 100 * delta / abs(raw["dMdt_Gt_yr"]),
               "firn_info": info, "epoch_note": cfg.firn.epoch_note}
        mp = man.path_for("mass_balance_firn.json")
        mp.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        man.add_output(mp)
        log.info(f"dM/dt bruto        : {raw['dMdt_Gt_yr']:+.2f} "
                 f"± {raw['sigma_dMdt_Gt_yr_correlated']:.2f} Gt/ano")
        log.info(f"dM/dt com firn     : {ice['dMdt_Gt_yr']:+.2f} "
                 f"± {ice['sigma_dMdt_Gt_yr_correlated']:.2f} Gt/ano")
        log.info(f"efeito do firn     : {delta:+.2f} Gt/ano "
                 f"({out['delta_percent']:+.1f}%)")
        if info.get("extrapolated"):
            log.warning("A taxa de FAC foi EXTRAPOLADA além de jun/2022 — declarar.")

    man.write()
    log.info(f"Experimento em {man.dir}")


if __name__ == "__main__":
    main()
