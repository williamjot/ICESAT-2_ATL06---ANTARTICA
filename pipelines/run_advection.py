"""
pipelines/run_advection.py
==========================
Quantifica o termo de advecção v·∇h e produz o dh/dt Lagrangiano.

O OBJETIVO É MEDIR, NÃO CORRIGIR O BALANÇO. A conservação de massa do projeto
é Euleriana e o ∇·(H·v) já contém a advecção; aplicar a conversão Lagrangiana
ao balanço contaria o efeito duas vezes. O que este passo entrega é:

  - quão grande é v·∇h comparado ao próprio dh/dt (é 5% ou 50% do sinal?);
  - onde ele é grande (o padrão espacial de dh/dt é interpretável como
    adelgaçamento dinâmico, ou está dominado por topografia advectada?);
  - quão sensível o termo é à suavização do declive — se variar muito entre
    escalas, é ruído do DEM, não sinal.

    data/interim/dhdt_grid.parquet + velocidade + REMA
        -> outputs/experiments/<nome>/
             advection_sensitivity.csv
             dhdt_lagrangian_grid.parquet
             advection_summary.json
             manifest.json

Uso: python pipelines/run_advection.py --name adv_v1
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
from thwaites.glaciology.advection import (
    advection_term, to_lagrangian, advection_sensitivity,
)
from thwaites.uncertainty.mass_balance import apply_coverage_mask


def main():
    ap = argparse.ArgumentParser(description="Termo de advecção e dh/dt Lagrangiano.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--smooth-km", type=float, default=5.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"advection_{args.name}")

    grid_p = cfg.paths.interim / "dhdt_grid.parquet"
    nodes_p = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
    vel_p = cfg.paths.data_dir / cfg.velocity.path
    for p, cmd in ((grid_p, "run_interpolation.py"), (vel_p, "fetch_velocity.py")):
        if not p.exists():
            raise FileNotFoundError(f"{p} não existe (rode {cmd}).")

    man = Manifest(cfg, args.name,
                   purpose="Quantificação do termo de advecção (dh/dt Lagrangiano)",
                   overwrite=args.overwrite, seed=0)
    man.set("physics", {
        "relation": "Dh/Dt = dh/dt|Euleriano + v.grad(h)",
        "scope": ("MEDIR o termo, nao corrigir o balanco de massa: a "
                  "conservacao usada e Euleriana e div(H*v) ja contem a "
                  "adveccao — aplicar a conversao la seria contagem dupla"),
        "matters_for": ["padrao espacial de dh/dt",
                        "dh/dt pontual como afinamento de coluna",
                        "formulacao Lagrangiana (Moholdt 2014, Shean 2019)"],
    })
    man.set("velocity_epoch", cfg.velocity.epoch_note)
    man.add_input(grid_p).add_input(vel_p)

    grid = pd.read_parquet(grid_p, engine="pyarrow")
    if nodes_p.exists():
        nodes = pd.read_parquet(nodes_p, engine="pyarrow")
        grid = apply_coverage_mask(grid, nodes, cfg.mass_balance.coverage_dist_m)
    gx = np.sort(grid["x"].unique())
    gy = np.sort(grid["y"].unique())
    log.info(f"{len(grid):,} células | grade {len(gy)} × {len(gx)}")

    # ---- 1. sensibilidade à suavização do declive --------------------------
    log.info("Sensibilidade do termo de advecção à suavização do declive...")
    sens = advection_sensitivity(gx, gy, cfg)
    sp = man.path_for("advection_sensitivity.csv")
    sens.to_csv(sp, index=False)
    man.add_output(sp)
    log.info("\n" + sens[["smooth_km", "adv_abs_median_m_yr", "adv_p90_abs_m_yr",
                          "speed_median_m_yr", "slope_median"]].to_string(index=False))
    spread = float(sens["adv_abs_median_m_yr"].max() / max(sens["adv_abs_median_m_yr"].min(), 1e-9))
    man.set("sensitivity_ratio_max_over_min", spread)
    if spread > 3:
        log.warning(f"|v·∇h| varia {spread:.1f}× entre escalas de suavização — "
                    f"o termo é sensível ao ruído do DEM; tratar como ordem de "
                    f"grandeza, não como correção precisa.")

    # ---- 2. termo na escala escolhida + Lagrangiano ------------------------
    adv, info = advection_term(gx, gy, cfg, smooth_km=args.smooth_km)
    GX, GY = np.meshgrid(gx, gy)
    flat = pd.DataFrame({"x": GX.ravel(), "y": GY.ravel(), "advection": adv.ravel()})
    merged = grid.merge(flat, on=["x", "y"], how="left")
    merged["dhdt_lagrangian"] = to_lagrangian(merged["pred"], merged["advection"])

    op = man.path_for("dhdt_lagrangian_grid.parquet")
    merged.to_parquet(op, index=False, engine="pyarrow", compression="snappy")
    man.add_output(op)

    e = merged["pred"].to_numpy(float)
    a = merged["advection"].to_numpy(float)
    l = merged["dhdt_lagrangian"].to_numpy(float)
    ok = np.isfinite(e) & np.isfinite(a)
    ratio = (float(np.nanmedian(np.abs(a[ok])) / np.nanmedian(np.abs(e[ok])))
             if ok.any() else np.nan)
    summary = {
        **info,
        "n_cells": int(ok.sum()),
        "dhdt_eulerian_median": float(np.nanmedian(e[ok])),
        "dhdt_lagrangian_median": float(np.nanmedian(l[ok])),
        "advection_median": float(np.nanmedian(a[ok])),
        "abs_advection_over_abs_dhdt": ratio,
        "sensitivity_ratio": spread,
        "scope_note": ("NAO aplicar ao balanco de massa Euleriano — "
                       "div(H*v) ja contem a adveccao (contagem dupla)"),
    }
    sj = man.path_for("advection_summary.json")
    sj.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    man.add_output(sj)
    man.set("result", summary).write()

    log.info(f"dh/dt Euleriano  : {summary['dhdt_eulerian_median']:+.4f} m/ano")
    log.info(f"termo v·∇h       : {summary['advection_median']:+.4f} m/ano "
             f"(|·| mediano {info['adv_abs_median_m_yr']:.4f})")
    log.info(f"dh/dt Lagrangiano: {summary['dhdt_lagrangian_median']:+.4f} m/ano")
    log.info(f"|advecção| / |dh/dt| = {100*ratio:.1f}% do sinal")
    log.info(f"Experimento em {man.dir}")


if __name__ == "__main__":
    main()
