"""
pipelines/run_velocity_check.py
===============================
Backlog #2 — cruza os nós de dh/dt com a velocidade MEaSUREs para identificar
zonas de "estabilidade aparente" (dh/dt ≈ 0 mas fluxo rápido).

    data/dhdt/dhdt_nodes.parquet + data/velocity_thwaites.nc
        -> data/interim/dhdt_nodes_velocity.parquet
        -> outputs/tables/velocity_crosscheck.json

Uso: python pipelines/run_velocity_check.py [--profile anual]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.validate.velocity import crosscheck_stable_zones


def main():
    ap = argparse.ArgumentParser(description="Cross-check dh/dt × velocidade.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--stable-dhdt", type=float, default=0.1)
    ap.add_argument("--fast-speed", type=float, default=100.0)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="velocity_check")

    nodes_p = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
    if not nodes_p.exists():
        raise FileNotFoundError(f"{nodes_p} não existe (rode run_dhdt.py).")
    nodes = pd.read_parquet(nodes_p)
    log.info(f"{len(nodes):,} nós dh/dt carregados")

    out, summary = crosscheck_stable_zones(
        nodes, cfg, stable_abs_dhdt=args.stable_dhdt, fast_speed_m_yr=args.fast_speed)

    dst = cfg.paths.interim / "dhdt_nodes_velocity.parquet"
    out.to_parquet(dst, index=False, engine="pyarrow", compression="snappy")
    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    (cfg.paths.tables / "velocity_crosscheck.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"Nós anotados -> {dst}")
    log.info(f"Resumo -> {cfg.paths.tables / 'velocity_crosscheck.json'}")


if __name__ == "__main__":
    main()
