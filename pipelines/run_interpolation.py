"""
pipelines/run_interpolation.py
==============================
Fase 5 — seleciona o método de interpolação por validação cruzada espacial e
gera o mapa de dh/dt com o vencedor.

    data/dhdt/dhdt_nodes.parquet
        -> outputs/tables/interp_cv_metrics.csv   (comparação dos candidatos)
        -> outputs/tables/interp_selection.json    (vencedor + variograma data-driven)
        -> data/interim/dhdt_grid.parquet          (mapa interpolado pelo vencedor)

Uso:
    python pipelines/run_interpolation.py [--profile anual]
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
from thwaites.interp.select import select_interpolator, interpolate_to_grid


def main():
    ap = argparse.ArgumentParser(description="Seleção de interpolador + mapa de dh/dt.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--nodes", default=None,
                    help="arquivo de nós em data/dhdt/ (default: dhdt_nodes.parquet; "
                         "use dhdt_nodes_qc.parquet para interpolar só os nós "
                         "que sobreviveram ao QC)")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="interpolation")

    nodes_path = cfg.paths.dhdt_dir / (args.nodes or "dhdt_nodes.parquet")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nós dh/dt não encontrados: {nodes_path} (rode run_dhdt.py).")
    nodes = pd.read_parquet(nodes_path, engine="pyarrow")
    log.info(f"{len(nodes):,} nós dh/dt carregados")

    result = select_interpolator(nodes, cfg)
    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    result["metrics"].to_csv(cfg.paths.tables / "interp_cv_metrics.csv", index=False)
    with open(cfg.paths.tables / "interp_selection.json", "w", encoding="utf-8") as fp:
        vg = {k: result["variogram"][k] for k in ("model", "nugget", "sill", "range_m", "sse")}
        json.dump({"winner": result["winner"], "variogram": vg}, fp, indent=2)
    log.info(f"Vencedor: {result['winner']} | variograma {result['variogram']['model']} "
             f"range={result['variogram']['range_m']:.0f} m")

    grid = interpolate_to_grid(nodes, cfg, result["winner"], result["variogram"])
    out = cfg.paths.interim / "dhdt_grid.parquet"
    grid.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    log.info(f"Mapa interpolado -> {out} ({len(grid):,} células)")


if __name__ == "__main__":
    main()
