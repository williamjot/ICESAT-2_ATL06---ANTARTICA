"""
pipelines/run_timeseries.py
===========================
Constrói a série temporal por nó (nó × ano) a partir dos tiles e aplica o
teste de tendência formal (Mann-Kendall + Sen's slope + FDR).

    data/tiles/*.parquet
        -> data/timeseries/node_series.parquet   (série longa)
        -> data/timeseries/node_trends.parquet    (tendência por nó)

Uso:
    python pipelines/run_timeseries.py [--profile anual]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.grid.tiles import load_manifest
from thwaites.timeseries.build import build_node_series
from thwaites.timeseries.trend import compute_trends


def main():
    ap = argparse.ArgumentParser(description="Séries temporais + tendência por nó.")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="timeseries")
    cfg.paths.timeseries_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(cfg)
    log.info(f"Construindo séries em {len(manifest)} tiles...")

    parts = []
    for e in manifest:
        tdf = pd.read_parquet(cfg.paths.tiles_dir / e["file"], engine="pyarrow")
        s = build_node_series(tdf, cfg, e["x_min"], e["x_max"], e["y_min"], e["y_max"])
        if len(s):
            s["tile"] = e["tile"]
            parts.append(s)

    if not parts:
        log.warning("Nenhuma série construída.")
        return

    series = pd.concat(parts, ignore_index=True)
    series_path = cfg.paths.timeseries_dir / "node_series.parquet"
    series.to_parquet(series_path, index=False, engine="pyarrow", compression="snappy")
    log.info(f"Série -> {series_path} ({len(series):,} linhas nó×ano)")

    trends = compute_trends(series, cfg)
    trends_path = cfg.paths.timeseries_dir / "node_trends.parquet"
    trends.to_parquet(trends_path, index=False, engine="pyarrow", compression="snappy")
    log.info(f"Tendência -> {trends_path} ({len(trends):,} nós)")


if __name__ == "__main__":
    main()
