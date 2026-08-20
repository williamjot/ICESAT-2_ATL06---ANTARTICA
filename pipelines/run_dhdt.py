"""
pipelines/run_dhdt.py
=====================
Calcula dh/dt (fitsec) em todos os tiles.

    data/tiles/*.parquet  ->  data/dhdt/*_dhdt.parquet + dhdt_nodes.parquet

Uso:
    python pipelines/run_dhdt.py [--profile anual]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.timeseries.dhdt import run_dhdt


def main():
    ap = argparse.ArgumentParser(description="Calcula dh/dt por tile.")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="dhdt")
    run_dhdt(cfg)


if __name__ == "__main__":
    main()
