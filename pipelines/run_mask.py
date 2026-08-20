"""
pipelines/run_mask.py
=====================
Aplica a máscara BedMachine: remove oceano e adiciona `mask_class`.

    data/interim/atl06_merged.parquet -> data/interim/atl06_masked.parquet

MEMÓRIA: streaming por row group. Materializar 24 M linhas exige cerca de
0,8 GB e não escala para uma ROI 2,4× maior. A janela do raster é calculada uma
vez a partir da bbox da ROI e reutilizada em todos os lotes.

Uso: python pipelines/run_mask.py [--profile anual]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.io.memory import iter_points, write_points_streaming, free_memory_gb
from thwaites.qc.mask import apply_bedmachine_mask, resolve_mask_path

COLS = ["lon", "lat", "h_elv", "s_elv", "t_year", "beam",
        "tide_ocean", "tide_equilibrium", "dac", "geoid"]


def main():
    ap = argparse.ArgumentParser(description="Aplica máscara BedMachine (streaming).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--batch-rows", type=int, default=2_000_000)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="mask")

    src = cfg.paths.interim / "atl06_merged.parquet"
    dst = cfg.paths.interim / "atl06_masked.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Consolidado não encontrado: {src} (rode run_consolidate.py).")
    tif = resolve_mask_path(cfg)
    if not tif.exists():
        raise FileNotFoundError(f"BedMachine não encontrado: {tif} (rode fetch_bedmachine.py).")

    import pyarrow.parquet as pq
    names = pq.ParquetFile(src).schema_arrow.names
    cols = [c for c in COLS if c in names]
    log.info(f"Entrada: {src.name} | máscara: {tif.name} | "
             f"lote {args.batch_rows:,} | livre {free_memory_gb():.1f} GB")

    stats = {"in": 0, "out": 0}

    def chunks():
        for df in iter_points(src, cols, batch_rows=args.batch_rows, do_downcast=False):
            stats["in"] += len(df)
            out = apply_bedmachine_mask(df, cfg, tif_path=tif)
            stats["out"] += len(out)
            log.info(f"  {stats['in']:,} lidas -> {stats['out']:,} mantidas "
                     f"(livre {free_memory_gb():.1f} GB)")
            yield out

    path, n = write_points_streaming(chunks(), dst)
    rem = stats["in"] - stats["out"]
    log.info(f"Mascarado -> {path} ({n:,} pontos; {rem:,} removidos = "
             f"{100*rem/max(stats['in'],1):.1f}%, sobretudo oceano)")
    log.info("Próximo: python pipelines/run_corrections.py")


if __name__ == "__main__":
    main()
