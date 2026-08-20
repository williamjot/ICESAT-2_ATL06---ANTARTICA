"""
pipelines/run_slope.py
======================
Correção de slope por DEM de referência: `h_res = h_corr − REMA`.

    data/interim/atl06_corrected.parquet + data/REMA_*.tif
        -> data/interim/atl06_slopecorr.parquet

MEMÓRIA:
  1. a tabela é processada em streaming por row group, evitando materializar
     20 M × 16 colunas e uma cópia (~1,2 GB de pico);
  2. `sample_rema_bilinear` lê janelas por bloco espacial na resolução nativa
     de 32 m, evitando materializar 1,28–3,1 GB de mosaico REMA.

Uso: python pipelines/run_slope.py [--profile anual]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.io.memory import iter_points, write_points_streaming, free_memory_gb
from thwaites.corrections.slope import apply_slope_reference, resolve_rema_path


def main():
    ap = argparse.ArgumentParser(description="Correção de slope por REMA (streaming).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--batch-rows", type=int, default=2_000_000)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="slope")
    if not cfg.slope.enabled:
        log.warning("slope.enabled=false — nada a fazer.")
        return

    src = cfg.paths.interim / "atl06_corrected.parquet"
    dst = cfg.paths.interim / "atl06_slopecorr.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Corrigido não encontrado: {src} (rode run_corrections.py).")
    rema = resolve_rema_path(cfg)
    if not rema.exists():
        raise FileNotFoundError(f"REMA não encontrado: {rema} (rode fetch_rema.py).")

    import pyarrow.parquet as pq
    cols = list(pq.ParquetFile(src).schema_arrow.names)
    log.info(f"Entrada: {src.name} | REMA: {rema.name} | "
             f"lote {args.batch_rows:,} | livre {free_memory_gb():.1f} GB")

    total = {"n": 0, "ref": 0}

    def chunks():
        for df in iter_points(src, cols, batch_rows=args.batch_rows, do_downcast=False):
            out = apply_slope_reference(df, cfg)
            total["n"] += len(out)
            total["ref"] += int(out["h_res"].notna().sum())
            log.info(f"  {total['n']:,} linhas processadas "
                     f"(livre {free_memory_gb():.1f} GB)")
            yield out

    path, n = write_points_streaming(chunks(), dst)
    frac = 100 * total["ref"] / max(total["n"], 1)
    log.info(f"Slope-referenciado -> {path} ({n:,} pontos, "
             f"{frac:.1f}% com h_res válido)")
    log.info("Próximo: python pipelines/run_filttrack.py")


if __name__ == "__main__":
    main()
