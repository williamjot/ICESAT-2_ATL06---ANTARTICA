"""
pipelines/run_corrections.py
============================
Aplica as correções geofísicas (maré oceânica + DAC) produzindo `h_corr`.

    data/interim/atl06_masked[_cats].parquet
        -> data/interim/atl06_corrected.parquet

MEMÓRIA: streaming por row group. Materializar 20 M linhas × 12 colunas e a
cópia de `apply_corrections` excederia 2,4 GB numa ROI 2,4× maior. Entrada e
saída nunca coexistem integralmente na memória.

Uso: python pipelines/run_corrections.py [--profile anual] [--input X.parquet]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.io.memory import iter_points, write_points_streaming, free_memory_gb
from thwaites.corrections import apply_corrections


def main():
    ap = argparse.ArgumentParser(description="Aplica correções geofísicas (streaming).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--input", default=None,
                    help="arquivo em data/interim/ (default: usa o _cats se existir)")
    ap.add_argument("--batch-rows", type=int, default=2_000_000)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="corrections")

    masked = cfg.paths.interim / "atl06_masked.parquet"
    cats = cfg.paths.interim / "atl06_masked_cats.parquet"

    if args.input:
        src = cfg.paths.interim / args.input
    elif cats.exists():
        # TRAVA DE OBSOLESCÊNCIA: escolher a entrada só por "o arquivo existe" é
        # perigoso — um _cats gerado a partir de uma máscara ANTIGA seria usado
        # silenciosamente no lugar da máscara recém-calculada, produzindo um
        # resultado incompatível sem erro de execução.
        if masked.exists() and cats.stat().st_mtime < masked.stat().st_mtime:
            log.warning(
                f"{cats.name} é MAIS ANTIGO que {masked.name} "
                f"(gerado a partir de uma máscara mais antiga) — IGNORADO. "
                f"Rode run_cats_tide.py de novo, ou force com --input.")
            src = masked
        else:
            src = cats
            log.info(f"usando {cats.name} (maré CATS2008)")
    else:
        src = masked
    dst = cfg.paths.interim / "atl06_corrected.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Mascarado não encontrado: {src} (rode run_mask.py).")

    import pyarrow.parquet as pq
    cols = list(pq.ParquetFile(src).schema_arrow.names)
    log.info(f"Entrada: {src.name} | lote {args.batch_rows:,} linhas | "
             f"livre {free_memory_gb():.1f} GB")

    total = {"n": 0}

    def chunks():
        for df in iter_points(src, cols, batch_rows=args.batch_rows, do_downcast=False):
            out = apply_corrections(df, cfg)
            total["n"] += len(out)
            log.info(f"  {total['n']:,} linhas corrigidas "
                     f"(livre {free_memory_gb():.1f} GB)")
            yield out

    path, n = write_points_streaming(chunks(), dst)
    log.info(f"Corrigido -> {path} ({n:,} pontos, coluna h_corr adicionada)")
    log.info("Próximo: python pipelines/run_slope.py")


if __name__ == "__main__":
    main()
