"""
pipelines/run_tiles.py
======================
Divide os pontos em tiles espaciais com halo.

    data/interim/atl06_filtered.parquet (ou slopecorr / corrected)
        -> data/tiles/tile_*.parquet + manifest.json

MEMÓRIA: usa `build_tiles_streaming` — uma passagem sobre o Parquet com um
writer por tile, mantendo na RAM apenas o lote corrente. Materializar a tabela
inteira e copiá-la a cada seleção de tile excederia os ~3 GB disponíveis na
máquina alvo.

Uso: python pipelines/run_tiles.py [--profile anual] [--batch-rows N]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.grid.tiles import build_tiles_streaming
from thwaites.io.memory import free_memory_gb


def main():
    ap = argparse.ArgumentParser(description="Cria tiles espaciais com halo (streaming).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--batch-rows", type=int, default=2_000_000,
                    help="linhas por lote (reduza se a memória apertar)")
    ap.add_argument("--input", default=None,
                    help="arquivo em data/interim/ (default: o mais avançado "
                         "disponível, começando por atl06_grounded.parquet)")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="tiles")

    # prioridade: grounded (recorte científico) > filtered (track_id)
    #           > slopecorr (h_res) > corrected
    # `grounded` vem primeiro porque é o recorte do ALVO da análise; usar
    # `filtered` quando ele existe traria de volta plataforma e rocha.
    if args.input:
        src = cfg.paths.interim / args.input
        if not src.exists():
            raise FileNotFoundError(f"Entrada não encontrada: {src}")
    else:
        candidates = [cfg.paths.interim / n for n in
                      ("atl06_grounded.parquet", "atl06_filtered.parquet",
                       "atl06_slopecorr.parquet", "atl06_corrected.parquet")]
        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            raise FileNotFoundError(
                f"Nenhuma entrada em {cfg.paths.interim} "
                f"(rode run_corrections.py / run_slope.py / run_filttrack.py).")

    log.info(f"Entrada: {src.name} | lote {args.batch_rows:,} linhas | "
             f"livre {free_memory_gb():.1f} GB")
    manifest = build_tiles_streaming(src, cfg, batch_rows=args.batch_rows)
    n_core = sum(e["n_core"] for e in manifest)
    n_halo = sum(e["n_with_halo"] for e in manifest)
    log.info(f"{len(manifest)} tiles | núcleo {n_core:,} | com halo {n_halo:,} "
             f"(+{100*(n_halo-n_core)/max(n_core,1):.0f}% de sobreposição)")
    log.info("Próximo: python pipelines/run_dhdt.py")


if __name__ == "__main__":
    main()
