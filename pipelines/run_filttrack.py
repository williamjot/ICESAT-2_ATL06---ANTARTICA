"""
pipelines/run_filttrack.py
==========================
Filtragem along-track (blunders de nuvem/neve soprada/margem de cisalhamento).

    data/interim/atl06_slopecorr.parquet  (ou atl06_corrected.parquet)
        -> data/interim/atl06_filtered.parquet   (+ coluna track_id)

ORÇAMENTO DE MEMÓRIA (medido: ~3,3 GB livres em máquina de 8 GB)
----------------------------------------------------------------
Duas passagens, nenhuma delas materializando a tabela inteira:

  1ª passagem: lê SÓ [beam, t_year, elevação] (3 colunas em vez de 16) e calcula
     a máscara de rejeição + track_id. Saída: dois vetores de ~20 M elementos
     (~180 MB), não um DataFrame de 1 GB.
  2ª passagem: relê o arquivo em row groups, aplica a máscara e grava
     incrementalmente. Entrada e saída nunca coexistem na memória.

Ordenar um DataFrame completo com `df.iloc[order]` elevaria o pico a cerca de
2 GB e poderia levar a swap.

Rode DEPOIS de run_slope.py e ANTES de run_tiles.py.
Uso: python pipelines/run_filttrack.py [--profile anual]
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.io.memory import read_points, iter_points, write_points_streaming, free_memory_gb
from thwaites.qc.filttrack import compute_along_track_mask, _height_column


def main():
    ap = argparse.ArgumentParser(description="Filtragem along-track (streaming).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--batch-rows", type=int, default=2_000_000)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="filttrack")

    slopecorr = cfg.paths.interim / "atl06_slopecorr.parquet"
    corrected = cfg.paths.interim / "atl06_corrected.parquet"
    src = slopecorr if slopecorr.exists() else corrected
    if not src.exists():
        raise FileNotFoundError(
            f"Entrada não encontrada em {cfg.paths.interim} "
            f"(rode run_corrections.py / run_slope.py).")
    dst = cfg.paths.interim / "atl06_filtered.parquet"

    if not cfg.filttrack.enabled:
        log.warning("filttrack desabilitado na config — nada a fazer.")
        return

    import pyarrow.parquet as pq
    names = pq.ParquetFile(src).schema_arrow.names
    hcol = next((c for c in ("h_res", "h_corr", "h_elv") if c in names), None)
    if hcol is None:
        raise ValueError(f"{src.name} sem coluna de elevação.")
    log.info(f"Entrada: {src.name} | elevação: {hcol} | livre: {free_memory_gb():.1f} GB")

    # ---- 1ª passagem: só 3 colunas -> máscara + track_id -------------------
    log.info("Passagem 1/2: calculando máscara along-track (3 colunas)...")
    light = read_points(src, ["beam", "t_year", hcol], do_downcast=False)
    bad, tid, stats = compute_along_track_mask(
        light["beam"].to_numpy(), light["t_year"].to_numpy(),
        light[hcol].to_numpy(dtype=np.float64), cfg)
    del light
    gc.collect()
    log.info(f"máscara pronta ({stats['n_bad']:,} rejeitados) | "
             f"livre: {free_memory_gb():.1f} GB")

    # ---- 2ª passagem: aplica em streaming e grava --------------------------
    log.info("Passagem 2/2: aplicando e gravando em row groups...")

    def chunks():
        pos = 0
        for df in iter_points(src, list(names), batch_rows=args.batch_rows,
                             do_downcast=False):
            n = len(df)
            sl = slice(pos, pos + n)
            keep = ~bad[sl]
            df = df.loc[keep].copy()
            df["track_id"] = tid[sl][keep]
            pos += n
            log.info(f"  {pos:,}/{len(bad):,} linhas processadas "
                     f"(livre {free_memory_gb():.1f} GB)")
            yield df

    path, total = write_points_streaming(chunks(), dst)
    log.info(f"Filtrado -> {path} ({total:,} pontos)")
    log.info("Próximo: python pipelines/run_tiles.py")


if __name__ == "__main__":
    main()
