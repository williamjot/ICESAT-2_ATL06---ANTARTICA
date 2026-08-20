"""
pipelines/run_validation.py
===========================
Prioridade 4 (§5): validação espacial/por trilha/temporal SEM vazamento de
observações.

Diferente da validação por nós (que deixa treino e teste compartilharem as
mesmas observações ATL06 pelos raios de busca sobrepostos), aqui a partição é
no nível da OBSERVAÇÃO e os nós são recalculados em cada fold.

As três estratégias medem capacidades DIFERENTES e são reportadas
separadamente (§5.3) — nunca agregadas numa nota única.

    data/interim/atl06_filtered.parquet
        -> outputs/experiments/<nome>/
             folds_manifest.json    (observações/trilhas/anos de cada fold)
             validation_table.csv   (uma linha por fold × método)
             validation_summary.csv (por método, COM dispersão entre folds)
             manifest.json

Uso:
    python pipelines/run_validation.py --name val_v1
    python pipelines/run_validation.py --name val_teste --strategies temporal
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
from thwaites.experiments.sensitivity import select_regions_streaming, load_region_points
from thwaites.validation.folds import (
    spatial_buffer_folds, track_folds, temporal_folds, default_buffer_m,
)
from thwaites.validation.evaluate import run_validation, summarize_by_method

COLUMNS = ["x", "y", "t_year", "beam", "track_id", "s_elv",
           "h_res", "h_corr", "h_elv", "mask_class"]


def main():
    ap = argparse.ArgumentParser(description="Validação sem vazamento (Prioridade 4).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--strategies", default="spatial,track,temporal")
    ap.add_argument("--methods", default="idw,ordinary_kriging,gaussian_kernel")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--block-km", type=float, default=50.0)
    ap.add_argument("--max-regions", type=int, default=3,
                    help="limita as sub-regiões (memória: cada uma carrega "
                         "núcleo + halo de raio+buffer)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"validation_{args.name}")

    src = cfg.paths.interim / "atl06_filtered.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} não existe (rode run_filttrack.py — "
                                f"a validação por trilha precisa de track_id).")

    strategies = [s.strip() for s in args.strategies.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    man = Manifest(cfg, args.name,
                   purpose="Prioridade 4 — validação sem vazamento de observações",
                   overwrite=args.overwrite, seed=0)
    man.set("strategies", strategies).set("methods", methods)
    man.set("leakage_note", (
        "Partição no nível da OBSERVAÇÃO; nós recalculados em cada fold. "
        "A validação por NÓS compartilha observações entre "
        "treino e teste pelos raios de busca sobrepostos (§5.1)."))
    man.add_input(src, columns=COLUMNS)

    # subconjunto representativo (mesma razão de custo/memória da Prioridade 2)
    log.info("Selecionando sub-regiões...")
    regions = select_regions_streaming(src, cfg)
    if args.max_regions and len(regions) > args.max_regions:
        # as mais densas primeiro: mais observações por fold, menos folds vazios
        regions = sorted(regions, key=lambda r: -r["n_points"])[:args.max_regions]
        log.info(f"limitado a {len(regions)} sub-regiões (memória)")
    man.set("regions", regions)
    import pyarrow.parquet as pq
    names = pq.ParquetFile(src).schema_arrow.names
    cols = [c for c in COLUMNS if c in names]
    buffer_m = default_buffer_m(cfg)
    points = load_region_points(src, regions, cols,
                                halo_m=cfg.dhdt.search_radius_m + buffer_m)
    points = points.reset_index(drop=True)
    log.info(f"{len(points):,} observações nas sub-regiões")

    # --- folds --------------------------------------------------------------
    all_folds = []
    if "spatial" in strategies:
        all_folds += spatial_buffer_folds(points["x"], points["y"],
                                          block_m=args.block_km * 1000,
                                          n_folds=args.n_folds, buffer_m=buffer_m)
    if "track" in strategies:
        if "track_id" not in points.columns:
            log.warning("sem track_id — pulando validação por trilha.")
        else:
            all_folds += track_folds(points["track_id"], n_folds=args.n_folds)
    if "temporal" in strategies:
        all_folds += temporal_folds(points["t_year"])

    man.set("buffer_m", buffer_m)
    man.set("folds", [f.summary() for f in all_folds])
    (man.path_for("folds_manifest.json")).write_text(
        json.dumps([f.summary() for f in all_folds], indent=2, ensure_ascii=False), encoding="utf-8")
    man.add_output(man.path_for("folds_manifest.json"))
    log.info(f"{len(all_folds)} folds no total | buffer {buffer_m/1000:.0f} km")

    # --- avaliação ----------------------------------------------------------
    table = run_validation(points, cfg, all_folds, methods)
    tp = man.path_for("validation_table.csv")
    table.to_csv(tp, index=False)
    man.add_output(tp)

    summary = summarize_by_method(table)
    sp = man.path_for("validation_summary.csv")
    summary.to_csv(sp, index=False)
    man.add_output(sp)

    if not summary.empty:
        log.info("\n" + summary.to_string(index=False))
        # §5.5: a seleção deve ser estável entre estratégias; conflitos são
        # reportados, não escondidos numa nota agregada.
        best = (summary.sort_values("rmse_mean").groupby("strategy")
                .first()["method"].to_dict())
        man.set("best_method_by_strategy", best)
        stable = len(set(best.values())) == 1
        man.set("method_selection_stable_across_strategies", stable)
        log.info(f"Melhor método por estratégia: {best}")
        if not stable:
            log.warning("A escolha do método NÃO é estável entre estratégias — "
                        "isso precisa ser discutido, não agregado (§5.5).")

    man.write()
    log.info(f"Experimento em {man.dir}")


if __name__ == "__main__":
    main()
