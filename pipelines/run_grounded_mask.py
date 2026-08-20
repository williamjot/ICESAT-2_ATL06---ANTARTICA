"""
pipelines/run_grounded_mask.py
==============================
Recorte científico para GELO ATERRADO + relatório de controle de qualidade.

    data/interim/atl06_filtered.parquet
        -> data/interim/atl06_grounded.parquet
        -> outputs/tables/grounded_mask_report.json

Critério (ver bloco `grounded` em config/default.yaml):
    mask == 2 (grounded_ice)  E  dist(não-aterrado) >= buffer_coast
                              E  dist(plataforma)   >= buffer_grounding_line

Por que este passo é separado de `run_mask.py`: aquele aplica a máscara LARGA
(tudo menos oceano), necessária para as etapas que precisam de gelo flutuante
(maré CATS, divergência de fluxo). Este aplica o recorte do ALVO CIENTÍFICO.
Fundi-los obrigaria a reprocessar tudo para mudar o escopo da análise.

MEMÓRIA: streaming por row group — a entrada tem ~54 M linhas.

Uso: python pipelines/run_grounded_mask.py [--profile P] [--gl-buffer M] [--coast-buffer M]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.io.memory import iter_points, write_points_streaming, free_memory_gb
from thwaites.qc.grounded_mask import (
    load_bedmachine_roi, distance_fields, sample_fields_at, grounded_keep_mask,
    BM_NAMES, BM_GROUNDED_ICE,
)

DIST_CACHE = "bedmachine_distfields.npz"


def get_distance_fields(cfg, log, rebuild: bool = False):
    """Campos de distância, com cache em disco (o cálculo leva ~1 min)."""
    cache = cfg.paths.interim / DIST_CACHE
    if cache.exists() and not rebuild:
        z = np.load(cache)
        log.info(f"campos de distância do cache: {cache.name}")
        return (z["sx"], z["sy"], z["mask"],
                {k: z[k] for k in ("dist_to_nongrounded", "dist_to_floating",
                                   "dist_to_ocean")})
    sx, sy, m = load_bedmachine_roi(cfg)
    F = distance_fields(m, abs(sx[1] - sx[0]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, sx=sx, sy=sy, mask=m,
                        **{k: v.astype(np.float32) for k, v in F.items()})
    log.info(f"campos de distância calculados e salvos -> {cache.name}")
    return sx, sy, m, F


def main():
    ap = argparse.ArgumentParser(description="Recorte de gelo aterrado + QC.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--input", default="atl06_filtered.parquet")
    ap.add_argument("--output", default="atl06_grounded.parquet")
    ap.add_argument("--gl-buffer", type=float, default=None,
                    help="override do buffer de linha de aterramento (m)")
    ap.add_argument("--coast-buffer", type=float, default=None,
                    help="override do buffer de costa (m)")
    ap.add_argument("--batch-rows", type=int, default=3_000_000)
    ap.add_argument("--rebuild-dist", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="grounded_mask")

    gl_buf = args.gl_buffer if args.gl_buffer is not None else cfg.grounded.buffer_grounding_line_m
    co_buf = args.coast_buffer if args.coast_buffer is not None else cfg.grounded.buffer_coast_m

    src = cfg.paths.interim / args.input
    dst = cfg.paths.interim / args.output
    if not src.exists():
        raise FileNotFoundError(f"{src} não existe (rode run_filttrack.py).")

    sx, sy, bm, F = get_distance_fields(cfg, log, rebuild=args.rebuild_dist)
    log.info(f"buffers: linha de aterramento {gl_buf:.0f} m | costa {co_buf:.0f} m")

    import pyarrow.parquet as pq
    cols = list(pq.ParquetFile(src).schema_arrow.names)

    # contadores do relatório
    stats = {"n_in": 0, "n_out": 0,
             "by_class_in": {}, "by_class_out": {},
             "removed_not_grounded": 0, "removed_coast": 0, "removed_gl": 0}

    def chunks():
        for df in iter_points(src, cols, batch_rows=args.batch_rows, do_downcast=False):
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()
            s = sample_fields_at(x, y, sx, sy, F)
            mc = df["mask_class"].to_numpy()

            is_gr = mc == BM_GROUNDED_ICE
            ok_coast = s["dist_to_nongrounded"] >= co_buf
            ok_gl = s["dist_to_floating"] >= gl_buf
            keep = grounded_keep_mask(mc, s["dist_to_nongrounded"],
                                      s["dist_to_floating"], co_buf, gl_buf)

            # contabiliza o MOTIVO da remoção em ordem de precedência, para que
            # as parcelas somem exatamente o total removido
            stats["removed_not_grounded"] += int((~is_gr).sum())
            stats["removed_coast"] += int((is_gr & ~ok_coast).sum())
            stats["removed_gl"] += int((is_gr & ok_coast & ~ok_gl).sum())

            for k, c in zip(*np.unique(mc, return_counts=True)):
                stats["by_class_in"][int(k)] = stats["by_class_in"].get(int(k), 0) + int(c)
            for k, c in zip(*np.unique(mc[keep], return_counts=True)):
                stats["by_class_out"][int(k)] = stats["by_class_out"].get(int(k), 0) + int(c)

            stats["n_in"] += len(df)
            stats["n_out"] += int(keep.sum())

            out = df.loc[keep].copy()
            # guarda as distâncias: alimentam a classificação de confiabilidade
            # e os mapas de QC sem precisar reamostrar o BedMachine depois
            out["dist_gl"] = s["dist_to_floating"][keep].astype(np.float32)
            out["dist_coast"] = s["dist_to_nongrounded"][keep].astype(np.float32)
            log.info(f"  {stats['n_in']:,} lidas -> {stats['n_out']:,} mantidas "
                     f"({100*stats['n_out']/max(stats['n_in'],1):.1f}%) "
                     f"| livre {free_memory_gb():.1f} GB")
            yield out

    path, n = write_points_streaming(chunks(), dst)

    rem = stats["n_in"] - stats["n_out"]
    report = {
        "input": src.name, "output": dst.name,
        "buffer_grounding_line_m": gl_buf, "buffer_coast_m": co_buf,
        "n_in": stats["n_in"], "n_out": stats["n_out"],
        "n_removed": rem,
        "pct_removed": 100.0 * rem / max(stats["n_in"], 1),
        "removed_by_reason": {
            "not_grounded_ice": stats["removed_not_grounded"],
            "within_coast_buffer": stats["removed_coast"],
            "within_grounding_line_buffer": stats["removed_gl"],
        },
        "by_class_before": {BM_NAMES.get(k, str(k)): v
                            for k, v in sorted(stats["by_class_in"].items())},
        "by_class_after": {BM_NAMES.get(k, str(k)): v
                           for k, v in sorted(stats["by_class_out"].items())},
        "mask_source": "BedMachine Antarctica v4, variável 'mask', 500 m, EPSG:3031",
        "criterion": ("mask==2 (grounded_ice) AND dist(não-aterrado)>=buffer_coast "
                      "AND dist(plataforma)>=buffer_grounding_line"),
    }
    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    rp = cfg.paths.tables / "grounded_mask_report.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"Gelo aterrado -> {path} ({n:,} pontos; {rem:,} removidos = "
             f"{report['pct_removed']:.1f}%)")
    log.info(f"  não-aterrado: {stats['removed_not_grounded']:,} | "
             f"buffer costa: {stats['removed_coast']:,} | "
             f"buffer linha de aterramento: {stats['removed_gl']:,}")
    log.info(f"Relatório -> {rp}")
    log.info("Próximo: run_tiles.py --input atl06_grounded.parquet")


if __name__ == "__main__":
    main()
