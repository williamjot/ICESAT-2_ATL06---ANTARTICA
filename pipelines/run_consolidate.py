"""
pipelines/run_consolidate.py
============================
Consolida os Parquets leves por grânulo (`data/processed`) em
`data/interim/atl06_merged.parquet`, sem reler arquivos HDF5 brutos.

Uso:
    python pipelines/run_consolidate.py [--profile anual]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.io.store import consolidate_parquets


def main():
    ap = argparse.ArgumentParser(description="Consolida Parquets por grânulo.")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"consolidate_{cfg.season.name}")

    out = cfg.paths.interim / "atl06_merged.parquet"
    roi = cfg.roi.bounding_box if cfg.roi else None
    excl_log = cfg.paths.logs / f"recorte_roi_{cfg.season.name}.md" if roi else None
    if roi:
        log.info(f"Recorte ROI ativo: lon {cfg.roi.lon_min}..{cfg.roi.lon_max}, "
                 f"lat {cfg.roi.lat_min}..{cfg.roi.lat_max}")

    qc_dir = cfg.paths.qc_flags
    qc_stats: dict = {}
    if cfg.atl06_qc.enabled:
        if not qc_dir.exists():
            raise FileNotFoundError(
                f"atl06_qc.enabled=true mas {qc_dir} não existe "
                f"(rode pipelines/fetch_qc_flags.py).")
        log.info(f"Filtro de qualidade ATL06 ATIVO — flags em {qc_dir.name}")
    else:
        log.info("Filtro de qualidade ATL06 desligado (atl06_qc.enabled=false)")

    path, n = consolidate_parquets(cfg.paths.processed, out, roi=roi,
                                   exclusion_log=excl_log,
                                   qc_dir=qc_dir, cfg=cfg, qc_stats=qc_stats)
    log.info(f"Consolidado {n:,} pontos -> {path}")

    if qc_stats.get("n_in"):
        rem = qc_stats["n_in"] - qc_stats["n_out"]
        log.info(f"QC de qualidade: {qc_stats['n_in']:,} -> {qc_stats['n_out']:,} "
                 f"({100*qc_stats['n_out']/qc_stats['n_in']:.1f}% mantidos; "
                 f"{rem:,} removidos)")
        for k, v in sorted(qc_stats["by_reason"].items(), key=lambda x: -x[1]):
            log.info(f"   reprovados por {k}: {v:,} "
                     f"({100*v/qc_stats['n_in']:.2f}%)")
        if qc_stats["granules_missing_flags"]:
            log.warning(f"{len(qc_stats['granules_missing_flags'])} grânulos SEM "
                        f"flags entraram sem filtro (amostra mista — ver relatório)")
        if qc_stats["granules_row_mismatch"]:
            log.warning(f"{len(qc_stats['granules_row_mismatch'])} grânulos com "
                        f"contagem divergente entraram sem filtro")
        import json
        rp = cfg.paths.tables / "consolidate_qc_report.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(qc_stats, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        log.info(f"Relatório do filtro -> {rp}")

    if excl_log:
        log.info(f"Log de arquivos desconsiderados -> {excl_log}")


if __name__ == "__main__":
    main()
