"""
pipelines/run_xover.py
======================
Análise de cruzamentos: dh/dt INDEPENDENTE do fitsec (validação cruzada de
método) + viés inter-feixe.

    data/interim/atl06_filtered.parquet
        -> data/interim/xovers.parquet
        -> outputs/tables/xover_summary.json     (comparação com o fitsec)
        -> outputs/tables/interbeam_bias.csv

Rode DEPOIS de run_filttrack.py (que cria o track_id) e de run_dhdt.py
(para a comparação com o pipeline principal).

Uso: python pipelines/run_xover.py [--profile anual]
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
from thwaites.qc.xover import find_crossovers, interbeam_bias


def main():
    ap = argparse.ArgumentParser(description="Cruzamentos (crossovers).")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="xover")

    src = cfg.paths.interim / "atl06_filtered.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} não existe (rode run_filttrack.py, que cria track_id).")

    # só as colunas que o xover usa (evita carregar 17 colunas de 19,7 M linhas)
    from thwaites.io.memory import read_points
    hcol = next((c for c in ("h_res", "h_corr", "h_elv")
                 if c in __import__("pyarrow.parquet", fromlist=["x"])
                 .ParquetFile(src).schema_arrow.names), "h_elv")
    df = read_points(src, ["track_id", "x", "y", "t_year", "beam", hcol])
    log.info(f"Lidos {len(df):,} pontos de {src.name}")

    xo = find_crossovers(df, cfg)
    out = cfg.paths.interim / "xovers.parquet"
    xo.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    log.info(f"Cruzamentos -> {out} ({len(xo):,})")

    cfg.paths.tables.mkdir(parents=True, exist_ok=True)

    # viés inter-feixe (cruzamentos quase-simultâneos)
    bias = interbeam_bias(xo)
    if not bias.empty:
        bias.to_csv(cfg.paths.tables / "interbeam_bias.csv", index=False)
        log.info(f"Viés inter-feixe -> interbeam_bias.csv "
                 f"(|viés| máx {bias['bias_m'].abs().max():.3f} m)")
    else:
        log.warning("Sem cruzamentos quase-simultâneos para estimar viés inter-feixe.")

    # comparação com o fitsec (validação independente)
    summary = {"n_xovers": int(len(xo))}
    v = xo["dhdt"].dropna()
    if len(v):
        summary.update({
            "xover_dhdt_median": float(v.median()),
            "xover_dhdt_mean": float(v.mean()),
            "xover_dhdt_mad": float(1.4826 * np.median(np.abs(v - v.median()))),
            "n_valid": int(len(v)),
        })
        nodes_p = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
        if nodes_p.exists():
            nodes = pd.read_parquet(nodes_p, columns=["dhdt"])
            summary["fitsec_dhdt_median"] = float(nodes["dhdt"].median())
            summary["difference_median"] = summary["xover_dhdt_median"] - summary["fitsec_dhdt_median"]
            log.info(f"VALIDAÇÃO — xover {summary['xover_dhdt_median']:+.3f} vs "
                     f"fitsec {summary['fitsec_dhdt_median']:+.3f} m/ano "
                     f"(diferença {summary['difference_median']:+.3f})")
    (cfg.paths.tables / "xover_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"Resumo -> {cfg.paths.tables / 'xover_summary.json'}")


if __name__ == "__main__":
    main()
