"""
pipelines/run_uncertainty.py
============================
Corrige a incerteza do dh/dt por JACKKNIFE SOBRE ANOS e re-propaga.

Por que: o `dhdt_err` do ajuste (≈0,001 m/ano) trata as observações de um nó
como independentes e é otimista por ~uma ordem de magnitude. Isso foi medido de
dois modos independentes — crossovers (|z|≈11,7) e validação temporal sem
vazamento (z-std≈6,2). A TAXA não muda (viés ~zero em ambas as validações);
só a incerteza é refeita.

    data/tiles/*.parquet + data/dhdt/dhdt_nodes.parquet
        -> data/dhdt/dhdt_nodes.parquet  (colunas de incerteza adicionadas;
                                          `dhdt_err` passa a ser a do jackknife)
        -> outputs/experiments/<nome>/uncertainty_report.json

Uso:
    python pipelines/run_uncertainty.py --name unc_v1
    python pipelines/run_uncertainty.py --name unc_teste --max-tiles 3
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.experiments.manifest import Manifest
from thwaites.grid.tiles import load_manifest
from thwaites.io.memory import free_memory_gb
from thwaites.timeseries.uncertainty import add_jackknife_uncertainty

COLS = ["x", "y", "t_year", "s_elv", "h_res", "h_corr", "h_elv"]


def main():
    ap = argparse.ArgumentParser(description="Incerteza por jackknife sobre anos.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--no-apply", action="store_true",
                    help="calcula e reporta, mas NÃO sobrescreve dhdt_err")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"uncertainty_{args.name}")

    nd_p = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
    if not nd_p.exists():
        raise FileNotFoundError(f"{nd_p} não existe (rode run_dhdt.py).")

    man = Manifest(cfg, args.name,
                   purpose="Correção da incerteza do dh/dt (jackknife sobre anos)",
                   overwrite=args.overwrite, seed=0)
    man.set("motivation", {
        "problem": ("dhdt_err formal trata observações do mesmo ano como "
                    "independentes; a amostra efetiva de uma taxa é o nº de épocas"),
        "evidence_crossovers": "|z| mediano 11.7; 12% dentro de 2 sigma (P5)",
        "evidence_validation": "z_std 6.2; cobertura 68% observada 0.31 (P4)",
        "method": "jackknife leave-one-year-out: var=(k-1)/k*sum((theta_i-mean)^2)",
        "note": "a TAXA não é recalculada — viés ~zero nas duas validações",
    })
    man.add_input(nd_p)

    nodes = pd.read_parquet(nd_p, engine="pyarrow")
    log.info(f"{len(nodes):,} nós | livre {free_memory_gb():.1f} GB")

    entries = load_manifest(cfg)
    if args.max_tiles:
        entries = entries[:args.max_tiles]
    log.info(f"processando {len(entries)} tiles")

    parts = []
    for e in entries:
        tp = cfg.paths.tiles_dir / e["file"]
        if not tp.exists():
            log.warning(f"tile ausente: {tp.name}")
            continue
        import pyarrow.parquet as pq
        names = pq.ParquetFile(tp).schema_arrow.names
        cols = [c for c in COLS if c in names]
        tdf = pd.read_parquet(tp, columns=cols, engine="pyarrow")
        sub = nodes[(nodes["x"] >= e["x_min"]) & (nodes["x"] < e["x_max"]) &
                    (nodes["y"] >= e["y_min"]) & (nodes["y"] < e["y_max"])]
        if sub.empty:
            continue
        parts.append(add_jackknife_uncertainty(tdf, sub, cfg))
        log.info(f"  {e['tile']}: {len(sub):,} nós | livre {free_memory_gb():.1f} GB")
        del tdf

    if not parts:
        log.error("nenhum nó processado.")
        return
    upd = pd.concat(parts, ignore_index=True)

    good = np.isfinite(upd["dhdt_err_jack"]) & np.isfinite(upd["dhdt_err_formal"]) \
        & (upd["dhdt_err_formal"] > 0)
    report = {
        "n_nodes": int(len(upd)),
        "n_with_jackknife": int(good.sum()),
        "err_formal_median": float(np.nanmedian(upd.loc[good, "dhdt_err_formal"])),
        "err_jackknife_median": float(np.nanmedian(upd.loc[good, "dhdt_err_jack"])),
        "inflation_median": float(np.nanmedian(upd.loc[good, "err_inflation"])),
        "inflation_p10": float(np.nanpercentile(upd.loc[good, "err_inflation"], 10)),
        "inflation_p90": float(np.nanpercentile(upd.loc[good, "err_inflation"], 90)),
        "dhdt_median": float(np.nanmedian(upd["dhdt"])),
        "field_std": float(np.nanstd(upd["dhdt"])),
        "n_years_median": float(np.nanmedian(upd["n_years_node"])),
        "applied_to_dhdt_err": (not args.no_apply),
    }

    if not args.no_apply:
        # preserva o original antes de sobrescrever
        bak = nd_p.with_name("dhdt_nodes_formalerr.parquet")
        if not bak.exists():
            shutil.copy2(nd_p, bak)
            log.info(f"backup do original -> {bak.name}")
        upd["dhdt_err"] = np.where(np.isfinite(upd["dhdt_err_jack"]),
                                   upd["dhdt_err_jack"], upd["dhdt_err"])
        upd.to_parquet(nd_p, index=False, engine="pyarrow", compression="snappy")
        man.add_output(nd_p)
        log.info(f"dhdt_err atualizado em {nd_p.name} "
                 f"(original preservado em {bak.name})")

    op = man.path_for("uncertainty_report.json")
    op.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    man.add_output(op)
    man.set("report", report).write()

    log.info(f"FATOR DE INFLAÇÃO mediano: {report['inflation_median']:.1f}× "
             f"(p10 {report['inflation_p10']:.1f}× — p90 {report['inflation_p90']:.1f}×)")
    log.info(f"erro mediano: {report['err_formal_median']:.5f} -> "
             f"{report['err_jackknife_median']:.5f} m/ano "
             f"(campo varia {report['field_std']:.3f} m/ano)")
    if not args.no_apply:
        log.info("Próximo: run_interpolation.py e run_mass_balance.py "
                 "(para re-propagar a incerteza corrigida)")


if __name__ == "__main__":
    main()
