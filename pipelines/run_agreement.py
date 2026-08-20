"""
pipelines/run_agreement.py
==========================
Prioridade 5 (§6): concordância ESPACIAL entre o ajuste local e os crossovers.

A pergunta (§6.1): a concordância das medianas globais esconde divergências
regionais? Uma diferença global pequena é perfeitamente compatível com dois
campos que discordam em setores opostos, com os erros se cancelando.

    data/interim/xovers.parquet + data/dhdt/dhdt_nodes.parquet
        -> outputs/experiments/<nome>/
             agreement_summary.json   (global, normalizado, espacial)
             matched_pairs.parquet    (para o mapa de diferenças, §6.6)
             hotspots.csv             (setores de discordância persistente)
             interbeam_sensitivity.csv
             manifest.json

RESSALVA REGISTRADA NO MANIFESTO (§6.5): crossovers NÃO são validação
independente — usam o mesmo produto altimétrico, máscara e correções.

Uso: python pipelines/run_agreement.py --name agree_v1
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
from thwaites.validation.agreement import assess_agreement, INDEPENDENCE_CAVEAT
from thwaites.qc.xover import interbeam_bias, interbeam_bias_sensitivity


def main():
    ap = argparse.ArgumentParser(description="Concordância local × crossovers (Prioridade 5).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-dist-km", type=float, default=5.0,
                    help="distância máxima crossover↔nó para parear")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"agreement_{args.name}")

    xo_p = cfg.paths.interim / "xovers.parquet"
    nd_p = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
    for p, cmd in ((xo_p, "run_xover.py"), (nd_p, "run_dhdt.py")):
        if not p.exists():
            raise FileNotFoundError(f"{p} não existe (rode {cmd}).")

    man = Manifest(cfg, args.name,
                   purpose="Prioridade 5 — concordância espacial local × crossovers",
                   overwrite=args.overwrite, seed=0)
    man.set("independence_caveat", INDEPENDENCE_CAVEAT)
    man.add_input(xo_p).add_input(nd_p)

    xo = pd.read_parquet(xo_p, engine="pyarrow")
    nodes = pd.read_parquet(nd_p, engine="pyarrow")
    log.info(f"{len(xo):,} crossovers | {len(nodes):,} nós")

    if "dhdt_err" not in xo.columns:
        log.warning("crossovers sem 'dhdt_err' — rode run_xover.py de novo para "
                    "obter a incerteza propagada (§6.3). A normalização por "
                    "incerteza combinada ficará indisponível.")

    res, matched, hotspots = assess_agreement(
        xo, nodes, cfg, max_dist_m=args.max_dist_km * 1000.0)

    if res.get("status") != "ok":
        log.error(f"análise não concluída: {res.get('status')}")
        man.set("result", res).write()
        return

    mp = man.path_for("matched_pairs.parquet")
    matched.to_parquet(mp, index=False, engine="pyarrow", compression="snappy")
    man.add_output(mp)
    if not hotspots.empty:
        hp = man.path_for("hotspots.csv")
        hotspots.to_csv(hp, index=False)
        man.add_output(hp)

    # --- viés inter-feixe com a mudança real removida (§6.3) ----------------
    expected = float(np.nanmedian(nodes["dhdt"]))
    bias = interbeam_bias(xo, max_dt_years=0.25, expected_dhdt=expected)
    sens = interbeam_bias_sensitivity(xo, expected_dhdt=expected)
    if not bias.empty:
        bp = man.path_for("interbeam_bias.csv")
        bias.to_csv(bp, index=False)
        man.add_output(bp)
    if not sens.empty:
        sp = man.path_for("interbeam_sensitivity.csv")
        sens.to_csv(sp, index=False)
        man.add_output(sp)
        log.info("\nSensibilidade do viés inter-feixe à janela:\n" +
                 sens.to_string(index=False))
    man.set("expected_dhdt_removed_for_bias", expected)

    sp_json = man.path_for("agreement_summary.json")
    sp_json.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    man.add_output(sp_json)
    man.set("result", res).write()

    log.info(f"Δ mediana (xover − local): {res['median_diff']:+.4f} m/ano")
    log.info(f"|z| mediano: {res['median_abs_z']:.2f} | "
             f"dentro de 2σ: {100*res['frac_within_2sigma']:.0f}%")
    log.info(f"hotspots: {res['n_hotspots']}/{res['n_cells']} células")
    if res["spatial"].get("spatially_structured"):
        log.warning("DIFERENÇAS COM ESTRUTURA ESPACIAL — a mediana global esconde "
                    "discordância regional (§6.1). Investigar antes do mapa final.")
    log.info(f"Experimento em {man.dir}")


if __name__ == "__main__":
    main()
