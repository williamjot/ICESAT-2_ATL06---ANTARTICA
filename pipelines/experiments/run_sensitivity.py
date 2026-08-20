"""
pipelines/run_sensitivity.py
============================
Prioridade 2 (§3 do PLANO): sensibilidade dos filtros de qualidade.

Varia UM parâmetro por vez a partir da configuração-base, em sub-regiões
representativas, e mede o efeito em cobertura, resíduo e dh/dt — com
diferenças PAREADAS por nó e IC bootstrap. Os critérios de aceite são
pré-definidos em `sensitivity:` na config e vão ao manifesto ANTES da
comparação (§3.5).

Entrada : data/interim/atl06_slopecorr.parquet (ANTES do filttrack)
Saídas  : outputs/experiments/<nome>/
            manifest.json          (§8 — reprodutibilidade)
            config_snapshot.json
            retention_table.csv    (§3.6 — retenção/rejeição por critério)
            sensitivity_table.csv  (§3.6 — sensibilidade das conclusões)
            nodes_<config>.parquet (nós de cada configuração)

Uso:
    python pipelines/run_sensitivity.py --name sens_v1
    python pipelines/run_sensitivity.py --name sens_rapido --only baseline,radius_10km
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.experiments.manifest import Manifest
from thwaites.experiments.sensitivity import (
    default_param_grid, apply_overrides, select_regions_streaming,
    load_region_points, run_single_config, compare_to_baseline,
    evaluate_acceptance,
)

COLUMNS = ["x", "y", "lon", "lat", "t_year", "beam", "s_elv",
           "h_res", "h_corr", "h_elv", "mask_class"]


def main():
    ap = argparse.ArgumentParser(description="Sensibilidade dos filtros (Prioridade 2).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True, help="nome do experimento (diretório)")
    ap.add_argument("--only", default=None,
                    help="lista separada por vírgula de configurações a rodar")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"sensitivity_{args.name}")

    src = cfg.paths.interim / "atl06_slopecorr.parquet"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} não existe. A sensibilidade parte do arquivo ANTES do "
            f"filttrack (senão seria filtragem dupla). Rode run_slope.py.")

    # ---- manifesto ANTES de qualquer comparação (§3.5 / §8) ----------------
    man = Manifest(cfg, args.name, purpose="Prioridade 2 — sensibilidade dos filtros",
                   overwrite=args.overwrite, seed=cfg.sensitivity.seed)
    man.set("acceptance_criteria_predefined", {
        "max_median_dhdt_shift": cfg.sensitivity.max_median_dhdt_shift,
        "max_residual_increase": cfg.sensitivity.max_residual_increase,
        "min_coverage_gain": cfg.sensitivity.min_coverage_gain,
        "max_xover_dispersion_increase": cfg.sensitivity.max_xover_dispersion_increase,
        "note": "registrados antes de observar qualquer resultado (§3.5)",
    })
    man.set("limitation", (
        "sensibilidade avaliada em sub-regiões representativas, não no domínio "
        "inteiro (custo). Declarar como limitação ao publicar."))

    # inspeciona o schema sem ler os dados
    import pyarrow.parquet as pq
    names = pq.ParquetFile(src).schema_arrow.names
    cols = [c for c in COLUMNS if c in names]
    man.add_input(src, columns=cols)

    # 1) escolhe as sub-regiões lendo só x/y/mask (streaming, leve)
    log.info("Selecionando sub-regiões representativas...")
    regions = select_regions_streaming(src, cfg)
    man.set("regions", regions)

    # 2) carrega SOMENTE os pontos dessas sub-regiões (+halo do maior raio testado)
    max_radius = max([cfg.dhdt.search_radius_m] +
                     [g["overrides"].get("dhdt.search_radius_m", 0)
                      for g in default_param_grid()])
    points = load_region_points(src, regions, cols, halo_m=max_radius)
    log.info(f"{len(points):,} pontos nas sub-regiões, colunas {cols}")

    grid = default_param_grid()
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        grid = [g for g in grid if g["name"] in want]
        if not any(g["name"] == "baseline" for g in grid):
            grid = [{"name": "baseline", "overrides": {}, "stage": "filter"}] + grid
    man.set("configurations", grid)
    log.info(f"{len(grid)} configurações: {[g['name'] for g in grid]}")

    # ---- execução ----------------------------------------------------------
    # MEMÓRIA: guardar apenas o CAMINHO dos nós de cada configuração, não os
    # DataFrames. Eles já vão para disco logo abaixo; manter todos em memória
    # ao mesmo tempo (uma dezena de configurações) somava centenas de MB sobre
    # os pontos das sub-regiões, e era o que derrubava o processo. A comparação
    # pareada relê o baseline e um alvo por vez.
    results, node_paths = [], {}
    for g in grid:
        log.info(f"--- configuração '{g['name']}' {g['overrides']} ---")
        cfg_run = apply_overrides(cfg, g["overrides"])
        nodes, stats = run_single_config(points, cfg_run, regions)
        out_p = man.path_for(f"nodes_{g['name']}.parquet")
        nodes.to_parquet(out_p, index=False, engine="pyarrow", compression="snappy")
        node_paths[g["name"]] = out_p
        del nodes
        man.add_output(out_p)
        stats.update({"config": g["name"], "overrides": json.dumps(g["overrides"])})
        results.append(stats)
        log.info(f"    retido {100*stats['frac_retained']:.2f}% | "
                 f"{stats['n_nodes']:,} nós | "
                 f"dh/dt mediano {stats.get('dhdt_median', float('nan')):+.4f}")

    retention = pd.DataFrame(results)
    rp = man.path_for("retention_table.csv")
    retention.to_csv(rp, index=False)
    man.add_output(rp)

    # ---- comparação pareada contra a base ----------------------------------
    # os pontos das sub-regiões não são mais necessários — liberar antes da
    # fase de comparação, que relê nós do disco
    del points

    base_nodes = pd.read_parquet(node_paths["baseline"], engine="pyarrow")
    base_stats = next(r for r in results if r["config"] == "baseline")
    rows = []
    for g in grid:
        if g["name"] == "baseline":
            continue
        target_nodes = pd.read_parquet(node_paths[g["name"]], engine="pyarrow")
        comp = compare_to_baseline(base_nodes, target_nodes, cfg)
        del target_nodes
        tstats = next(r for r in results if r["config"] == g["name"])
        acc = evaluate_acceptance(comp, base_stats, tstats, cfg)
        rows.append({"config": g["name"], "overrides": json.dumps(g["overrides"]),
                     **{k: v for k, v in comp.items() if not isinstance(v, list)},
                     "dhdt_diff_ci95_lo": comp.get("dhdt_diff_ci95", [None, None])[0],
                     "dhdt_diff_ci95_hi": comp.get("dhdt_diff_ci95", [None, None])[1],
                     "passes": acc["passes"],
                     "would_replace_baseline": acc["would_replace_baseline"]})
        log.info(f"    {g['name']}: Δmediana {comp.get('dhdt_diff_median', float('nan')):+.4f} "
                 f"m/ano | cobertura {100*comp.get('coverage_change', 0):+.1f}% | "
                 f"passa={acc['passes']}")

    sens = pd.DataFrame(rows)
    sp = man.path_for("sensitivity_table.csv")
    sens.to_csv(sp, index=False)
    man.add_output(sp)

    mp = man.write()
    log.info(f"Manifesto -> {mp}")
    log.info(f"Experimento completo em {man.dir}")


if __name__ == "__main__":
    main()
