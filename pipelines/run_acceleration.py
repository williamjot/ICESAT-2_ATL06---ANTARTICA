"""
pipelines/run_acceleration.py
=============================
Prioridade 3 (§4), recorte de ACELERAÇÃO: há evidência estatística de
aceleração do adelgaçamento?

A aceleração NÃO é estimada em todo nó — cada nó passa por seis critérios
independentes (§4.4) e só é reportada onde todos são atendidos. O motivo da
rejeição fica registrado nó a nó, para auditoria.

    data/tiles/*.parquet -> outputs/experiments/<nome>/
                              accel_nodes.parquet
                              accel_summary.json
                              manifest.json

Uso: python pipelines/run_acceleration.py --name accel_v1
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
from thwaites.grid.tiles import load_manifest
from thwaites.experiments.manifest import Manifest
from thwaites.timeseries.acceleration import acceleration_field, AccelCriteria
from thwaites.io.memory import free_memory_gb


def main():
    ap = argparse.ArgumentParser(description="Aceleração com seleção de modelo (Prioridade 3).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--restart", action="store_true",
                    help="descarta o checkpoint e refaz todos os tiles")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="limita o nº de tiles (teste rápido)")
    ap.add_argument("--boot-iters", type=int, default=200)
    ap.add_argument("--qc-nodes", default="dhdt_nodes_qc.parquet",
                    help="produto de nós validados para recortar a saída")
    ap.add_argument("--no-filter-nodes", action="store_true",
                    help="aceita nós fora do produto de gelo aterrado (audit.)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"acceleration_{args.name}")

    crit = AccelCriteria(boot_iters=args.boot_iters)
    man = Manifest(cfg, args.name, purpose="Prioridade 3 — aceleração (§4.4)",
                   overwrite=args.overwrite, seed=0)
    man.set("acceleration_criteria", {
        "min_years": crit.min_years, "alpha": crit.alpha,
        "delta_aicc": crit.delta_aicc, "require_oos_gain": crit.require_oos_gain,
        "boot_iters": crit.boot_iters,
        "min_boot_sign_frac": crit.min_boot_sign_frac,
        "max_loyo_shift_frac": crit.max_loyo_shift_frac,
        "max_resid_ac1": crit.max_resid_ac1,
        "note": ("IC via bootstrap em BLOCOS DE ANO — o erro formal e o AICc "
                 "tratam as observações como independentes e superestimam a "
                 "significância de um padrão temporal cuja amostra efetiva é o "
                 "nº de anos."),
    })
    man.set("scope_note", (
        "Sazonalidade NÃO modelada: a base é só JJA e o ciclo anual é "
        "não-identificável nessa amostragem (§4.2). Consequência: a hipótese "
        "'JJA é conservador' permanece PREMISSA NÃO TESTADA, não resultado."))

    entries = load_manifest(cfg)
    if args.max_tiles:
        entries = entries[:args.max_tiles]
    log.info(f"Avaliando aceleração em {len(entries)} tiles "
             f"(bootstrap {crit.boot_iters} iterações por nó)")

    # MEMÓRIA: ler somente as colunas usadas por `acceleration_field`. Um tile
    # inteiro com 20 colunas e 2,16 M linhas ocupa ~350 MB em pandas, além das
    # cópias internas, e pode exceder os ~3 GB disponíveis.
    import pyarrow.parquet as pq
    NEEDED = ["x", "y", "lon", "lat", "t_year", "h_res", "h_corr", "h_elv", "s_elv"]

    # CHECKPOINT POR TILE — esta etapa leva mais de 12 h (o bootstrap refaz o
    # ajuste `boot_iters` vezes por nó). Cada tile é gravado assim que termina
    # O checkpoint fica FORA do diretório do experimento, de propósito.
    #
    # `experiment_dir(..., overwrite=True)` faz `shutil.rmtree` na árvore
    # inteira — comportamento correto para saídas, que não podem sobreviver de
    # uma execução para outra fingindo ser novas. Mas o checkpoint não é saída:
    # é cálculo caro reaproveitável. Com ele dentro de `man.dir`, um
    # `--overwrite` (pedido, por exemplo, só para recalcular o resumo)
    # apagaria também as horas de bootstrap. Somente `--restart` limpa o
    # checkpoint: `--overwrite` descarta saídas e `--restart` descarta cálculo.
    ckpt = cfg.paths.interim / f"accel_ckpt_{args.name}"
    ckpt.mkdir(parents=True, exist_ok=True)

    done = {p.stem for p in ckpt.glob("*.parquet")}
    if done:
        log.info(f"checkpoint: {len(done)} tiles já processados serão pulados "
                 f"(use --restart para refazer todos)")
    if args.restart:
        for p in ckpt.glob("*.parquet"):
            p.unlink()
        done = set()
        log.info("--restart: checkpoint limpo")

    todo = [e for e in entries if e["tile"] not in done]
    log.info(f"a processar: {len(todo)} de {len(entries)} tiles")

    for i, e in enumerate(todo, 1):
        p = cfg.paths.tiles_dir / e["file"]
        avail = pq.ParquetFile(p).schema_arrow.names
        cols = [c for c in NEEDED if c in avail]
        tdf = pd.read_parquet(p, columns=cols, engine="pyarrow")
        nd = acceleration_field(tdf, cfg, e["x_min"], e["x_max"],
                                e["y_min"], e["y_max"], criteria=crit)
        del tdf                      # devolve o tile antes do próximo
        if len(nd):
            nd["tile"] = e["tile"]
        # grava SEMPRE, mesmo vazio: a presença do arquivo é o que marca o tile
        # como concluído, e um tile legitimamente sem nós não deve ser refeito
        nd.to_parquet(ckpt / f"{e['tile']}.parquet", index=False)
        log.info(f"  [{i}/{len(todo)}] {e['tile']}: {len(nd):,} nós "
                 f"| livre {free_memory_gb():.1f} GB")

    parts = []
    for f in sorted(ckpt.glob("*.parquet")):
        try:
            g = pd.read_parquet(f)
            if len(g):
                parts.append(g)
        except Exception as ex:
            log.warning(f"checkpoint ilegível {f.name}: {type(ex).__name__}")

    if not parts:
        log.warning("Nenhum nó avaliado.")
        return
    log.info(f"consolidando {len(parts)} tiles com nós "
             f"(de {len(list(ckpt.glob('*.parquet')))} processados)")
    nodes = pd.concat(parts, ignore_index=True)

    # RECORTE AO PRODUTO VALIDADO DE GELO ATERRADO.
    #
    # `acceleration_field` monta a sua própria grade de nós a partir dos tiles
    # BRUTOS e não aplica o filtro de gelo aterrado do run_qc_report (classe do
    # BedMachine, buffers de linha de aterramento e de costa, fração aterrada
    # mínima). Sem este recorte, 1.478 dos 8.709 nós (17%) ficavam fora do
    # produto validado — e, pior, 18 dos 49 nós com aceleração SUSTENTADA
    # (37%) estavam entre eles. É a mesma classe de erro que já havia posto 771
    # nós sobre o oceano: nó em grade regular não é nó cientificamente válido.
    qc_path = cfg.paths.dhdt_dir / args.qc_nodes
    if args.no_filter_nodes:
        log.warning("filtro de nós DESLIGADO — a amostra inclui nós fora do "
                    "produto de gelo aterrado.")
    elif not qc_path.exists():
        raise FileNotFoundError(
            f"{qc_path} não existe (rode run_qc_report.py). Use "
            f"--no-filter-nodes para aceitar explicitamente nós não validados.")
    else:
        qc = pd.read_parquet(qc_path, columns=["x", "y"])
        key = set(zip(np.rint(qc["x"]).astype(np.int64),
                      np.rint(qc["y"]).astype(np.int64)))
        keep = np.fromiter(
            ((x, y) in key for x, y in zip(np.rint(nodes["x"]).astype(np.int64),
                                           np.rint(nodes["y"]).astype(np.int64))),
            bool, len(nodes))
        n0 = len(nodes)
        nodes = nodes[keep].copy()
        log.info(f"recorte ao gelo aterrado validado ({qc_path.name}): "
                 f"{n0:,} -> {len(nodes):,} nós "
                 f"({100*len(nodes)/max(n0,1):.1f}% mantidos)")
        man.set("node_filter", {
            "fonte": str(qc_path),
            "n_antes": int(n0), "n_depois": int(len(nodes)),
            "motivo": ("acceleration_field gera nós em grade regular a partir "
                       "dos tiles brutos, sem o filtro de gelo aterrado do "
                       "run_qc_report"),
        })
    if not len(nodes):
        log.warning("Nenhum nó sobrou após o recorte.")
        return
    out_p = man.path_for("accel_nodes.parquet")
    nodes.to_parquet(out_p, index=False, engine="pyarrow", compression="snappy")
    man.add_output(out_p)

    sup = nodes["accel_supported"].astype(bool)
    summary = {
        "n_nodes": int(len(nodes)),
        "n_accel_supported": int(sup.sum()),
        "frac_supported": float(sup.mean()),
        "accel_median_supported": (float(nodes.loc[sup, "accel"].median())
                                   if sup.any() else None),
        "dhdt_median": float(nodes["dhdt"].median()),
        "top_rejection_reasons": (nodes.loc[~sup, "reason"]
                                  .value_counts().head(6).to_dict()),
    }
    (man.path_for("accel_summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    man.add_output(man.path_for("accel_summary.json"))
    man.write()

    log.info(f"Aceleração sustentada em {summary['n_accel_supported']:,}/"
             f"{summary['n_nodes']:,} nós ({100*summary['frac_supported']:.1f}%)")
    log.info(f"Experimento em {man.dir}")


if __name__ == "__main__":
    main()
