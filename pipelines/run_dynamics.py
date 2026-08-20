"""
pipelines/run_dynamics.py
=========================
Prioridade 7 (§7): integração com a velocidade do gelo.

Responde: regiões com dh/dt ≈ 0 estão REALMENTE estáveis, ou há dinâmica
mascarada? A classe `sem_tendencia_mas_fluxo_rapido` é o possível precursor
de mudança dinâmica destacado no §7.4.

    data/dhdt/dhdt_nodes.parquet + data/velocity_thwaites.nc
        -> outputs/experiments/<nome>/
             nodes_dynamics.parquet   (nós + velocidade + classe conjunta)
             dynamics_summary.json
             manifest.json

TRAVAS (registradas no manifesto):
  - aceleração de fluxo NÃO é derivada de mosaico único (§7.2);
  - "estável" exige compatibilidade com zero DENTRO da incerteza (§7.5);
  - sem cobertura de velocidade -> "inconclusivo", nunca "estável";
  - correlações reportadas com significância corrigida por autocorrelação (§7.4).

Uso: python pipelines/run_dynamics.py --name dyn_v1
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
from thwaites.validate.velocity import (
    aggregate_velocity_to_nodes, distance_to_grounding_line,
    joint_classification, summarize_dynamics, ACCELERATION_BLOCKED_MSG,
)


def main():
    ap = argparse.ArgumentParser(description="Dinâmica: dh/dt × velocidade (Prioridade 7).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fast-speed", type=float, default=100.0,
                    help="limiar de fluxo rápido (m/ano)")
    ap.add_argument("--no-grounding-line", action="store_true",
                    help="pula a distância à linha de aterramento")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"dynamics_{args.name}")

    nd_p = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
    vel_p = cfg.paths.data_dir / cfg.velocity.path
    if not nd_p.exists():
        raise FileNotFoundError(f"{nd_p} não existe (rode run_dhdt.py).")
    if not vel_p.exists():
        raise FileNotFoundError(
            f"{vel_p} não existe (rode pipelines/fetch_velocity.py).")

    man = Manifest(cfg, args.name,
                   purpose="Prioridade 7 — integração com velocidade do gelo",
                   overwrite=args.overwrite, seed=0)
    man.set("velocity_product", {
        "short_name": cfg.velocity.short_name, "version": cfg.velocity.version,
        "epoch_note": cfg.velocity.epoch_note,
        "vx_var": cfg.velocity.vx_var, "vy_var": cfg.velocity.vy_var,
    })
    man.set("acceleration_status", ACCELERATION_BLOCKED_MSG)
    man.set("classification_rules", {
        "fast_speed_m_yr": args.fast_speed,
        "stable": "|dh/dt| <= 1.96·σ (compatível com zero DENTRO da incerteza)",
        "thinning_significant": "dh/dt + 1.96·σ < 0",
        "no_velocity_or_no_sigma": "inconclusivo (nunca 'estável' por omissão)",
    })
    man.add_input(nd_p).add_input(vel_p)

    nodes = pd.read_parquet(nd_p, engine="pyarrow")
    log.info(f"{len(nodes):,} nós de dh/dt")

    # --- velocidade agregada (não interpolada — evita resolução falsa) ------
    log.info("Agregando velocidade na vizinhança dos nós...")
    vel = aggregate_velocity_to_nodes(nodes["x"].to_numpy(), nodes["y"].to_numpy(), cfg)
    for c in ("speed", "speed_mad", "vx", "vy", "n_pixels"):
        nodes[c] = vel[c].to_numpy()

    # --- distância à linha de aterramento (§7.3) ---------------------------
    if not args.no_grounding_line:
        try:
            log.info("Calculando distância à linha de aterramento...")
            nodes["dist_gl_m"] = distance_to_grounding_line(
                nodes["x"].to_numpy(), nodes["y"].to_numpy(), cfg)
        except Exception as e:
            log.warning(f"distância à linha de aterramento indisponível: {e}")

    # --- classificação conjunta e resumo ------------------------------------
    nodes = joint_classification(nodes, cfg, fast_speed_m_yr=args.fast_speed)

    # Comprimento de correlação para o n_eff dos testes de associação.
    #
    # Usa o comprimento de correlação do ERRO (não o alcance do variograma do
    # campo de dh/dt, que mede a estrutura do sinal e seria grande demais).
    L = cfg.mass_balance.correlation_length_m
    if L:
        log.info(f"comprimento de correlação (do ERRO, config): {L:.0f} m")
    else:
        sel = cfg.paths.tables / "interp_selection.json"
        if sel.exists():
            try:
                L = json.loads(sel.read_text())["variogram"]["range_m"]
                log.warning(
                    f"mass_balance.correlation_length_m não definido — caindo no "
                    f"alcance do variograma do SINAL ({L:.0f} m). Isso mede a "
                    f"estrutura glaciológica, não o erro, e tende a subestimar "
                    f"n_eff. Defina o valor na config.")
            except Exception:
                pass
    res = summarize_dynamics(nodes, cfg, correlation_length_m=L)

    out_p = man.path_for("nodes_dynamics.parquet")
    nodes.to_parquet(out_p, index=False, engine="pyarrow", compression="snappy")
    man.add_output(out_p)
    sj = man.path_for("dynamics_summary.json")
    sj.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    man.add_output(sj)
    man.set("result", res).write()

    log.info(f"Classes: {res['class_counts']}")
    for k in ("assoc_dhdt_speed", "assoc_dhdt_dist_gl"):
        if k in res and "spearman_r" in res[k]:
            a = res[k]
            log.info(f"{k}: r={a['spearman_r']:+.3f} | n={a['n']:,} -> "
                     f"n_eff={a['n_effective']:.0f} | p_ingênuo={a['p_naive']:.2e} "
                     f"-> p_corrigido={a['p_autocorr_corrected']:.3f} | "
                     f"significativo (corrigido): {a['significant_corrected']}")
    log.info(f"Experimento em {man.dir}")


if __name__ == "__main__":
    main()
