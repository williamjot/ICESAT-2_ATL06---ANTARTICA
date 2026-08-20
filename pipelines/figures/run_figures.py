"""
pipelines/run_figures.py
========================
Gera as figuras do projeto a partir dos produtos já calculados.

    data/interim/dhdt_grid.parquet + data/dhdt/dhdt_nodes_qc.parquet
    data/timeseries/node_trends.parquet
        -> outputs/figures/map_dhdt.png
        -> outputs/figures/hist_dhdt.png
        -> outputs/figures/map_trend_significance.png

Uso:
    python pipelines/run_figures.py [--profile anual]
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
from thwaites.viz import (fig_dhdt_map, fig_dhdt_hist, fig_trend_significance,
                          fig_xover_validation, fig_uncertainty_map,
                          fig_basal_melt_map, fig_dhdt_vs_velocity,
                          fig_mass_budget)


def _load_json(path: Path, newer_than: Path | None = None, log=None):
    """
    Lê um JSON de resumo tolerando encoding (resumos foram gravados em execuções
    diferentes, alguns em latin-1). Devolve None se não existir.

    TRAVA DE OBSOLESCÊNCIA: se `newer_than` for dado e o JSON for mais antigo
    que esse arquivo, ele é descartado. Isso impede que um resumo de outra ROI
    seja exibido sob um título incompatível.
    """
    if not path.exists():
        return None
    if newer_than is not None and newer_than.exists():
        if path.stat().st_mtime < newer_than.stat().st_mtime:
            msg = (f"{path.name} é mais ANTIGO que {newer_than.name} — "
                   f"resumo obsoleto, IGNORADO "
                   f"(rode a etapa que o gera).")
            if log is not None:
                log.warning(msg)
            else:
                print(f"AVISO: {msg}")
            return None
    for enc in ("utf-8", "latin-1"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def _glaciological(cfg, nodes, log, figs, exp_name: str,
                   mass_exp_name: str | None = None) -> None:
    """
    Diagramas glaciológicos. Cada um depende de um produto auxiliar que pode não
    existir; a ausência é registrada como aviso e a figura é pulada, em vez de
    interromper as demais.
    """
    # Referência de frescor: os nós de dh/dt são o produto raiz desta ROI.
    ref = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"

    flux_nc = cfg.paths.interim / "flux_divergence.nc"
    if flux_nc.exists() and flux_nc.stat().st_mtime < ref.stat().st_mtime:
        log.warning(f"{flux_nc.name} é mais ANTIGO que {ref.name} (ROI incompatível) "
                    f"— mapa de derretimento basal PULADO para não rotular "
                    f"dado de outra região (rode run_flux.py).")
    elif flux_nc.exists():
        out = fig_basal_melt_map(flux_nc, cfg, figs / "map_basal_melt.png")
        log.info(f"figura -> {out}")
    else:
        log.warning("flux_divergence.nc ausente — pulando mapa de derretimento "
                    "basal (rode run_flux.py).")

    vel = cfg.paths.data_dir / cfg.velocity.path
    if vel.exists():
        try:
            out = fig_dhdt_vs_velocity(nodes, vel, cfg, figs / "dhdt_vs_velocity.png")
            log.info(f"figura -> {out}")
        except ValueError as e:
            # tipicamente: recorte de velocidade menor que a ROI atual
            log.warning(f"dh/dt × velocidade pulado: {e}")
    else:
        log.warning(f"{vel.name} ausente — pulando dh/dt × velocidade "
                    "(rode fetch_velocity.py).")

    exp = cfg.paths.outputs_dir / "experiments" / exp_name
    mass_exp = cfg.paths.outputs_dir / "experiments" / (mass_exp_name or exp_name)
    summaries = {
        "mass_balance": (_load_json(mass_exp / "mass_balance.json", ref, log)
                         or _load_json(cfg.paths.tables / "mass_balance_raw_elevation.json", ref, log)),
        "flux": _load_json(cfg.paths.tables / "flux_summary.json", ref, log),
        "firn": (_load_json(exp / "firn_summary.json", ref, log)
                 or _load_json(exp / "mass_balance_firn.json", ref, log)),
    }
    if any(v for v in summaries.values()):
        out = fig_mass_budget(summaries, cfg, figs / "mass_budget.png")
        log.info(f"figura -> {out}")
    else:
        log.warning("nenhum resumo de balanço encontrado — pulando diagrama de "
                    "orçamento (rode run_mass_balance.py).")


def main():
    ap = argparse.ArgumentParser(description="Gera as figuras do projeto.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--experiment", default="ase_v1",
                    help="nome do experimento onde procurar os resumos de firn")
    ap.add_argument("--mass-experiment", default=None,
                    help="experimento do balanço de massa; por padrão usa --experiment")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="figures")
    figs = cfg.paths.figures

    # PRODUTO VALIDADO, não a grade crua de nós.
    #
    # `dhdt_nodes_qc.parquet` é o produto validado após o filtro espacial. Usar a
    # grade crua incluiria nós sobre oceano, plataforma e zonas de buffer, fazendo
    # figuras e tabelas descreverem populações diferentes. A ausência do produto
    # validado deve causar falha explícita, sem fallback silencioso.
    nodes_path = cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet"
    if not nodes_path.exists():
        raise FileNotFoundError(
            f"{nodes_path} ausente — rode pipelines/run_qc_report.py. As figuras "
            f"NÃO devem usar dhdt_nodes.parquet, que é anterior ao filtro de nós "
            f"por posição e inclui nós sobre oceano e plataforma.")
    nodes = pd.read_parquet(nodes_path)
    log.info(f"nós: {len(nodes):,} do produto validado ({nodes_path.name})")

    grid_path = cfg.paths.interim / "dhdt_grid.parquet"
    if grid_path.exists():
        grid = pd.read_parquet(grid_path)
        out = fig_dhdt_map(grid, nodes, cfg, figs / "map_dhdt.png")
        log.info(f"figura -> {out}")
        if "var" in grid.columns:
            out = fig_uncertainty_map(grid, nodes, cfg, figs / "map_uncertainty.png")
            log.info(f"figura -> {out}")
    else:
        log.warning("dhdt_grid.parquet ausente — pulando mapa (rode run_interpolation.py).")

    out = fig_dhdt_hist(nodes, cfg, figs / "hist_dhdt.png")
    log.info(f"figura -> {out}")

    xo_path = cfg.paths.interim / "xovers.parquet"
    if xo_path.exists():
        xo = pd.read_parquet(xo_path)
        if xo["dhdt"].notna().any():
            # estatisticas pareadas (Prioridade 5) para a caixa de aviso:
            # distribuicoes semelhantes nao implicam concordancia ponto a ponto
            ps = None
            summ = (cfg.paths.outputs_dir / "experiments" / "agree_v1"
                    / "agreement_summary.json")
            if summ.exists():
                try:
                    ps = json.loads(summ.read_text(encoding="latin-1"))
                except Exception:
                    ps = None
            out = fig_xover_validation(xo, nodes, cfg,
                                       figs / "xover_validation.png",
                                       paired_stats=ps)
            log.info(f"figura -> {out}")
    else:
        log.info("xovers.parquet ausente — pulando validação por crossover "
                 "(rode run_xover.py).")

    trends_path = cfg.paths.timeseries_dir / "node_trends.parquet"
    if trends_path.exists():
        trends = pd.read_parquet(trends_path)
        out = fig_trend_significance(trends, cfg, figs / "map_trend_significance.png")
        log.info(f"figura -> {out}")
    else:
        log.warning("node_trends.parquet ausente — pulando mapa de tendência.")

    _glaciological(cfg, nodes, log, figs, args.experiment, args.mass_experiment)

    log.info(f"Figuras em {figs}")


if __name__ == "__main__":
    main()
