"""
Fase 6 — converte uma grade explícita de taxa de elevação em balanço de massa.

Por segurança científica, produtos bruto e corrigido por firn nunca compartilham
o mesmo nome de saída. O modo versionado é obrigatório para resultados usados em
figuras ou texto científico:

    python pipelines/run_mass_balance.py --name ase_jja_raw
    python pipelines/run_mass_balance.py --grid outputs/experiments/firn/firn_corrected_grid.parquet \
        --value-col dhdt_ice --name ase_jja_firn
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.experiments.manifest import Manifest
from thwaites.logging import setup_logging
from thwaites.uncertainty.mass_balance import apply_coverage_mask, compute_mass_balance


def _resolve_grid(value: str, cfg) -> Path:
    """Resolve nome em data/interim ou caminho relativo à raiz do projeto."""
    p = Path(value)
    if p.is_absolute():
        return p
    if len(p.parts) == 1:
        return cfg.paths.interim / p
    return cfg.paths.base_dir / p


def main():
    ap = argparse.ArgumentParser(description="Balanço de massa com produto explicitamente identificado.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--nodes", default="dhdt_nodes_qc.parquet")
    ap.add_argument("--grid", default="dhdt_grid.parquet",
                    help="nome em data/interim ou caminho relativo/absoluto")
    ap.add_argument("--value-col", default="pred",
                    help="coluna de taxa: 'pred' (bruta) ou 'dhdt_ice' (FAC corrigida)")
    ap.add_argument("--name", default=None,
                    help="nome do experimento versionado; recomendado para resultado científico")
    ap.add_argument("--output", default=None, help="JSON de saída explícito")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="mass_balance")
    grid_path = _resolve_grid(args.grid, cfg)
    nodes_path = cfg.paths.dhdt_dir / args.nodes
    if not nodes_path.exists():
        nodes_path = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
    if not grid_path.exists() or not nodes_path.exists():
        raise FileNotFoundError("Grade ou nós ausentes; rode interpolação/dhdt antes.")

    grid = pd.read_parquet(grid_path, engine="pyarrow")
    nodes = pd.read_parquet(nodes_path, engine="pyarrow")
    if args.value_col not in grid.columns:
        raise ValueError(f"coluna '{args.value_col}' ausente em {grid_path.name}: {list(grid.columns)}")
    log.info(f"grade: {grid_path.name} ({len(grid):,}) | nós: {nodes_path.name} ({len(nodes):,})")

    L = cfg.mass_balance.correlation_length_m
    if L is None:
        sel_path = cfg.paths.tables / "interp_selection.json"
        if not sel_path.exists():
            raise ValueError("Defina mass_balance.correlation_length_m ou rode run_interpolation.py.")
        L = json.loads(sel_path.read_text(encoding="utf-8"))["variogram"]["range_m"]
    covered = apply_coverage_mask(grid, nodes, cfg.mass_balance.coverage_dist_m)
    result = compute_mass_balance(covered, cfg, correlation_length_m=L,
                                  value_col=args.value_col)
    result.update({
        "input_grid": str(grid_path),
        "value_column": args.value_col,
        "product_status": ("firn_corrected" if args.value_col == "dhdt_ice"
                           else "raw_elevation_not_firn_corrected"),
    })

    manifest = None
    if args.name:
        manifest = Manifest(cfg, args.name,
                            purpose=f"Balanço de massa ({result['product_status']})",
                            overwrite=args.overwrite, seed=0)
        manifest.add_input(grid_path, columns=[args.value_col, "x", "y", "var"])
        manifest.add_input(nodes_path, columns=["x", "y"])
        manifest.set("mass_balance", result)
        out = Path(args.output) if args.output else manifest.path_for("mass_balance.json")
    else:
        out = (Path(args.output) if args.output else
               cfg.paths.tables / "mass_balance_raw_elevation.json")
        log.warning("Sem --name: saída legada será marcada como bruta e não corrigida por firn.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if manifest is not None:
        manifest.add_output(out)
        manifest.write()
    log.info(f"Balanço de massa -> {out}")


if __name__ == "__main__":
    main()
