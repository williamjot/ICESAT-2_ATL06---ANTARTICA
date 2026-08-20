"""Processa os sete subsets anuais JJA do GLORYS12V1 para o ASE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites.ocean.glorys import compute_jja_metrics

DEFAULT_INPUT = ROOT / "data" / "ocean" / "regional" / "glorys12v1"
DEFAULT_OUTPUT = ROOT / "outputs" / "mecanismo_oceanico_regional"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mcdw-theta-min", type=float, default=0.0)
    parser.add_argument("--mcdw-salinity-min", type=float, default=34.5)
    args = parser.parse_args()
    paths = sorted(args.input.glob("glorys12v1_ASE_JJA_*.nc"))
    if len(paths) != 7:
        raise FileNotFoundError(
            f"esperados 7 subsets JJA (2019–2025), encontrados {len(paths)} em "
            f"{args.input}. Execute pipelines/fetch_glorys12v1_jja.py.")
    with xr.open_mfdataset(paths, combine="by_coords", parallel=True,
                           chunks={"time": 1}) as source:
        metrics = compute_jja_metrics(
            source.load(), mcdw_theta_min=args.mcdw_theta_min,
            mcdw_salinity_min=args.mcdw_salinity_min)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "glorys12v1_ase_jja_ocean_metrics.nc"
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for name in metrics.data_vars}
    metrics.to_netcdf(path, encoding=encoding)
    report = {
        "status": "METRICAS_OCEANICAS_CONCLUIDAS",
        "source_files": [str(item) for item in paths],
        "output": str(path),
        "years": [int(value) for value in metrics.year.values],
        "variables": list(metrics.data_vars),
        "mcdw_definition": metrics.attrs["mcdw_definition"],
        "caveats": [
            "GLORYS12V1 não resolve pequenas cavidades subplataforma",
            "mCDW depende de limiares candidatos que exigem análise de sensibilidade",
            "conteúdo de calor não equivale a transporte de calor através de uma seção",
        ],
    }
    report_path = args.output / "glorys12v1_ase_jja_ocean_metrics_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"Métricas -> {path}")
    print(f"Relatório -> {report_path}")


if __name__ == "__main__":
    main()

