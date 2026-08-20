"""Baixa somente os subsets JJA do GLORYS12V1 para a ROI do ASE.

O fluxo é reiniciável e evita baixar meses fora de JJA: um NetCDF comprimido
por ano, contendo junho, julho e agosto. Credenciais nunca são recebidas por
argumento deste script nem escritas nos logs; use ``copernicusmarine login``.

Produto confirmado pelo usuário em 2026-08-07:
``cmems_mod_glo_phy_my_0.083deg_P1M-m``; thetao, so, uo, vo; 0–1000 m;
115°W–95°W, 77,5°S–73°S; 2019–2025; JJA.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "ocean" / "regional" / "glorys12v1"
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1M-m"
VARIABLES = ("thetao", "so", "uo", "vo")
YEARS = tuple(range(2019, 2026))
BOUNDS = {
    "minimum_longitude": -115.0,
    "maximum_longitude": -95.0,
    "minimum_latitude": -77.5,
    "maximum_latitude": -73.0,
    "minimum_depth": 0.0,
    "maximum_depth": 1000.0,
}


def credential_hint() -> bool:
    """Detecta somente a presença de credenciais, sem ler ou exibi-las."""
    if (os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") and
            os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")):
        return True
    base = Path.home() / ".copernicusmarine"
    candidates = (
        base / ".copernicusmarine-credentials",
        base / ".netrc",
        base / "_netrc",
        base / "motuclient-python.ini",
    )
    return any(path.exists() and path.stat().st_size > 0 for path in candidates)


def yearly_request(year: int, output: Path, *, dry_run: bool = False) -> list[str]:
    filename = f"glorys12v1_ASE_JJA_{year}.nc"
    command = [
        "copernicusmarine", "subset",
        "--dataset-id", DATASET_ID,
        "--start-datetime", f"{year}-06-01T00:00:00",
        "--end-datetime", f"{year}-08-31T23:59:59",
        "--minimum-longitude", str(BOUNDS["minimum_longitude"]),
        "--maximum-longitude", str(BOUNDS["maximum_longitude"]),
        "--minimum-latitude", str(BOUNDS["minimum_latitude"]),
        "--maximum-latitude", str(BOUNDS["maximum_latitude"]),
        "--minimum-depth", str(BOUNDS["minimum_depth"]),
        "--maximum-depth", str(BOUNDS["maximum_depth"]),
        "--coordinates-selection-method", "outside",
        "--output-directory", str(output),
        "--output-filename", filename,
        "--file-format", "netcdf",
        "--netcdf-compression-level", "4",
        "--skip-existing",
        "--disable-progress-bar",
    ]
    for variable in VARIABLES:
        command.extend(("--variable", variable))
    if dry_run:
        command.extend(("--dry-run", "--response-fields", "all"))
    return command


def validate_file(path: Path, year: int) -> dict:
    with xr.open_dataset(path) as ds:
        missing = set(VARIABLES).difference(ds.data_vars)
        if missing:
            raise ValueError(f"{path.name}: variáveis ausentes: {sorted(missing)}")
        times = ds.time.dt
        observed_years = set(int(v) for v in times.year.values)
        observed_months = set(int(v) for v in times.month.values)
        if observed_years != {year} or observed_months != {6, 7, 8}:
            raise ValueError(
                f"{path.name}: tempo inesperado, anos={observed_years}, "
                f"meses={observed_months}")
        return {
            "file": path.name,
            "bytes": path.stat().st_size,
            "dimensions": {name: int(size) for name, size in ds.sizes.items()},
            "variables": list(VARIABLES),
            "time_start": str(ds.time.values.min()),
            "time_end": str(ds.time.values.max()),
        }


def write_manifest(output: Path, years: list[int], status: str,
                   files: list[dict] | None = None) -> Path:
    payload = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "product_id": "GLOBAL_MULTIYEAR_PHY_001_030",
            "dataset_id": DATASET_ID,
            "doi": "10.48670/moi-00021",
            "variables": list(VARIABLES),
            "temporal_resolution": "monthly",
        },
        "subset": {**BOUNDS, "season": "JJA", "years": years},
        "strategy": "um arquivo comprimido por JJA; execução reiniciável",
        "files": files or [],
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    years = sorted(set(args.years))
    invalid = [year for year in years if year not in YEARS]
    if invalid:
        raise ValueError(f"anos fora do escopo confirmado: {invalid}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plan_only:
        path = write_manifest(args.output, years, "PLANEJADO_AGUARDANDO_DOWNLOAD")
        print(f"Plano -> {path}")
        return 0
    if not credential_hint():
        write_manifest(args.output, years, "BLOQUEADO_CREDENCIAIS_COPERNICUS")
        print(
            "Credenciais Copernicus Marine não configuradas. Execute uma vez, "
            "fora deste script:\n  copernicusmarine login\n"
            "A senha é digitada de forma oculta e não deve ser enviada no chat.",
            file=sys.stderr)
        return 2

    for year in years:
        command = yearly_request(year, args.output, dry_run=args.dry_run)
        subprocess.run(command, check=True)
    if args.dry_run:
        write_manifest(args.output, years, "DRY_RUN_CONCLUIDO")
        return 0

    files = [validate_file(
        args.output / f"glorys12v1_ASE_JJA_{year}.nc", year) for year in years]
    path = write_manifest(args.output, years, "DOWNLOAD_VALIDADO", files)
    print(f"Manifesto -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

