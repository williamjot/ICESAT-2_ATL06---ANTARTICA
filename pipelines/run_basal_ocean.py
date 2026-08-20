"""Cadeia curta e incremental: derretimento basal + mecanismo oceânico.

Evita o ramo completo do projeto e pula automaticamente etapas cujas saídas
estão mais novas que todas as entradas diretas. Cada estágio roda em processo
separado para devolver memória ao Windows.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config  # noqa: E402


@dataclass
class Stage:
    name: str
    script: str
    inputs: list[Path]
    outputs: list[Path]
    args: list[str]


def _fresh(stage: Stage) -> bool:
    if not stage.outputs or any(not path.exists() for path in stage.outputs):
        return False
    if any(not path.exists() for path in stage.inputs):
        return False
    return min(path.stat().st_mtime_ns for path in stage.outputs) >= max(
        path.stat().st_mtime_ns for path in stage.inputs)


def _stages(profile: str) -> list[Stage]:
    cfg = load_config(profile)
    code = ROOT / "pipelines"
    common = [ROOT / "config" / "default.yaml", ROOT / "config" / f"{profile}.yaml"]
    ocean = ROOT / "data" / "ocean" / "bas_melt_01468" / "Thwaites_MAVS_Timeseries_TSV.dat"
    bedmachine = next(iter(sorted(cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))))
    grounding_gl = ROOT / "data" / "grounding" / "processed" / "InSAR_GL_ASE_v02.1.gpkg"
    grounding_gz = ROOT / "data" / "grounding" / "processed" / "Antarctic_GZ_ASE_2018-2020_v01.1.gpkg"
    return [
        Stage("grounding_spacetime", "run_space_time_grounding_mask.py",
              [cfg.paths.interim / "atl06_filtered.parquet", grounding_gl,
               grounding_gz, code / "run_space_time_grounding_mask.py", *common],
              [cfg.paths.interim / "atl06_floating_spacetime.parquet",
               cfg.paths.interim / "atl06_grounding_classification.parquet",
               cfg.paths.interim / "grounding_mask_audit_5km.parquet",
               cfg.paths.tables / "grounding_space_time_report.json"], []),
        Stage("shelf_mask", "run_shelf_mask.py",
              [cfg.paths.interim / "atl06_floating_spacetime.parquet",
               cfg.paths.data_dir / cfg.shelf.fronts_path,
               code / "run_shelf_mask.py", *common],
              [cfg.paths.interim / "atl06_shelf_dynamic.parquet",
               cfg.paths.interim / "shelf_mask_audit_5km.parquet",
               cfg.paths.tables / "shelf_mask_dynamic_report.json"],
              ["--input", "atl06_floating_spacetime.parquet"]),
        Stage("shelf_windows", "run_shelf_windows.py",
              [cfg.paths.interim / "atl06_shelf_dynamic.parquet",
               cfg.paths.data_dir / "velocity_itslive_annual.nc",
               code / "run_shelf_windows.py", *common],
              [cfg.paths.dhdt_dir / "shelf_lagrangian_windows.parquet"], []),
        Stage("shelf_divergence", "run_shelf_divergence.py",
              [cfg.paths.dhdt_dir / "shelf_lagrangian_windows.parquet",
               cfg.paths.data_dir / "velocity_itslive_annual.nc", bedmachine,
               code / "run_shelf_divergence.py", *common],
              [cfg.paths.dhdt_dir / "shelf_windows_divergence.parquet"], []),
        Stage("basal_melt", "run_basal_melt.py",
              [cfg.paths.dhdt_dir / "shelf_windows_divergence.parquet",
               cfg.paths.data_dir / "smb_thwaites.nc",
               cfg.paths.data_dir / cfg.firn.path,
               code / "run_basal_melt.py", *common],
              [cfg.paths.dhdt_dir / "shelf_basal_melt.parquet",
               cfg.paths.tables / "basal_melt_report.json"], []),
        Stage("ocean_mechanism", "run_ocean_mechanism.py",
              [cfg.paths.dhdt_dir / "shelf_basal_melt.parquet", ocean,
               code / "run_ocean_mechanism.py",
               ROOT / "thwaites" / "ocean" / "bas_melt.py", *common],
              [cfg.paths.tables / "ocean_mechanism_report.json",
               cfg.paths.figures / "ocean_melt_timeseries.png",
               cfg.paths.figures / "ocean_melt_speed_vs_thermal.png",
               cfg.paths.figures / "basal_melt_thwaites_ocean_site.png"], []),
    ]


def main():
    parser = argparse.ArgumentParser(description="Cadeia incremental basal + oceano.")
    parser.add_argument("--profiles", nargs="+", default=["jja", "djf"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from-stage", choices=[
        "grounding_spacetime", "shelf_mask", "shelf_windows",
        "shelf_divergence", "basal_melt", "ocean_mechanism"])
    args = parser.parse_args()

    ocean_file = ROOT / "data" / "ocean" / "bas_melt_01468" / "Thwaites_MAVS_Timeseries_TSV.dat"
    if not ocean_file.exists():
        subprocess.run([sys.executable, str(ROOT / "pipelines" / "fetch_ocean_melt.py")],
                       cwd=ROOT, check=True)

    for profile in args.profiles:
        print(f"\n[{profile.upper()}] cadeia basal + oceano")
        started = args.from_stage is None
        for stage in _stages(profile):
            if args.from_stage == stage.name:
                started = True
            if not started:
                continue
            if not args.force and _fresh(stage):
                print(f"  SKIP {stage.name}: cache atualizado")
                continue
            missing = [str(path) for path in stage.inputs if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{stage.name}: entradas ausentes: {missing}")
            command = [sys.executable, str(ROOT / "pipelines" / stage.script),
                       "--profile", profile, *stage.args]
            print(f"  RUN  {stage.name}")
            t0 = time.perf_counter()
            subprocess.run(command, cwd=ROOT, check=True)
            print(f"       concluído em {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
