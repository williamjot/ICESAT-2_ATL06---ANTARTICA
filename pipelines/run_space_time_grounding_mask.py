"""Classifica ATL06 em grounded/GZ/floating/unknown com suporte temporal."""

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.io.memory import iter_points, free_memory_gb
from thwaites.logging import setup_logging
from thwaites.qc.grounded_mask import load_bedmachine_roi, sample_fields_at
from thwaites.qc.grounding_zone import (
    FLOATING_CONFIDENT, GROUNDED_CONFIDENT, STATE_NAMES, SUPPORT_NAMES,
    build_grounding_fields, classify_points,
)


class _Writer:
    def __init__(self, path):
        self.path, self.writer, self.n = Path(path), None, 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, frame):
        import pyarrow as pa
        import pyarrow.parquet as pq
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self.writer.write_table(table)
        self.n += len(frame)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--input", default="atl06_filtered.parquet")
    parser.add_argument("--grounded-output", default="atl06_grounded_spacetime.parquet")
    parser.add_argument("--floating-output", default="atl06_floating_spacetime.parquet")
    parser.add_argument("--classification-output", default="atl06_grounding_classification.parquet")
    parser.add_argument("--audit-output", default="grounding_mask_audit_5km.parquet")
    parser.add_argument("--audit-cell-m", type=float, default=5000.0)
    parser.add_argument("--gl", default="data/grounding/processed/InSAR_GL_ASE_v02.1.gpkg")
    parser.add_argument("--gz", default="data/grounding/processed/Antarctic_GZ_ASE_2018-2020_v01.1.gpkg")
    parser.add_argument("--width-quantile", type=float, default=.95)
    parser.add_argument("--positional-uncertainty-m", type=float, default=500.)
    parser.add_argument("--densify-step-m", type=float, default=250.)
    parser.add_argument("--coast-buffer-m", type=float, default=None)
    parser.add_argument("--batch-rows", type=int, default=1_000_000)
    args = parser.parse_args()

    import geopandas as gpd
    import pyarrow.parquet as pq
    from scipy.ndimage import distance_transform_edt

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="space_time_grounding_mask")
    source = cfg.paths.interim / args.input
    gl_path, gz_path = ROOT / args.gl, ROOT / args.gz
    if not source.exists():
        raise FileNotFoundError(source)
    if not gl_path.exists() or not gz_path.exists():
        raise FileNotFoundError("produtos NSIDC ausentes; rode fetch_grounding_products.py")
    gl, gz = gpd.read_file(gl_path), gpd.read_file(gz_path)
    sx, sy, bm = load_bedmachine_roi(cfg)
    fields = build_grounding_fields(
        sx, sy, bm, gl, gz,
        years=range(cfg.temporal.year_start, cfg.temporal.year_end + 1),
        step_m=args.densify_step_m, width_quantile=args.width_quantile,
        positional_uncertainty_m=args.positional_uncertainty_m)

    # Oceano + rocha, sem incluir gelo flutuante: o buffer costeiro nao pode
    # reintroduzir a antiga grounding line estatica do BedMachine.
    invalid_ground = np.isin(bm, [0, 1])
    pixel_m = abs(float(sx[1] - sx[0]))
    dist_land_ocean = distance_transform_edt(~invalid_ground).astype(np.float32) * pixel_m
    coast_buffer = (args.coast_buffer_m if args.coast_buffer_m is not None
                    else cfg.grounded.buffer_coast_m)
    names = list(pq.ParquetFile(source).schema_arrow.names)
    if missing := {"x", "y", "t_year", "mask_class"}.difference(names):
        raise ValueError(f"entrada sem colunas: {sorted(missing)}")

    writers = {
        "grounded": _Writer(cfg.paths.interim / args.grounded_output),
        "floating": _Writer(cfg.paths.interim / args.floating_output),
        "classification": _Writer(cfg.paths.interim / args.classification_output),
    }
    stats, by_year_state, by_year_support = Counter(), Counter(), Counter()
    audit_parts = []
    try:
        for frame in iter_points(source, names, batch_rows=args.batch_rows,
                                 do_downcast=False):
            result = classify_points(frame.x, frame.y, frame.t_year,
                                     frame.mask_class, fields)
            for name, values in result.items():
                frame[name] = values
            sampled = sample_fields_at(frame.x, frame.y, sx, sy,
                                       {"dist_land_ocean_m": dist_land_ocean})
            frame["dist_land_ocean_m"] = sampled["dist_land_ocean_m"].astype(np.float32)
            acquisition_year = np.floor(frame.t_year.to_numpy()).astype(int)
            state, support = frame.grounding_state.to_numpy(), frame.grounding_support.to_numpy()
            audit = pd.DataFrame({
                "cell_x": np.floor(frame.x.to_numpy() / args.audit_cell_m).astype(np.int32),
                "cell_y": np.floor(frame.y.to_numpy() / args.audit_cell_m).astype(np.int32),
                "year": acquisition_year.astype(np.int16),
                "grounding_state": state.astype(np.int8),
            })
            audit_parts.append(
                audit.groupby(["cell_x", "cell_y", "year", "grounding_state"],
                              observed=True).size().rename("n").reset_index())
            for year in np.unique(acquisition_year):
                select = acquisition_year == year
                for code, count in zip(*np.unique(state[select], return_counts=True)):
                    by_year_state[(int(year), int(code))] += int(count)
                for code, count in zip(*np.unique(support[select], return_counts=True)):
                    by_year_support[(int(year), int(code))] += int(count)
            grounded = ((state == GROUNDED_CONFIDENT) &
                        (frame.dist_land_ocean_m.to_numpy() >= coast_buffer))
            floating = state == FLOATING_CONFIDENT
            stats.update(n_in=len(frame), n_grounded=int(grounded.sum()),
                         n_floating=int(floating.sum()),
                         n_grounding_zone=int((state == 2).sum()),
                         n_unknown=int((state == 0).sum()),
                         grounded_removed_coast=int(((state == GROUNDED_CONFIDENT) & ~grounded).sum()))
            writers["classification"].write(frame[[
                "x", "y", "t_year", "mask_class", "grounding_state",
                "grounding_support", "grounding_line_year",
                "dist_observed_gl_m", "transition_radius_m"]])
            writers["grounded"].write(frame.loc[grounded])
            writers["floating"].write(frame.loc[floating])
            log.info(f"{stats['n_in']:,} lidos | grounded {stats['n_grounded']:,} | "
                     f"GZ {stats['n_grounding_zone']:,} | floating {stats['n_floating']:,} | "
                     f"unknown {stats['n_unknown']:,} | livre {free_memory_gb():.1f} GB")
            del frame
            gc.collect()
    finally:
        for writer in writers.values():
            writer.close()

    audit_path = cfg.paths.interim / args.audit_output
    audit_table = pd.concat(audit_parts, ignore_index=True)
    audit_table = (audit_table.groupby(
        ["cell_x", "cell_y", "year", "grounding_state"], observed=True)["n"]
        .sum().reset_index())
    audit_table["x"] = (audit_table["cell_x"] + 0.5) * args.audit_cell_m
    audit_table["y"] = (audit_table["cell_y"] + 0.5) * args.audit_cell_m
    audit_table.to_parquet(audit_path, index=False)

    report = {
        "method": "space_time_grounding_classification_v1",
        "input": str(source),
        "outputs": {name: str(w.path) for name, w in writers.items()},
        "audit_output": str(audit_path),
        "audit_cell_m": float(args.audit_cell_m),
        "products": {"grounding_lines": "NSIDC-0498 v2.1",
                     "grounding_zone_up_dn": "NSIDC-0778 v1.1",
                     "static_topological_prior": "BedMachine Antarctica v4"},
        "roi_lonlat": [cfg.roi.lon_min, cfg.roi.lon_max,
                       cfg.roi.lat_min, cfg.roi.lat_max],
        "state_codes": STATE_NAMES, "support_codes": SUPPORT_NAMES,
        "counts": dict(stats),
        "by_year_state": {str(y): {STATE_NAMES[c]: by_year_state.get((y, c), 0)
                                    for c in STATE_NAMES}
                          for y in range(cfg.temporal.year_start, cfg.temporal.year_end + 1)},
        "by_year_support": {str(y): {SUPPORT_NAMES[c]: by_year_support.get((y, c), 0)
                                      for c in SUPPORT_NAMES}
                            for y in range(cfg.temporal.year_start, cfg.temporal.year_end + 1)},
        "width_model": fields.width_report,
        "coast_buffer_m": float(coast_buffer),
        "missing_epoch_policy": "historical transition envelope -> unknown; no temporal interpolation",
        "limitations": [
            "BedMachine v4 remains a static topological prior away from observed transitions.",
            "NSIDC-0778 widths are from 2018 and are transferred by glacier name.",
            "Unmatched glacier names use the median p95 width in the ROI and support code 3."]}
    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    target = cfg.paths.tables / "grounding_space_time_report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"relatorio -> {target}")


if __name__ == "__main__":
    main()
