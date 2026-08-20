"""Controle visual e temporal dos novos produtos de aterramento na ROI dh/dt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config


def _km(value, _position):
    return f"{value / 1000:.0f}"


def _read_products(cfg):
    base = cfg.paths.data_dir / "grounding" / "processed"
    gl = gpd.read_file(base / "InSAR_GL_ASE_v02.1.gpkg")
    gz = gpd.read_file(base / "Antarctic_GZ_ASE_2018-2020_v01.1.gpkg")
    is2 = pd.read_parquet(base / "IS2GZANT_v01_ASE.parquet")
    is2 = gpd.GeoDataFrame(
        is2,
        geometry=gpd.points_from_xy(is2.longitude, is2.latitude),
        crs=4326,
    ).to_crs(epsg=cfg.area.epsg_polar)
    return gl, gz, is2


def _coverage_report(gl, gz, is2, cfg):
    study = gl[gl.Year.between(cfg.temporal.year_start, cfg.temporal.year_end)]
    thwaites = study[study.Glac_Name.str.contains("Thwaites", case=False, na=False)]
    ice_sheet_gz = gz[gz.Type.eq("Ice Sheet")]

    sensor_counts = {}
    for (year, sensor), count in thwaites.groupby(["Year", "Sensor"]).size().items():
        sensor_counts.setdefault(str(int(year)), {})[str(sensor)] = int(count)

    errors = {}
    for name, values in is2.groupby("feature_type").nominal_error.unique().items():
        errors[str(name)] = [float(value) for value in values if np.isfinite(value)]

    return {
        "roi": {
            "label": cfg.roi.label,
            "bbox_epsg4326": list(cfg.roi.bounding_box),
        },
        "nsidc_0498_release": "2.1",
        "grounding_lines_roi_total": int(len(gl)),
        "grounding_lines_study_by_year": {
            str(int(year)): int(count)
            for year, count in study.groupby("Year").size().items()
        },
        "thwaites_lines_by_year_sensor": sensor_counts,
        "years_without_thwaites_line": [
            year for year in range(cfg.temporal.year_start, cfg.temporal.year_end + 1)
            if year not in set(thwaites.Year.astype(int))
        ],
        "nsidc_0778_release": "1.1",
        "grounding_zone_ice_sheet_by_year": {
            str(int(year)): int(count)
            for year, count in ice_sheet_gz.groupby("Year").size().items()
        },
        "thwaites_grounding_zone": ice_sheet_gz[ice_sheet_gz.Name.eq("Thwaites")][
            ["Year", "Sensor", "Type", "Boundary"]
        ].to_dict(orient="records"),
        "is2gzant_v1": {
            "points_by_feature": {
                str(name): int(count)
                for name, count in is2.groupby("feature_type").size().items()
            },
            "tracks_by_feature": {
                str(name): int(count)
                for name, count in is2.groupby("feature_type").track.nunique().items()
            },
            "nominal_error_m": errors,
        },
        "methodological_decision": (
            "NSIDC-0778 define a largura física observada da zona de flexão; "
            "NSIDC-0498 fornece posições datadas. IS2GZANT é validação independente."
        ),
        "limitations": [
            "Na ROI, os pares Up/Dn do NSIDC-0778 para Thwaites são de 2018.",
            "Ausência de linha em um ano não autoriza interpolação temporal automática.",
            "As linhas DInSAR têm incerteza amostral documentada de aproximadamente ±500 m.",
            "IS2GZANT v1 deriva de ATL06 v3 e cobre março de 2019 a setembro de 2020.",
        ],
    }


def _plot_map(gl, gz, is2, nodes, cfg, output):
    study = gl[gl.Year.between(cfg.temporal.year_start, cfg.temporal.year_end)].copy()
    thwaites = study[study.Glac_Name.str.contains("Thwaites", case=False, na=False)].copy()
    gz_ice = gz[gz.Type.eq("Ice Sheet")].copy()
    gz_thwaites = gz_ice[gz_ice.Name.eq("Thwaites")].copy()

    finite = nodes[np.isfinite(nodes.dhdt) & np.isfinite(nodes.x) & np.isfinite(nodes.y)]
    bound = float(np.nanquantile(np.abs(finite.dhdt), 0.98))
    bound = max(0.5, min(bound, 3.0))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    years = np.arange(cfg.temporal.year_start, cfg.temporal.year_end + 1)
    cmap_year = plt.get_cmap("viridis", len(years))
    year_colors = {year: cmap_year(i) for i, year in enumerate(years)}

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0), height_ratios=(1.25, .75))
    ax_all = fig.add_subplot(grid[:, 0])
    ax_zoom = fig.add_subplot(grid[0, 1])
    ax_time = fig.add_subplot(grid[1, 1])

    for ax in (ax_all, ax_zoom):
        scatter = ax.scatter(
            finite.x, finite.y, c=finite.dhdt, s=5, cmap="RdBu",
            norm=norm, alpha=.48, linewidths=0, rasterized=True,
        )
        for year, subset in study.groupby("Year"):
            subset.plot(ax=ax, color=year_colors[int(year)], linewidth=1.2, alpha=.9)
        gz_ice[gz_ice.Boundary.eq("Up")].plot(
            ax=ax, color="black", linewidth=2.0, linestyle="--")
        gz_ice[gz_ice.Boundary.eq("Dn")].plot(
            ax=ax, color="black", linewidth=2.0, linestyle="-")
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(FuncFormatter(_km))
        ax.yaxis.set_major_formatter(FuncFormatter(_km))
        ax.set_xlabel("Easting EPSG:3031 (km)")
        ax.set_ylabel("Northing EPSG:3031 (km)")
        ax.grid(color="0.85", linewidth=.5)

    ax_all.set_title("ASE — dh/dt e produtos observacionais de aterramento")
    ax_all.set_xlim(finite.x.min() - 20_000, finite.x.max() + 20_000)
    ax_all.set_ylim(finite.y.min() - 20_000, finite.y.max() + 20_000)

    if thwaites.empty:
        raise ValueError("nenhuma linha de Thwaites no NSIDC-0498 para 2019–2025")
    x0, y0, x1, y1 = thwaites.total_bounds
    pad = 45_000
    ax_zoom.set_xlim(x0 - pad, x1 + pad)
    ax_zoom.set_ylim(y0 - pad, y1 + pad)
    ax_zoom.set_title("Thwaites — linha datada, zona F–H e validação ICESat-2")

    point_styles = {
        "Point_F": ("o", "tab:green", 20, "ICESat-2 F"),
        "Point_H": ("^", "tab:orange", 24, "ICESat-2 H"),
        "Point_Ib": ("x", "tab:purple", 18, "ICESat-2 Ib"),
    }
    for name, (marker, color, size, label) in point_styles.items():
        points = is2[is2.feature_type.eq(name)]
        points = points.cx[x0 - pad:x1 + pad, y0 - pad:y1 + pad]
        ax_zoom.scatter(
            points.geometry.x, points.geometry.y, marker=marker, color=color,
            s=size, linewidths=.8, alpha=.7, label=label,
        )
    if not gz_thwaites.empty:
        gz_thwaites[gz_thwaites.Boundary.eq("Up")].plot(
            ax=ax_zoom, color="black", linewidth=3, linestyle="--")
        gz_thwaites[gz_thwaites.Boundary.eq("Dn")].plot(
            ax=ax_zoom, color="black", linewidth=3, linestyle="-")

    all_counts = study.groupby("Year").size().reindex(years, fill_value=0)
    th_counts = thwaites.groupby("Year").size().reindex(years, fill_value=0)
    width = .38
    ax_time.bar(years - width / 2, all_counts, width=width, color="0.65", label="ROI")
    ax_time.bar(years + width / 2, th_counts, width=width,
                color=[year_colors[int(year)] for year in years], label="Thwaites")
    ax_time.set_title("Cobertura temporal NSIDC-0498")
    ax_time.set_xlabel("Ano da aquisição")
    ax_time.set_ylabel("Número de feições")
    ax_time.set_xticks(years)
    ax_time.grid(axis="y", color="0.85", linewidth=.5)
    ax_time.legend(frameon=False)

    legend_years = [
        Line2D([0], [0], color=year_colors[int(year)], lw=2, label=str(year))
        for year in years
    ]
    legend_bounds = [
        Line2D([0], [0], color="black", lw=2, ls="--", label="GZ Up — 2018"),
        Line2D([0], [0], color="black", lw=2, ls="-", label="GZ Dn — 2018"),
    ]
    ax_all.legend(handles=legend_years + legend_bounds, ncol=3, frameon=False,
                  loc="lower left", fontsize=8)
    ax_zoom.legend(frameon=False, loc="lower left", fontsize=8, ncol=3)

    colorbar = fig.colorbar(scatter, ax=[ax_all, ax_zoom], shrink=.72, pad=.02)
    colorbar.set_label("dh/dt JJA (m/ano) — apenas contexto espacial")
    fig.suptitle(
        "Controle de cobertura — linha e zona de aterramento na ROI dos mapas dh/dt",
        fontsize=14, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="QA dos produtos de aterramento.")
    parser.add_argument("--profile", default="jja")
    args = parser.parse_args()

    cfg = load_config(args.profile)
    gl, gz, is2 = _read_products(cfg)
    nodes = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes.parquet")

    # Produto de referência comum às estações; não deve ser duplicado em JJA/DJF.
    output_dir = cfg.paths.base_dir / "outputs" / "grounding"
    report = _coverage_report(gl, gz, is2, cfg)
    report_path = output_dir / "grounding_coverage_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    _plot_map(gl, gz, is2, nodes, cfg, output_dir / "grounding_products_qa.png")


if __name__ == "__main__":
    main()
