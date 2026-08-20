"""Mapas JJA para toda a ROI da Amundsen Sea Embayment.

ROI geográfica exata: 115°W–95°W; 77,5°S–73°S. A borda é densificada antes
da reprojeção para EPSG:3031; os dados são mascarados pelo polígono projetado,
não por uma caixa retangular em coordenadas polares.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.path import Path as MplPath
from pyproj import Transformer

from thwaites import load_config
from thwaites.diagnostics import aggregate_basal_cells
from thwaites.logging import setup_logging
from thwaites.viz.basemap import draw_basemap, draw_calving_fronts, add_scale_bar

from run_basal_diagnostic_maps import (
    _load_bed_geometry,
    _load_velocity,
    _safe_stats,
)

LON_MIN, LON_MAX = -115.0, -95.0
LAT_MIN, LAT_MAX = -77.5, -73.0
DPI = 220


def roi_polygon(n_per_edge=240):
    """Borda densificada da ROI em EPSG:3031 e Path para mascaramento."""
    bottom_lon = np.linspace(LON_MIN, LON_MAX, n_per_edge)
    right_lat = np.linspace(LAT_MIN, LAT_MAX, n_per_edge)
    top_lon = np.linspace(LON_MAX, LON_MIN, n_per_edge)
    left_lat = np.linspace(LAT_MAX, LAT_MIN, n_per_edge)
    lon = np.r_[bottom_lon, np.full(n_per_edge, LON_MAX),
                top_lon, np.full(n_per_edge, LON_MIN), bottom_lon[:1]]
    lat = np.r_[np.full(n_per_edge, LAT_MIN), right_lat,
                np.full(n_per_edge, LAT_MAX), left_lat, LAT_MIN]
    transformer = Transformer.from_crs(4326, 3031, always_xy=True)
    x, y = transformer.transform(lon, lat)
    vertices = np.column_stack([x, y])
    extent = (float(np.min(x)), float(np.max(x)),
              float(np.min(y)), float(np.max(y)))
    return np.asarray(x), np.asarray(y), MplPath(vertices), extent


def inside_roi(path: MplPath, x, y):
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    selected = path.contains_points(
        np.column_stack([x_arr.ravel(), y_arr.ravel()]), radius=1.0)
    return selected.reshape(x_arr.shape)


def filter_frame_roi(data, path, x_col, y_col):
    keep = inside_roi(path, data[x_col].to_numpy(), data[y_col].to_numpy())
    return data.loc[keep].copy()


def draw_roi_base(ax, cfg, roi_x, roi_y, extent, target_px=800):
    draw_basemap(ax, cfg, *extent, target_px=target_px)
    draw_calving_fronts(ax, cfg, 2022.5)
    ax.plot(roi_x / 1000, roi_y / 1000, color="#111111", lw=1.25,
            ls=(0, (4, 2)), zorder=12)
    ax.set_xlim(extent[0] / 1000, extent[1] / 1000)
    ax.set_ylim(extent[2] / 1000, extent[3] / 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    ax.grid(alpha=0.18, lw=0.4, ls="--")
    add_scale_bar(ax, 100)


def add_shelf_labels(ax, records):
    for shelf, group in records.groupby("shelf"):
        ax.text(group.x_ref.median() / 1000, group.y_ref.median() / 1000,
                str(shelf), fontsize=7, ha="center", va="center", zorder=15,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#555555",
                          alpha=0.78, lw=0.45))


def plot_basal(cells, records, cfg, roi_x, roi_y, extent, path):
    values = cells.basal_melt.to_numpy(dtype=float)
    limit = float(max(np.nanquantile(np.abs(values), 0.95), 1.0))
    fig, ax = plt.subplots(figsize=(10.8, 8.7), constrained_layout=True,
                           facecolor="white")
    draw_roi_base(ax, cfg, roi_x, roi_y, extent)
    artist = ax.scatter(
        cells.x / 1000, cells.y / 1000, c=values, s=32, marker="s",
        cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        edgecolor="#333333", linewidth=0.15, zorder=9, rasterized=True)
    cbar = fig.colorbar(artist, ax=ax, shrink=0.80, extend="both", pad=0.02)
    cbar.set_label("Derretimento basal (m gelo/ano; positivo = derretimento)")
    add_shelf_labels(ax, records)
    ax.set_title(
        "Derretimento basal Lagrangiano — Amundsen Sea Embayment, JJA\n"
        f"2019–2025 · {len(cells)} células observadas de 5 km · "
        f"mediana {np.median(values):+.2f} m/ano",
        loc="left", fontweight="bold")
    ax.text(
        0.014, 0.985,
        "PRODUTO EXPLORATÓRIO\nsem interpolação entre células\n"
        "linha tracejada = limite exato da ROI",
        transform=ax.transAxes, va="top", fontsize=7.6, color="#8b0000",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#8b0000",
                  alpha=0.94), zorder=20)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_basal_uncertainty(cells, records, cfg, roi_x, roi_y, extent, path):
    values = cells.sigma_stat_lower.to_numpy(dtype=float)
    vmax = float(max(np.nanquantile(values, 0.95), 1.0))
    fig, ax = plt.subplots(figsize=(10.8, 8.7), constrained_layout=True,
                           facecolor="white")
    draw_roi_base(ax, cfg, roi_x, roi_y, extent)
    artist = ax.scatter(
        cells.x / 1000, cells.y / 1000, c=values, s=32, marker="s",
        cmap="viridis", norm=Normalize(vmin=0, vmax=vmax),
        edgecolor="#333333", linewidth=0.15, zorder=9, rasterized=True)
    cbar = fig.colorbar(artist, ax=ax, shrink=0.80, extend="max", pad=0.02)
    cbar.set_label("σ estatístico mínimo de m_b (m/ano)")
    add_shelf_labels(ax, records)
    ax.set_title(
        "Incerteza estatística mínima do derretimento basal — ASE, JJA\n"
        f"mediana {np.nanmedian(values):.2f} m/ano · não inclui todas as fontes",
        loc="left", fontweight="bold")
    ax.text(
        0.014, 0.985,
        "LIMITE INFERIOR\nexclui incerteza de SMB, FAC, maré, trajetória,\n"
        "velocidade, divergência e espessura",
        transform=ax.transAxes, va="top", fontsize=7.5, color="#8b0000",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#8b0000",
                  alpha=0.94), zorder=20)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def trusted_jja_nodes(nodes):
    selected = nodes.copy()
    if "reliability" in selected:
        selected = selected[
            selected.reliability.astype(str).str.startswith("confi", na=False)]
    finite = (np.isfinite(selected.dhdt) & np.isfinite(selected.dhdt_err))
    selected = selected[finite].copy()
    selected["thinning_significant"] = (
        selected.dhdt + 1.96 * selected.dhdt_err < 0)
    return selected


def plot_diagnostic(cells, nodes, velocity, geometry, cfg, roi_x, roi_y,
                    roi_path, extent, path):
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 11.3), constrained_layout=True,
                             facecolor="white")
    for ax in axes.ravel():
        draw_roi_base(ax, cfg, roi_x, roi_y, extent, target_px=650)

    ax = axes[0, 0]
    melt = cells.basal_melt.to_numpy(dtype=float)
    limit = float(max(np.quantile(np.abs(melt), 0.95), 1.0))
    artist = ax.scatter(
        cells.x / 1000, cells.y / 1000, c=melt, s=24, marker="s",
        cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        edgecolor="none", zorder=9, rasterized=True)
    fig.colorbar(artist, ax=ax, shrink=0.78, extend="both", pad=0.02,
                 label="m_b (m gelo/ano)")
    ax.set_title("a. Derretimento basal JJA\nperda potencial de suporte flutuante",
                 loc="left", fontweight="bold")

    ax = axes[0, 1]
    gx, gy = np.meshgrid(velocity["x"], velocity["y"])
    in_roi = inside_roi(roi_path, gx, gy)
    velocity_qc = (in_roi & (velocity["speed"] >= 100)
                   & (velocity["count"] == len(velocity["years"]))
                   & (velocity["trend_r2"] >= 0.50))
    trend = np.where(velocity_qc, velocity["trend_percent"], np.nan)
    vlim = float(max(np.nanquantile(np.abs(trend), 0.97), 0.5))
    artist = ax.pcolormesh(
        velocity["x"] / 1000, velocity["y"] / 1000, trend,
        cmap="BrBG_r", norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim),
        shading="auto", alpha=0.85, zorder=8, rasterized=True)
    fig.colorbar(artist, ax=ax, shrink=0.78, extend="both", pad=0.02,
                 label="tendência da velocidade (%/ano)")
    ax.set_title("b. Mudança dinâmica 2019–2025\n"
                 "7 anos, R²≥0,50 e velocidade ≥100 m/ano",
                 loc="left", fontweight="bold")

    ax = axes[1, 0]
    non_sig = nodes[~nodes.thinning_significant]
    sig = nodes[nodes.thinning_significant]
    ax.scatter(non_sig.x / 1000, non_sig.y / 1000, s=5, color="#8d8d8d",
               alpha=0.28, edgecolor="none", zorder=8, rasterized=True)
    if len(sig):
        tlim = float(max(abs(np.quantile(sig.dhdt, 0.02)), 0.5))
        artist = ax.scatter(sig.x / 1000, sig.y / 1000, c=sig.dhdt, s=9,
                            cmap="magma_r", vmin=-tlim, vmax=0,
                            edgecolor="none", zorder=9, rasterized=True)
        fig.colorbar(artist, ax=ax, shrink=0.78, extend="min", pad=0.02,
                     label="dh/dt JJA (m/ano)")
    ax.set_title("c. Adelgaçamento do gelo aterrado — JJA\n"
                 "colorido quando IC95% está abaixo de zero",
                 loc="left", fontweight="bold")

    ax = axes[1, 1]
    bx, by = np.meshgrid(geometry["x"], geometry["y"])
    slope = geometry["along_flow_slope_m_km"].copy()
    slope[~inside_roi(roi_path, bx, by)] = np.nan
    slim = float(max(np.nanquantile(np.abs(slope), 0.97), 1.0))
    artist = ax.pcolormesh(
        geometry["x"] / 1000, geometry["y"] / 1000, slope,
        cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-slim, vcenter=0, vmax=slim),
        shading="auto", alpha=0.85, zorder=8, rasterized=True)
    fig.colorbar(artist, ax=ax, shrink=0.78, extend="both", pad=0.02,
                 label="declive do leito no fluxo (m/km)")
    ax.set_title("d. Geometria marinha até 100 km da linha de aterramento\n"
                 "positivo = leito retrógrado; suavização σ=2 km",
                 loc="left", fontweight="bold")

    fig.suptitle(
        "Diagnóstico observacional JJA — Amundsen Sea Embayment",
        fontsize=15, fontweight="bold")
    fig.text(
        0.5, -0.018,
        "ROI: 115°W–95°W; 77,5°S–73°S. Camadas independentes; o mapa não "
        "fornece probabilidade ou data de colapso.",
        ha="center", fontsize=9, color="#8b0000", fontweight="bold")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return trend, slope


def main():
    cfg = load_config("jja")
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="ase_jja_diagnostic_maps")
    roi_x, roi_y, roi_path, extent = roi_polygon()
    output = ROOT / "outputs" / "diagnostico_ase_jja"
    output.mkdir(parents=True, exist_ok=True)

    melt_path = cfg.paths.dhdt_dir / "shelf_basal_melt.parquet"
    melt = pd.read_parquet(melt_path)
    melt = filter_frame_roi(melt, roi_path, "x_ref", "y_ref")
    melt = melt[melt.shelf.notna()].copy()
    cells = aggregate_basal_cells(melt, 5_000.0, shelf_pattern="")
    if cells.empty:
        raise RuntimeError("Nenhuma parcela basal JJA dentro da ROI")
    plot_basal(cells, melt, cfg, roi_x, roi_y, extent,
               output / "mapa_derretimento_basal_ase_jja.png")
    plot_basal_uncertainty(cells, melt, cfg, roi_x, roi_y, extent,
                           output / "mapa_incerteza_basal_ase_jja.png")

    nodes = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet")
    nodes = filter_frame_roi(trusted_jja_nodes(nodes), roi_path, "x", "y")
    velocity = _load_velocity(cfg.paths.data_dir / "velocity_itslive_annual.nc", extent)
    bed_path = sorted(cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))[0]
    geometry = _load_bed_geometry(bed_path, extent, velocity)
    trend, slope = plot_diagnostic(
        cells, nodes, velocity, geometry, cfg, roi_x, roi_y, roi_path, extent,
        output / "mapa_diagnostico_ase_jja.png")
    # Prévia leve para inspeção e visualização no aplicativo; o PNG completo
    # permanece como produto científico em alta resolução.
    from PIL import Image
    preview_source = output / "mapa_diagnostico_ase_jja.png"
    preview = Image.open(preview_source)
    preview.thumbnail((1600, 1600))
    preview.save(output / "mapa_diagnostico_ase_jja_preview.png", optimize=True)

    shelf_stats = {}
    selected_melt = melt.copy()
    if "reliable" in selected_melt:
        selected_melt = selected_melt[selected_melt.reliable.astype(bool)]
    for shelf, group in selected_melt.groupby("shelf"):
        shelf_stats[str(shelf)] = _safe_stats(group.basal_melt)

    valid_trend = np.isfinite(trend)
    valid_slope = np.isfinite(slope)
    report = {
        "status": "DIAGNOSTICO_OBSERVACIONAL_JJA_NAO_PROGNOSTICO",
        "roi": {
            "name": "Amundsen Sea Embayment, West Antarctica",
            "longitude": [-115.0, -95.0], "latitude": [-77.5, -73.0],
            "projection": "EPSG:3031",
        },
        "season": "JJA only",
        "period": "2019-2025",
        "basal_melt_cells_5km": {
            **_safe_stats(cells.basal_melt),
            "sigma_statistical_lower_bound": _safe_stats(cells.sigma_stat_lower),
            "by_shelf_records": shelf_stats,
            "no_spatial_interpolation": True,
        },
        "grounded_dhdt_jja": {
            "trusted_nodes": int(len(nodes)),
            "significant_thinning_nodes": int(nodes.thinning_significant.sum()),
            "fraction_significant_thinning": float(nodes.thinning_significant.mean()),
            "significant_dhdt": _safe_stats(
                nodes.loc[nodes.thinning_significant, "dhdt"]),
        },
        "velocity_trend_2019_2025": {
            "filter": "7 years, OLS R2>=0.50, median speed>=100 m/yr",
            **_safe_stats(trend[valid_trend]),
            "fraction_positive": float(np.mean(trend[valid_trend] > 0)),
        },
        "marine_bed_slope_within_100km_grounding_line": {
            "bed_smoothing": "Gaussian sigma=2 km before gradient",
            **_safe_stats(slope[valid_slope]),
            "fraction_retrograde_positive": float(np.mean(slope[valid_slope] > 0)),
            "bed_error_m": _safe_stats(geometry["errbed"][valid_slope]),
        },
        "limitations": [
            "JJA caracteriza apenas o inverno austral; nao representa o ciclo anual completo",
            "m_b tem somente limite inferior da incerteza estatistica",
            "BedMachine tem epoca nominal aproximada de 2015",
            "GSFC-FDM termina em junho de 2022 e foi extrapolado ate 2025",
            "o mapa diagnostica sinais presentes; nao estima probabilidade ou data de colapso",
        ],
    }
    report_path = output / "diagnostico_ase_jja.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    for path in sorted(output.glob("*.png")):
        log.info(f"mapa -> {path}")
    log.info(f"relatorio -> {report_path}")


if __name__ == "__main__":
    main()
