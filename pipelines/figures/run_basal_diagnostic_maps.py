"""Mapas de derretimento basal e diagnóstico integrado da Thwaites.

Produtos:
  outputs/diagnostico/mapa_derretimento_basal_jja.png
  outputs/diagnostico/mapa_derretimento_basal_djf.png
  outputs/diagnostico/mapa_derretimento_basal_sazonal.png
  outputs/diagnostico/mapa_diagnostico_vulnerabilidade.png
  outputs/diagnostico/diagnostico_vulnerabilidade.json

O diagnóstico reúne sinais independentes; não estima probabilidade ou data de
colapso e não usa um índice composto com pesos arbitrários.
"""

from __future__ import annotations

import argparse
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
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

from thwaites import load_config
from thwaites.diagnostics import (
    aggregate_basal_cells,
    along_flow_bed_slope,
    consensus_thinning,
    seasonal_basal_contrast,
    velocity_percent_trend,
)
from thwaites.logging import setup_logging
from thwaites.viz.basemap import draw_basemap, draw_calving_fronts, add_scale_bar

DPI = 220
INK = "#1f2933"


def _limits(frames, column, quantile=0.95, floor=1.0):
    values = np.concatenate([
        frame[column].to_numpy(dtype=float) for frame in frames if len(frame)])
    values = values[np.isfinite(values)]
    return float(max(np.quantile(np.abs(values), quantile), floor))


def _finish_map(ax, extent, *, scale_km=20):
    ax.set_xlim(extent[0] / 1000, extent[1] / 1000)
    ax.set_ylim(extent[2] / 1000, extent[3] / 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    ax.grid(alpha=0.18, lw=0.4, ls="--")
    add_scale_bar(ax, scale_km)


def _base(ax, cfg, extent, front_epoch=2022.5, target_px=900):
    draw_basemap(ax, cfg, *extent, target_px=target_px)
    draw_calving_fronts(ax, cfg, front_epoch)


def _plot_basal_single(cells, cfg, extent, limit, season, path):
    fig, ax = plt.subplots(figsize=(9.2, 8.2), facecolor="white",
                           constrained_layout=True)
    _base(ax, cfg, extent)
    scatter = ax.scatter(
        cells["x"] / 1000, cells["y"] / 1000,
        c=cells["basal_melt"], s=46, marker="s", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        edgecolor="#333333", linewidth=0.18, zorder=8, rasterized=True)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.78, extend="both", pad=0.02)
    cbar.set_label("Derretimento basal (m gelo/ano; positivo = derretimento)")
    sigma = cells["sigma_stat_lower"].median()
    ax.set_title(
        f"Derretimento basal Lagrangiano — Thwaites {season.upper()}\n"
        f"mediana {cells['basal_melt'].median():+.2f} m/ano · "
        f"{len(cells)} células observadas de 5 km",
        loc="left", fontweight="bold")
    ax.text(
        0.015, 0.985,
        "PRODUTO EXPLORATÓRIO\n"
        f"σ estatístico mediano ≥ {sigma:.2f} m/ano\n"
        "sem interpolação entre células",
        transform=ax.transAxes, va="top", fontsize=7.5, color="#8b0000",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#8b0000",
                  alpha=0.93), zorder=20)
    _finish_map(ax, extent)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_basal_comparison(jja, djf, contrast, cfg, extent, path):
    melt_limit = _limits([jja, djf], "basal_melt")
    delta_limit = _limits([contrast], "basal_melt_djf_minus_jja")
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.7), facecolor="white",
                             constrained_layout=True)
    for ax in axes:
        _base(ax, cfg, extent, target_px=650)
    for ax, frame, title in zip(
            axes[:2], (jja, djf), ("a. JJA — inverno", "b. DJF — verão")):
        artist = ax.scatter(
            frame["x"] / 1000, frame["y"] / 1000,
            c=frame["basal_melt"], s=34, marker="s", cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-melt_limit, vcenter=0, vmax=melt_limit),
            edgecolor="none", zorder=8, rasterized=True)
        ax.set_title(title, loc="left", fontweight="bold")
        _finish_map(ax, extent)
    cbar = fig.colorbar(artist, ax=axes[:2], shrink=0.78, extend="both", pad=0.01)
    cbar.set_label("m_b (m gelo/ano)")
    artist_delta = axes[2].scatter(
        contrast["x"] / 1000, contrast["y"] / 1000,
        c=contrast["basal_melt_djf_minus_jja"], s=34, marker="s",
        cmap="PuOr_r",
        norm=TwoSlopeNorm(vmin=-delta_limit, vcenter=0, vmax=delta_limit),
        edgecolor="none", zorder=8, rasterized=True)
    axes[2].set_title("c. Contraste DJF − JJA", loc="left", fontweight="bold")
    _finish_map(axes[2], extent)
    cbar = fig.colorbar(artist_delta, ax=axes[2], shrink=0.78,
                        extend="both", pad=0.01)
    cbar.set_label("diferença sazonal (m gelo/ano)")
    fig.suptitle(
        "Derretimento basal observado na Thwaites — comparação sazonal",
        fontsize=14, fontweight="bold")
    fig.text(
        0.5, -0.025,
        "Medianas em células de 5 km; o contraste existe somente onde JJA e DJF "
        "têm dados. Não é uma interpolação nem uma tendência temporal.",
        ha="center", fontsize=8, color="#555555")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_velocity(path, extent, stride=4):
    from netCDF4 import Dataset, num2date

    with Dataset(path) as source:
        x_all = np.asarray(source["x"][:], dtype=float)
        y_all = np.asarray(source["y"][:], dtype=float)
        ix = np.where((x_all >= extent[0]) & (x_all <= extent[1]))[0]
        iy = np.where((y_all >= extent[2]) & (y_all <= extent[3]))[0]
        xs = slice(int(ix.min()), int(ix.max()) + 1, stride)
        ys = slice(int(iy.min()), int(iy.max()) + 1, stride)
        x = x_all[xs]
        y = y_all[ys]
        vx = np.ma.filled(source["vx"][:, ys, xs], np.nan).astype(float)
        vy = np.ma.filled(source["vy"][:, ys, xs], np.nan).astype(float)
        dates = num2date(source["time"][:], source["time"].units)
        years = np.asarray([date.year for date in dates], dtype=float)
    if y[0] > y[-1]:
        y, vx, vy = y[::-1], vx[:, ::-1], vy[:, ::-1]
    speed = np.hypot(vx, vy)
    slope, trend_percent, count = velocity_percent_trend(speed, years)
    mean_speed = np.divide(
        np.nansum(speed, axis=0), count,
        out=np.full(speed.shape[1:], np.nan), where=count > 0)
    prediction = mean_speed[None, :, :] + slope[None, :, :] * (
        years[:, None, None] - years.mean())
    residual_ss = np.nansum((speed - prediction) ** 2, axis=0)
    total_ss = np.nansum((speed - mean_speed[None, :, :]) ** 2, axis=0)
    trend_r2 = np.divide(
        total_ss - residual_ss, total_ss,
        out=np.full_like(total_ss, np.nan), where=total_ss > 0)
    median_vx = np.ma.median(np.ma.masked_invalid(vx), axis=0).filled(np.nan)
    median_vy = np.ma.median(np.ma.masked_invalid(vy), axis=0).filled(np.nan)
    median_speed = np.ma.median(np.ma.masked_invalid(speed), axis=0).filled(np.nan)
    return {
        "x": x, "y": y, "vx": median_vx,
        "vy": median_vy, "speed": median_speed,
        "trend_percent": trend_percent, "trend_r2": trend_r2,
        "count": count, "years": years,
    }


def _load_bed_geometry(path, extent, velocity):
    from netCDF4 import Dataset

    with Dataset(path) as source:
        x_all = np.asarray(source["x"][:], dtype=float)
        y_all = np.asarray(source["y"][:], dtype=float)
        ix = np.where((x_all >= extent[0]) & (x_all <= extent[1]))[0]
        iy = np.where((y_all >= extent[2]) & (y_all <= extent[3]))[0]
        xs = slice(int(ix.min()), int(ix.max()) + 1)
        ys = slice(int(iy.min()), int(iy.max()) + 1)
        x, y = x_all[xs], y_all[ys]
        bed = np.ma.filled(source["bed"][ys, xs], np.nan).astype(float)
        mask = np.asarray(source["mask"][ys, xs])
        errbed = np.ma.array(source["errbed"][ys, xs], dtype=float).filled(np.nan)
    if y[0] > y[-1]:
        y, bed, mask, errbed = y[::-1], bed[::-1], mask[::-1], errbed[::-1]

    points_y, points_x = np.meshgrid(y, x, indexing="ij")
    points = np.column_stack([points_y.ravel(), points_x.ravel()])
    u = RegularGridInterpolator(
        (velocity["y"], velocity["x"]), velocity["vx"],
        bounds_error=False, fill_value=np.nan)(points).reshape(bed.shape)
    v = RegularGridInterpolator(
        (velocity["y"], velocity["x"]), velocity["vy"],
        bounds_error=False, fill_value=np.nan)(points).reshape(bed.shape)
    # A geometria de instabilidade é de grande escala. Um gradiente calculado
    # diretamente a 500 m amplifica o erro célula a célula do BedMachine.
    # Suavização gaussiana σ=2 km antes da derivada remove essa textura sem
    # preencher lacunas nem alterar a máscara marinha.
    sigma_px = 2_000.0 / max(abs(np.median(np.diff(x))), 1.0)
    finite = np.isfinite(bed)
    weight = gaussian_filter(finite.astype(float), sigma=sigma_px)
    smooth_bed = np.divide(
        gaussian_filter(np.where(finite, bed, 0.0), sigma=sigma_px), weight,
        out=np.full_like(bed, np.nan), where=weight > 0.05)
    slope = along_flow_bed_slope(smooth_bed, x, y, u, v)
    grounded = mask == 2
    dy = float(np.median(np.diff(y)))
    dx = float(np.median(np.diff(x)))
    distance_gl_km = distance_transform_edt(
        grounded, sampling=(abs(dy), abs(dx))) / 1000.0
    marine_gz = grounded & (bed < 0) & (distance_gl_km <= 100.0)
    slope[~marine_gz] = np.nan
    return {
        "x": x, "y": y, "bed": bed, "bed_smoothed": smooth_bed,
        "mask": mask, "errbed": errbed,
        "along_flow_slope_m_km": slope, "distance_gl_km": distance_gl_km,
        "marine_gz": marine_gz,
    }


def _diagnostic_map(cfg, basal, thinning, velocity, geometry, extent, path):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.5), facecolor="white",
                             constrained_layout=True)
    for ax in axes.ravel():
        _base(ax, cfg, extent, target_px=650)

    ax = axes[0, 0]
    limit = _limits([basal], "basal_melt")
    art = ax.scatter(basal.x / 1000, basal.y / 1000, c=basal.basal_melt,
                     s=34, marker="s", cmap="RdBu_r",
                     norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                     edgecolor="none", zorder=8, rasterized=True)
    fig.colorbar(art, ax=ax, shrink=0.78, extend="both", pad=0.02,
                 label="m_b (m gelo/ano)")
    ax.set_title("a. Perda do suporte flutuante\nderretimento basal JJA+DJF",
                 loc="left", fontweight="bold")

    ax = axes[0, 1]
    velocity_qc = ((velocity["speed"] >= 100)
                   & (velocity["count"] == len(velocity["years"]))
                   & (velocity["trend_r2"] >= 0.50))
    vt = np.where(velocity_qc, velocity["trend_percent"], np.nan)
    vlim = float(max(np.nanquantile(np.abs(vt), 0.97), 0.5))
    art = ax.pcolormesh(
        velocity["x"] / 1000, velocity["y"] / 1000, vt,
        cmap="BrBG_r", norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim),
        shading="auto", alpha=0.84, zorder=7, rasterized=True)
    fig.colorbar(art, ax=ax, shrink=0.78, extend="both", pad=0.02,
                 label="tendência da velocidade (%/ano)")
    ax.set_title("b. Resposta dinâmica\nITS_LIVE 2019–2025; 7 anos, R²≥0,50, v≥100 m/ano",
                 loc="left", fontweight="bold")

    ax = axes[1, 0]
    nonsig = thinning[~thinning.significant_both]
    sig = thinning[thinning.significant_both]
    ax.scatter(nonsig.x / 1000, nonsig.y / 1000, s=4, color="#999999",
               alpha=0.22, edgecolor="none", zorder=7, rasterized=True)
    if len(sig):
        tlim = float(max(abs(np.nanquantile(sig.dhdt_consensus, 0.02)), 0.5))
        art = ax.scatter(sig.x / 1000, sig.y / 1000, c=sig.dhdt_consensus,
                         s=10, cmap="magma_r", vmin=-tlim, vmax=0,
                         edgecolor="none", zorder=8, rasterized=True)
        fig.colorbar(art, ax=ax, shrink=0.78, extend="min", pad=0.02,
                     label="dh/dt consensual (m/ano)")
    ax.set_title("c. Adelgaçamento persistente no gelo aterrado\n"
                 "IC95% abaixo de zero em JJA e DJF",
                 loc="left", fontweight="bold")

    ax = axes[1, 1]
    slope = geometry["along_flow_slope_m_km"]
    slim = float(max(np.nanquantile(np.abs(slope), 0.97), 1.0))
    art = ax.pcolormesh(
        geometry["x"] / 1000, geometry["y"] / 1000, slope,
        cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-slim, vcenter=0, vmax=slim),
        shading="auto", alpha=0.85, zorder=7, rasterized=True)
    fig.colorbar(art, ax=ax, shrink=0.78, extend="both", pad=0.02,
                 label="declive do leito no fluxo (m/km)")
    ax.set_title("d. Geometria marinha até 100 km da linha de aterramento\n"
                 "positivo = leito retrógrado (aprofunda para o interior)",
                 loc="left", fontweight="bold")

    for ax in axes.ravel():
        _finish_map(ax, extent, scale_km=50)
    fig.suptitle(
        "Diagnóstico observacional de vulnerabilidade — Geleira Thwaites",
        fontsize=15, fontweight="bold")
    fig.text(
        0.5, -0.018,
        "As quatro camadas são evidências separadas. O mapa NÃO fornece "
        "probabilidade, data ou trajetória futura de colapso.",
        ha="center", color="#8b0000", fontsize=9, fontweight="bold")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _safe_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "median": None, "p10": None, "p90": None}
    return {
        "n": int(len(values)), "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Mapas de derretimento basal e vulnerabilidade da Thwaites.")
    parser.add_argument("--cell-km", type=float, default=5.0)
    parser.add_argument("--front-epoch", type=float, default=2022.5)
    args = parser.parse_args()

    cfg = load_config("jja")
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="basal_diagnostic_maps")
    cfg_jja, cfg_djf = load_config("jja"), load_config("djf")
    melt_jja = pd.read_parquet(cfg_jja.paths.dhdt_dir / "shelf_basal_melt.parquet")
    melt_djf = pd.read_parquet(cfg_djf.paths.dhdt_dir / "shelf_basal_melt.parquet")
    cell_size = args.cell_km * 1000.0
    cells_jja = aggregate_basal_cells(melt_jja, cell_size)
    cells_djf = aggregate_basal_cells(melt_djf, cell_size)
    contrast = seasonal_basal_contrast(melt_jja, melt_djf, cell_size)
    combined = aggregate_basal_cells(pd.concat([melt_jja, melt_djf]), cell_size)
    if min(len(cells_jja), len(cells_djf), len(contrast)) == 0:
        raise RuntimeError("Cobertura insuficiente para mapas sazonais da Thwaites")

    all_x = np.r_[cells_jja.x, cells_djf.x]
    all_y = np.r_[cells_jja.y, cells_djf.y]
    basal_extent = (float(all_x.min() - 30e3), float(all_x.max() + 30e3),
                    float(all_y.min() - 30e3), float(all_y.max() + 30e3))
    diagnostic_extent = (
        float(all_x.min() - 60e3), float(all_x.max() + 90e3),
        float(all_y.min() - 140e3), float(all_y.max() + 90e3),
    )
    output = ROOT / "outputs" / "diagnostico"
    output.mkdir(parents=True, exist_ok=True)
    pooled_limit = _limits([cells_jja, cells_djf], "basal_melt")
    _plot_basal_single(cells_jja, cfg, basal_extent, pooled_limit, "JJA",
                       output / "mapa_derretimento_basal_jja.png")
    _plot_basal_single(cells_djf, cfg, basal_extent, pooled_limit, "DJF",
                       output / "mapa_derretimento_basal_djf.png")
    _plot_basal_comparison(
        cells_jja, cells_djf, contrast, cfg, basal_extent,
        output / "mapa_derretimento_basal_sazonal.png")

    nodes_jja = pd.read_parquet(cfg_jja.paths.dhdt_dir / "dhdt_nodes_qc.parquet")
    nodes_djf = pd.read_parquet(cfg_djf.paths.dhdt_dir / "dhdt_nodes_qc.parquet")
    thinning = consensus_thinning(nodes_jja, nodes_djf)
    thinning = thinning[
        thinning.x.between(diagnostic_extent[0], diagnostic_extent[1])
        & thinning.y.between(diagnostic_extent[2], diagnostic_extent[3])]

    velocity_path = cfg.paths.data_dir / "velocity_itslive_annual.nc"
    velocity = _load_velocity(velocity_path, diagnostic_extent)
    bed_path = sorted(cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))[0]
    geometry = _load_bed_geometry(bed_path, diagnostic_extent, velocity)
    _diagnostic_map(
        cfg, combined, thinning, velocity, geometry, diagnostic_extent,
        output / "mapa_diagnostico_vulnerabilidade.png")

    valid_velocity = ((velocity["speed"] >= 100)
                      & (velocity["count"] == len(velocity["years"]))
                      & (velocity["trend_r2"] >= 0.50)
                      & np.isfinite(velocity["trend_percent"]))
    valid_slope = np.isfinite(geometry["along_flow_slope_m_km"])
    report = {
        "status": "DIAGNOSTICO_OBSERVACIONAL_NAO_PROGNOSTICO",
        "title": "Sinais observados e vulnerabilidade estrutural da Thwaites",
        "epsg": 3031,
        "period": "2019-2025",
        "inputs": {
            "altimetry": "ICESat-2 ATL06 v007, JJA e DJF",
            "basal_melt": "m_b = a_s - DH/Dt - H*div(v), Lagrangiano",
            "velocity": "ITS_LIVE anual 2019-2025",
            "geometry": "BedMachine Antarctica v4.1, epoca nominal ~2015",
            "fronts": "IceLines/Sentinel-1, frente mais proxima de 2022.5",
        },
        "basal_melt_cells_5km": {
            "jja": _safe_stats(cells_jja.basal_melt),
            "djf": _safe_stats(cells_djf.basal_melt),
            "djf_minus_jja_common_cells": _safe_stats(
                contrast.basal_melt_djf_minus_jja),
            "statistical_sigma_is_lower_bound_only": True,
        },
        "grounded_thinning_consensus": {
            "n_common_nodes_in_map": int(len(thinning)),
            "n_significant_both_seasons": int(thinning.significant_both.sum()),
            "fraction_significant_both": float(thinning.significant_both.mean()),
            "dhdt_significant_nodes": _safe_stats(
                thinning.loc[thinning.significant_both, "dhdt_consensus"]),
        },
        "velocity_trend_percent_per_year": {
            "selection": "median speed >=100 m/yr, 7 annual epochs and OLS R2>=0.50",
            **_safe_stats(velocity["trend_percent"][valid_velocity]),
            "fraction_positive": float(np.mean(
                velocity["trend_percent"][valid_velocity] > 0)),
        },
        "marine_bed_geometry_within_100km_of_grounding_line": {
            "bed_smoothing_before_gradient": "Gaussian sigma=2 km",
            "along_flow_slope_m_per_km": _safe_stats(
                geometry["along_flow_slope_m_km"][valid_slope]),
            "fraction_retrograde_positive": float(np.mean(
                geometry["along_flow_slope_m_km"][valid_slope] > 0)),
            "bed_error_m": _safe_stats(geometry["errbed"][valid_slope]),
        },
        "interpretation": (
            "Coexistencia de perda do suporte flutuante, mudanca da velocidade, "
            "adelgacamento aterrado e leito retrogrado indica vulnerabilidade. "
            "Nao determina se, quando ou por qual trajetoria ocorrera colapso."),
        "collapse_prediction_available": False,
        "missing_for_collapse_forecast": [
            "definicao operacional de colapso e horizonte de previsao",
            "serie temporal validada da linha de aterramento e taxa de recuo",
            "batimetria e geometria da cavidade subglacial com incerteza menor",
            "temperatura, salinidade, correntes e turbulencia oceanicas em 3-D",
            "espessura/draft contemporaneos, dano, fraturas e buttressing",
            "tracao basal e reologia do gelo calibradas por inversao",
            "SMB e firn completos ate 2025; o GSFC-FDM atual termina em 2022",
            "propagacao conjunta das incertezas e covariancias",
            "modelo acoplado gelo-oceano calibrado e ensemble de cenarios",
            "validacao externa com InSAR/GNSS, ApRES, radar e observacoes oceanicas",
        ],
    }
    report_path = output / "diagnostico_vulnerabilidade.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    for path in sorted(output.glob("mapa_*.png")):
        log.info(f"mapa -> {path}")
    log.info(f"relatorio -> {report_path}")


if __name__ == "__main__":
    main()
