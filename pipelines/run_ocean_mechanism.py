"""Diagnóstico observacional do mecanismo oceânico junto à TEIS.

Combina a série BAS/ITGC MELT com o produto Lagrangiano de derretimento basal.
Como o dado oceânico é de um único ponto, a colocalização espacial é reportada
explicitamente e não é tratada como validação de toda a plataforma.
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

from thwaites import load_config  # noqa: E402
from thwaites.logging import setup_logging  # noqa: E402
from thwaites.ocean.bas_melt import (  # noqa: E402
    BAS_MELT_DOI,
    BAS_MELT_LAT,
    BAS_MELT_LON,
    BAS_MELT_PRESSURE_DBAR,
    load_bas_melt,
    summarize_ocean_forcing,
)


def _collocate_basal_melt(parcels: pd.DataFrame, site_x: float, site_y: float,
                          target_year: float, radii_km=(5.0, 10.0, 20.0, 40.0)) -> dict:
    parcels = parcels.copy()
    shelf_filter = None
    if "shelf" in parcels.columns:
        shelf_filter = "Thwaites"
        parcels = parcels[
            parcels["shelf"].astype(str).str.contains(
                shelf_filter, case=False, regex=False, na=False)]
    if "reliable" in parcels:
        parcels = parcels[parcels["reliable"].astype(bool)]
    parcels = parcels[np.isfinite(parcels["basal_melt"])]
    if parcels.empty:
        return {"status": "sem parcelas confiáveis"}
    if "t_center" in parcels:
        centers = np.sort(parcels["t_center"].dropna().unique())
        chosen = float(centers[np.argmin(np.abs(centers - target_year))])
        parcels = parcels[np.isclose(parcels["t_center"], chosen)]
    else:
        chosen = None
    distance_km = np.hypot(parcels["x_ref"] - site_x,
                           parcels["y_ref"] - site_y) / 1000.0
    parcels = parcels.assign(distance_to_bas_site_km=distance_km)
    nearest = parcels.nsmallest(min(20, len(parcels)), "distance_to_bas_site_km")
    result = {
        "shelf_filter": shelf_filter,
        "selected_t_center": chosen,
        "n_selected_window": int(len(parcels)),
        "nearest_distance_km": float(distance_km.min()),
        "nearest_20_basal_melt_m_yr_median": float(nearest["basal_melt"].median()),
        "by_radius": {},
    }
    for radius in radii_km:
        local = parcels[parcels["distance_to_bas_site_km"] <= radius]
        result["by_radius"][f"{radius:g}km"] = {
            "n": int(len(local)),
            "basal_melt_m_yr_median": (
                float(local["basal_melt"].median()) if len(local) else None),
            "basal_melt_m_yr_p10": (
                float(local["basal_melt"].quantile(0.10)) if len(local) else None),
            "basal_melt_m_yr_p90": (
                float(local["basal_melt"].quantile(0.90)) if len(local) else None),
        }
    return result


def _make_figures(data: pd.DataFrame, figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    figure_dir.mkdir(parents=True, exist_ok=True)
    monthly = data.set_index("time").resample("MS").median(numeric_only=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(monthly.index, monthly["thermal_driving_c"], marker="o", color="#b2182b")
    axes[0].set_ylabel("Forçamento térmico (°C)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(monthly.index, monthly["speed_cm_s"], marker="o", color="#2166ac")
    axes[1].set_ylabel("Velocidade horizontal (cm/s)")
    axes[1].set_xlabel("Mês")
    axes[1].grid(alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Forçamento oceânico observado — BAS/ITGC MELT, TEIS")
    time_path = figure_dir / "ocean_melt_timeseries.png"
    fig.savefig(time_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    color = mdates.date2num(data["time"].dt.to_pydatetime())
    scatter = ax.scatter(data["thermal_driving_c"], data["speed_cm_s"],
                         c=color, s=8, alpha=0.35, cmap="viridis", rasterized=True)
    ax.set_xlabel("Forçamento térmico (°C)")
    ax.set_ylabel("Velocidade horizontal (cm/s)")
    ax.grid(alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    colorbar.set_label("Data")
    ax.set_title("Calor disponível × circulação horizontal")
    scatter_path = figure_dir / "ocean_melt_speed_vs_thermal.png"
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)
    return [time_path, scatter_path]


def _make_basal_site_figure(parcels: pd.DataFrame, site_x: float, site_y: float,
                            target_year: float, figure_dir: Path,
                            season_name: str) -> Path | None:
    """Mapa do produto 5 restrito a Thwaites e à janela oceânica mais próxima."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Circle

    selected = parcels.copy()
    if "shelf" in selected.columns:
        selected = selected[selected["shelf"].astype(str).str.contains(
            "Thwaites", case=False, regex=False, na=False)]
    if "reliable" in selected.columns:
        selected = selected[selected["reliable"].astype(bool)]
    selected = selected[np.isfinite(selected["basal_melt"])]
    if selected.empty:
        return None
    chosen = None
    if "t_center" in selected.columns:
        centers = np.sort(selected["t_center"].dropna().unique())
        chosen = float(centers[np.argmin(np.abs(centers - target_year))])
        selected = selected[np.isclose(selected["t_center"], chosen)]
    values = selected["basal_melt"].to_numpy(dtype=float)
    limit = float(max(abs(np.nanquantile(values, 0.05)),
                      abs(np.nanquantile(values, 0.95)), 1.0))
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    points = ax.scatter(
        selected["x_ref"] / 1000.0, selected["y_ref"] / 1000.0,
        c=values, s=28, cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        edgecolors="none", alpha=0.85,
    )
    sx, sy = site_x / 1000.0, site_y / 1000.0
    ax.scatter([sx], [sy], marker="*", s=250, c="#ffd92f", edgecolor="black",
               linewidth=0.9, zorder=5, label="BAS MELT")
    for radius, linestyle in ((10, "-"), (20, "--"), (40, ":")):
        ax.add_patch(Circle((sx, sy), radius, fill=False, color="black",
                            lw=0.8, ls=linestyle, alpha=0.65))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("EPSG:3031 x (km)")
    ax.set_ylabel("EPSG:3031 y (km)")
    window = f"t centro {chosen:.2f}" if chosen is not None else "todas as janelas"
    ax.set_title(f"Derretimento basal Lagrangiano — Thwaites {season_name.upper()}\n{window}")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    cbar = fig.colorbar(points, ax=ax)
    cbar.set_label("m_b (m gelo/ano; positivo = derretimento)")
    path = figure_dir / "basal_melt_thwaites_ocean_site.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Mecanismo oceânico observado na TEIS.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--ocean-file", default=(
        "ocean/bas_melt_01468/Thwaites_MAVS_Timeseries_TSV.dat"))
    parser.add_argument("--basal-file", default="shelf_basal_melt.parquet")
    parser.add_argument("--pressure-dbar", type=float, default=BAS_MELT_PRESSURE_DBAR)
    args = parser.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="ocean_mechanism")
    ocean_path = cfg.paths.data_dir / args.ocean_file
    basal_path = cfg.paths.dhdt_dir / args.basal_file
    data = load_bas_melt(ocean_path, pressure_dbar=args.pressure_dbar)
    processed = ocean_path.with_name("bas_melt_processed.parquet")
    data.to_parquet(processed, index=False)
    report = {
        "status": "DIAGNOSTICO_OBSERVACIONAL_DE_UM_PONTO",
        "season_profile": cfg.season.name,
        "source": {
            "dataset": "BAS/ITGC MELT 01468",
            "doi": BAS_MELT_DOI,
            "latitude": BAS_MELT_LAT,
            "longitude": BAS_MELT_LON,
            "pressure_dbar": args.pressure_dbar,
            "note": ("a coluna Conductivity esta rotulada como PSU no TSV, mas os "
                     "metadados declaram precisao em S/m; interpretada como "
                     "mS/cm e convertida para salinidade pratica por PSS-78"),
        },
        "ocean_forcing": summarize_ocean_forcing(data),
    }
    forcing = report["ocean_forcing"]
    max_tidal_r2 = max(
        item["r2"]
        for variable in forcing["harmonics"].values()
        for item in variable.values()
    )
    report["mechanism_assessment"] = {
        "classification": "CALOR_DISPONIVEL_COM_TRANSFERENCIA_VERTICAL_LIMITADA",
        "evidence_from_this_series": {
            "thermal_driving_median_c": forcing["thermal_driving_c"]["median"],
            "horizontal_speed_median_cm_s": forcing["speed_cm_s"]["median"],
            "maximum_individual_tidal_harmonic_r2": float(max_tidal_r2),
            "speed_vs_thermal_driving_spearman_rho": (
                forcing["speed_vs_thermal_driving_spearman"]["rho"]),
        },
        "interpretation": (
            "A serie sustenta agua termicamente capaz de derreter gelo e "
            "circulacao horizontal fraca. E consistente com limitacao da entrega "
            "vertical de calor, mas nao mede estratificacao nem turbulencia vertical."),
        "literature_context": {
            "doi": "10.1038/s41586-022-05586-0",
            "mechanism": ("na TEIS, baixa velocidade e forte estratificacao "
                          "restringem a mistura vertical e suprimem o derretimento"),
        },
        "causality_status": "CONSISTENTE_MAS_NAO_IDENTIFICADA_INDEPENDENTEMENTE",
    }

    basal_figure = None
    if basal_path.exists():
        from pyproj import Transformer
        transformer = Transformer.from_crs(4326, cfg.area.epsg_polar, always_xy=True)
        site_x, site_y = transformer.transform(BAS_MELT_LON, BAS_MELT_LAT)
        parcels = pd.read_parquet(basal_path)
        mid_year = data["time"].dt.year.add(
            (data["time"].dt.dayofyear - 1) / 365.25).median()
        report["basal_melt_collocation"] = _collocate_basal_melt(
            parcels, site_x, site_y, float(mid_year))
        basal_figure = _make_basal_site_figure(
            parcels, site_x, site_y, float(mid_year), cfg.paths.figures,
            cfg.season.name)
    else:
        report["basal_melt_collocation"] = {
            "status": f"produto basal ausente: {basal_path}"}

    report["interpretation_limits"] = [
        "a colocalizacao basal inclui apenas parcelas cujo rotulo de plataforma contem Thwaites",
        "a série oceânica representa um único ponto da TEIS e não toda a plataforma",
        "o produto basal usa janelas plurianuais; não há resolução temporal compatível para correlação evento a evento",
        "velocidade horizontal × forçamento térmico é proxy advectivo, não fluxo turbulento de calor",
        "sem turbulência vertical, geometria 3-D e experimento oceânico não se atribui causalidade",
        "o p-valor da correlação não corrige autocorrelação temporal",
    ]
    report["mechanism_tests"] = {
        "lateral_advection": "diagnosticável por velocidade horizontal, direção e proxy U×(T−Tf)",
        "tidal_modulation": "diagnóstico harmônico M2/S2/K1/O1; não equivale a fluxo vertical",
        "vertical_turbulent_heat_flux": "não identificável neste arquivo",
        "geometry_feedback": "requer batimetria/draft espacial e, para causalidade, MITgcm",
    }
    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.tables / "ocean_mechanism_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    figures = _make_figures(data, cfg.paths.figures)
    if basal_figure is not None:
        figures.append(basal_figure)
    log.info(f"BAS MELT processado -> {processed} ({len(data):,} registros)")
    log.info(f"Relatório -> {report_path}")
    for figure in figures:
        log.info(f"Figura -> {figure}")


if __name__ == "__main__":
    main()
