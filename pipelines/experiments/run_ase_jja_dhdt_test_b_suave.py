"""Versão suavizada e em alta resolução cartográfica do teste B de dh/dt.

O campo continua derivado dos nós fitsec de 5 km. A suavização usa IDW com
oito vizinhos e potência dois, limitada a 30 km e ao gelo aterrado BedMachine.
A exportação a 360 dpi aumenta a resolução da figura, não a resolução
física da estimativa.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from pipelines.run_ase_jja_atl06_continuous import idw
from pipelines.run_ase_jja_basal_dhdt_products_500m import (
    DHDT_LEVELS,
    KM,
    add_dhdt_colorbar,
    draw_base,
    load_base_assets,
)
from pipelines.run_ase_jja_diagnostic_maps import roi_polygon
from pipelines.run_ase_jja_dhdt_interpolation_tests import load_nodes


CMAP = "RdBu"
IDW_K = 8
IDW_POWER = 2.0
MAX_SUPPORT_M = 30_000.0
DPI = 360


def interpolate_smooth(nodes, assets):
    """IDW suavizado na máscara nativa, sem extrapolar além de 30 km."""
    surface = np.full(assets["high_mask"].shape, np.nan, dtype=np.float32)
    domain = (assets["high_mask"] == 2) & assets["native_roi"]
    query = np.column_stack((assets["hxx"][domain], assets["hyy"][domain]))
    source_xy = nodes[["x", "y"]].to_numpy(dtype=float)
    source_z = nodes.dhdt.to_numpy(dtype=float)
    prediction = idw(
        source_xy, source_z, query, k=IDW_K, power=IDW_POWER)
    nearest = cKDTree(source_xy).query(query, k=1)[0]
    prediction[nearest > MAX_SUPPORT_M] = np.nan
    surface[domain] = prediction.astype(np.float32)
    return surface, domain


def plot(cfg, assets, roi_x, roi_y, extent, surface, output):
    fig, ax = plt.subplots(figsize=(12.0, 9.8), constrained_layout=True)
    draw_base(ax, cfg, assets, roi_x, roi_y, extent)
    # Escala realmente contínua, preservando zero como centro perceptual.
    norm = TwoSlopeNorm(
        vmin=float(DHDT_LEVELS.min()),
        vcenter=0.0,
        vmax=float(DHDT_LEVELS.max()),
    )
    artist = ax.imshow(
        np.ma.masked_invalid(surface),
        extent=assets["image_extent"],
        origin=assets["image_origin"],
        interpolation="bilinear",
        cmap=CMAP,
        norm=norm,
        zorder=7,
        rasterized=True,
    )
    add_dhdt_colorbar(fig, artist, ax=ax)
    ax.set_title(
        "dh/dt JJA · interpolação IDW suavizada no gelo aterrado\n"
        "8 vizinhos · potência 2 · suporte máximo 30 km · escala contínua",
        loc="left",
        fontweight="bold",
    )
    path = output / "teste_B3_dhdt_idw_suave_continuo_360dpi.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    cfg = load_config("jja")
    log = setup_logging(
        cfg.paths.logs,
        level=cfg.logging.level,
        run_name="ase_jja_dhdt_test_b_suave",
    )
    output = ROOT / "outputs" / "diagnostico_ase_jja" / "testes_dhdt"
    output.mkdir(parents=True, exist_ok=True)
    roi_x, roi_y, roi_path, extent = roi_polygon()
    assets = load_base_assets(cfg, extent, roi_path, target_px=1400)
    nodes = load_nodes(cfg, roi_path)
    surface, domain = interpolate_smooth(nodes, assets)
    path = plot(cfg, assets, roi_x, roi_y, extent, surface, output)

    report = {
        "source": "nós fitsec/QC JJA de 5 km",
        "input_nodes": int(len(nodes)),
        "interpolation": {
            "method": "IDW",
            "k": IDW_K,
            "power": IDW_POWER,
            "maximum_support_km": MAX_SUPPORT_M / 1000.0,
            "cv_rmse_m_per_year": 0.37049853547210215,
            "cv_mae_m_per_year": 0.21706573487404604,
            "cv_bias_m_per_year": 0.02871505399401294,
        },
        "rendering": {
            "bedmachine_mask_m": 500,
            "export_dpi": DPI,
            "colormap": CMAP,
            "normalization": "TwoSlopeNorm(-3, 0, 1), contínua",
            "image_interpolation": "bilinear",
            "mapped_grounded_pixels": int(np.isfinite(surface).sum()),
            "grounded_domain_pixels": int(domain.sum()),
        },
        "effective_source_spacing_m": cfg.dhdt.node_spacing_m,
        "warning": (
            "A suavização e 360 dpi aumentam a continuidade e a resolução "
            "cartográfica, não a resolução física dos dados de dh/dt."
        ),
        "output": str(path),
    }
    report_path = output / "teste_B3_dhdt_idw_suave_continuo_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"mapa suave -> {path}")
    log.info(f"relatório -> {report_path}")


if __name__ == "__main__":
    main()
