"""Teste A/B de representação espacial do dh/dt JJA no ASE.

A: nós fitsec de 5 km, sem interpolação adicional; ausências permanecem NaN.
B: os mesmos nós, interpolados por IDW escolhido por validação espacial em
   blocos e limitado a 30 km, sempre dentro do gelo aterrado BedMachine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.viz.produtos import para_grade
from pipelines.run_ase_jja_atl06_continuous import idw
from pipelines.run_ase_jja_basal_dhdt_products_500m import (
    CMAP,
    DHDT_LEVELS,
    DPI,
    KM,
    add_dhdt_colorbar,
    draw_base,
    draw_surface,
    load_base_assets,
)
from pipelines.run_ase_jja_diagnostic_maps import inside_roi, roi_polygon


BLOCK_M = 50_000.0
BUFFER_M = 15_000.0
MAX_SUPPORT_M = 30_000.0
N_FOLDS = 5
CANDIDATES = tuple(
    (k, power) for k in (4, 8, 16, 32) for power in (1.0, 2.0, 3.0))


def load_nodes(cfg, roi_path):
    nodes = pd.read_parquet(
        cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
        columns=["x", "y", "dhdt", "mask_class", "reliability"])
    keep = inside_roi(roi_path, nodes.x.to_numpy(), nodes.y.to_numpy())
    finite = np.isfinite(nodes.x) & np.isfinite(nodes.y) & np.isfinite(nodes.dhdt)
    return nodes[keep & finite & nodes.mask_class.eq(2)].copy()


def spatial_folds(nodes):
    bx = np.floor(nodes.x.to_numpy() / BLOCK_M).astype(np.int64)
    by = np.floor(nodes.y.to_numpy() / BLOCK_M).astype(np.int64)
    blocks = list(zip(bx, by))
    unique = sorted(set(blocks))
    rng = np.random.default_rng(0)
    shuffled = np.asarray(unique, dtype=object)[rng.permutation(len(unique))]
    mapping = {tuple(block): index % N_FOLDS
               for index, block in enumerate(shuffled)}
    return np.asarray([mapping[block] for block in blocks], dtype=int)


def select_idw(nodes):
    """Seleciona k/p por CV em blocos, com buffer igual ao raio fitsec."""
    folds = spatial_folds(nodes)
    rows = []
    for k, power in CANDIDATES:
        observed, predicted, nearest_distance = [], [], []
        for fold in range(N_FOLDS):
            train = nodes[folds != fold].copy()
            test = nodes[folds == fold].copy()
            if len(train) < k or not len(test):
                continue
            # Nós próximos do bloco teste compartilham observações ATL06 devido
            # ao raio fitsec; o buffer reduz esse vazamento espacial.
            distance_to_test = cKDTree(
                test[["x", "y"]].to_numpy()).query(
                    train[["x", "y"]].to_numpy(), k=1)[0]
            train = train[distance_to_test > BUFFER_M]
            if len(train) < k:
                continue
            query = test[["x", "y"]].to_numpy()
            pred = idw(
                train[["x", "y"]].to_numpy(), train.dhdt.to_numpy(), query,
                k=k, power=power)
            distance = cKDTree(
                train[["x", "y"]].to_numpy()).query(query, k=1)[0]
            supported = distance <= MAX_SUPPORT_M
            observed.extend(test.dhdt.to_numpy()[supported])
            predicted.extend(pred[supported])
            nearest_distance.extend(distance[supported])
        observed = np.asarray(observed, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        residual = predicted - observed
        rows.append({
            "k": int(k),
            "power": float(power),
            "n": int(len(residual)),
            "coverage_fraction": float(len(residual) / len(nodes)),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
            "median_nearest_train_km": float(
                np.median(nearest_distance) / 1000),
        })
    table = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    return table.iloc[0].to_dict(), table


def interpolate_native(nodes, assets, winner):
    """Interpola na máscara visual de 500 m sem exceder suporte de 30 km."""
    surface = np.full(assets["high_mask"].shape, np.nan, dtype=np.float32)
    domain = ((assets["high_mask"] == 2) & assets["native_roi"])
    query = np.column_stack((assets["hxx"][domain], assets["hyy"][domain]))
    source_xy = nodes[["x", "y"]].to_numpy()
    prediction = idw(
        source_xy, nodes.dhdt.to_numpy(), query,
        k=int(winner["k"]), power=float(winner["power"]))
    distance = cKDTree(source_xy).query(query, k=1)[0]
    prediction[distance > MAX_SUPPORT_M] = np.nan
    surface[domain] = prediction.astype(np.float32)
    return surface, domain, distance


def common_figure(cfg, assets, roi_x, roi_y, extent):
    fig, ax = plt.subplots(figsize=(10.8, 8.8), constrained_layout=True)
    draw_base(ax, cfg, assets, roi_x, roi_y, extent)
    return fig, ax


def plot_no_interpolation(cfg, assets, roi_x, roi_y, extent, nodes, norm, output):
    fig, ax = common_figure(cfg, assets, roi_x, roi_y, extent)
    x, y, field = para_grade(nodes, "dhdt", res=5000.0)
    artist = ax.pcolormesh(
        x * KM, y * KM, np.ma.masked_invalid(field),
        cmap=CMAP, norm=norm, shading="nearest", zorder=7,
        rasterized=True)
    add_dhdt_colorbar(fig, artist, ax=ax)
    ax.set_title(
        "Teste A · dh/dt fitsec nos nós originais de 5 km\n"
        "sem interpolação adicional · células ausentes permanecem sem dado",
        loc="left", fontweight="bold")
    path = output / "teste_A_dhdt_5km_sem_interpolacao.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path, int(np.isfinite(field).sum()), int(field.size)


def plot_idw(cfg, assets, roi_x, roi_y, extent, surface, norm, winner, output):
    fig, ax = common_figure(cfg, assets, roi_x, roi_y, extent)
    artist = draw_surface(ax, surface, assets, norm, zorder=7)
    add_dhdt_colorbar(fig, artist, ax=ax)
    ax.set_title(
        "Teste B · dh/dt fitsec + interpolação IDW no gelo aterrado\n"
        f"k={int(winner['k'])} · potência={winner['power']:.0f} · "
        "suporte máximo=30 km · máscara visual de 500 m",
        loc="left", fontweight="bold")
    path = output / "teste_B_dhdt_idw_30km_mascara_500m.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    cfg = load_config("jja")
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="ase_jja_dhdt_interpolation_tests")
    output = ROOT / "outputs" / "diagnostico_ase_jja" / "testes_dhdt"
    output.mkdir(parents=True, exist_ok=True)
    roi_x, roi_y, roi_path, extent = roi_polygon()
    assets = load_base_assets(cfg, extent, roi_path)
    nodes = load_nodes(cfg, roi_path)
    log.info(f"nós fitsec/QC dentro da ROI: {len(nodes):,}")

    winner, cv = select_idw(nodes)
    log.info(f"CV IDW:\n{cv.to_string(index=False)}")
    log.info(
        f"vencedor k={int(winner['k'])}, p={winner['power']:.0f}, "
        f"RMSE={winner['rmse']:.3f} m/ano")
    surface, domain, _ = interpolate_native(nodes, assets, winner)
    norm = BoundaryNorm(DHDT_LEVELS, plt.get_cmap(CMAP).N, clip=False)

    path_a, cells_a, grid_cells_a = plot_no_interpolation(
        cfg, assets, roi_x, roi_y, extent, nodes, norm, output)
    path_b = plot_idw(
        cfg, assets, roi_x, roi_y, extent, surface, norm, winner, output)

    report = {
        "source_method": {
            "name": "fitsec/regressão espaço-temporal local ponderada",
            "node_spacing_km": cfg.dhdt.node_spacing_m / 1000,
            "search_radius_km": cfg.dhdt.search_radius_m / 1000,
            "minimum_points": cfg.dhdt.min_points,
            "spatial_polynomial_order": cfg.dhdt.poly_order,
            "temporal_polynomial_order": cfg.dhdt.temp_order,
            "weighted_by_elevation_uncertainty": cfg.dhdt.use_weights,
            "robust_iterations": cfg.dhdt.max_iter,
        },
        "input_nodes": int(len(nodes)),
        "test_A": {
            "method": "sem interpolação espacial adicional; para_grade + pcolormesh",
            "finite_5km_cells": cells_a,
            "rectangular_grid_cells": grid_cells_a,
            "output": str(path_a),
        },
        "test_B": {
            "method": "IDW após fitsec, restrito ao gelo aterrado",
            "selection": "CV espacial 5 folds, blocos 50 km, buffer 15 km",
            "winner": winner,
            "candidates": cv.to_dict(orient="records"),
            "maximum_support_km": MAX_SUPPORT_M / 1000,
            "native_grounded_pixels": int(domain.sum()),
            "mapped_native_pixels": int(np.isfinite(surface).sum()),
            "visual_mask_resolution_m": 500,
            "effective_source_node_spacing_m": cfg.dhdt.node_spacing_m,
            "output": str(path_b),
        },
        "warning": (
            "A máscara de 500 m melhora apenas o recorte cartográfico; não "
            "transforma o dh/dt em observação de 500 m."),
    }
    report_path = output / "teste_interpolacao_dhdt_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"teste A -> {path_a}")
    log.info(f"teste B -> {path_b}")
    log.info(f"relatório -> {report_path}")


if __name__ == "__main__":
    main()
