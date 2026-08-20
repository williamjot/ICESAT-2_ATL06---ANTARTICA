"""Campo contínuo exploratório de derretimento basal JJA no ASE.

Este produto NÃO replica o ajuste espaço-temporal de Meng et al. (2025): ele
interpola as estimativas lagrangianas de ``shelf_basal_melt.parquet`` já
derivadas de ATL06 v007. O objetivo é preencher a plataforma sem esconder onde
o mapa é observado, interpolado ou apenas fracamente sustentado.

Saídas
------
outputs/diagnostico_ase_jja/
  mapa_derretimento_basal_atl06_jja_continuo.png
  mapa_suporte_atl06_jja_continuo.png
  basal_melt_atl06_jja_continuous.parquet
  basal_melt_atl06_jja_continuous_report.json
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
from matplotlib.colors import BoundaryNorm, Normalize
from netCDF4 import Dataset
from scipy.spatial import cKDTree

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.viz.basemap import draw_basemap, draw_calving_fronts, add_scale_bar
from run_ase_jja_diagnostic_maps import inside_roi, roi_polygon


CELL_M = 5_000.0
BLOCK_M = 50_000.0
N_FOLDS = 5
MAX_SUPPORT_M = 75_000.0
HIGH_SUPPORT_M = 30_000.0
DPI = 220
BASAL_CMAP = "RdBu_r"
BASAL_LEVELS = np.arange(-60.0, 61.0, 10.0)


def aggregate_cells(records: pd.DataFrame) -> pd.DataFrame:
    """Mediana por célula, preservando plataforma e suporte temporal."""
    d = records.copy()
    if "reliable" in d:
        d = d[d.reliable.astype(bool)]
    finite = np.isfinite(d.x_ref) & np.isfinite(d.y_ref) & np.isfinite(d.basal_melt)
    d = d[finite].copy()
    d["cell_x"] = np.floor(d.x_ref / CELL_M).astype(np.int64)
    d["cell_y"] = np.floor(d.y_ref / CELL_M).astype(np.int64)
    out = d.groupby(["shelf", "cell_x", "cell_y"], as_index=False).agg(
        basal_melt=("basal_melt", "median"),
        sigma_stat=("basal_melt_sigma_stat_lower_bound", "median"),
        n_records=("basal_melt", "size"),
        n_windows=("t_center", "nunique"),
    )
    out["x"] = (out.cell_x + 0.5) * CELL_M
    out["y"] = (out.cell_y + 0.5) * CELL_M
    return out


def idw(train_xy, train_z, query_xy, *, k: int, power: float,
        return_spread: bool = False):
    """IDW local escalável; devolve também distância e dispersão local."""
    tree = cKDTree(np.asarray(train_xy, dtype=float))
    kk = min(int(k), len(train_xy))
    dist, idx = tree.query(np.asarray(query_xy, dtype=float), k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    z = np.asarray(train_z, dtype=float)[idx]
    exact = dist[:, 0] < 1e-6
    weights = 1.0 / np.maximum(dist, 1.0) ** float(power)
    weights /= weights.sum(axis=1, keepdims=True)
    pred = np.sum(weights * z, axis=1)
    pred[exact] = z[exact, 0]
    if not return_spread:
        return pred
    spread = np.sqrt(np.sum(weights * (z - pred[:, None]) ** 2, axis=1))
    spread[exact] = 0.0
    return pred, dist[:, 0], spread


def spatial_folds(cells: pd.DataFrame) -> np.ndarray:
    """Atribui blocos inteiros a folds; evita vazamento espacial ponto a ponto."""
    bx = np.floor(cells.x.to_numpy() / BLOCK_M).astype(np.int64)
    by = np.floor(cells.y.to_numpy() / BLOCK_M).astype(np.int64)
    block = pd.Series(list(zip(bx, by)))
    unique = sorted(block.unique())
    rng = np.random.default_rng(0)
    shuffled = np.asarray(unique, dtype=object)[rng.permutation(len(unique))]
    mapping = {tuple(value): i % N_FOLDS for i, value in enumerate(shuffled)}
    return np.asarray([mapping[tuple(value)] for value in block], dtype=int)


def select_idw(cells: pd.DataFrame):
    """Seleciona k e potência por CV espacial, sempre dentro da plataforma."""
    folds = spatial_folds(cells)
    candidates = [(k, p) for k in (8, 16, 32) for p in (1.0, 2.0, 3.0)]
    rows = []
    for k, power in candidates:
        observed, predicted = [], []
        for fold in range(N_FOLDS):
            train = cells[folds != fold]
            test = cells[folds == fold]
            for shelf, target in test.groupby("shelf"):
                source = train[train.shelf == shelf]
                if len(source) < 4:
                    continue
                estimate = idw(
                    source[["x", "y"]].to_numpy(), source.basal_melt.to_numpy(),
                    target[["x", "y"]].to_numpy(), k=k, power=power)
                observed.extend(target.basal_melt.to_numpy())
                predicted.extend(estimate)
        observed = np.asarray(observed, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        residual = predicted - observed
        rows.append({
            "k": k, "power": power, "n": int(len(residual)),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
        })
    table = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    winner = table.iloc[0].to_dict()
    return winner, table


def floating_grid(mask_source: Path, roi_path, extent):
    """Grade de 5 km com fração flutuante calculada na máscara de 500 m.

    Uma célula entra se contiver ao menos um pixel flutuante. A geometria exata
    de 500 m volta a ser aplicada ao renderizar a figura final.
    """
    gx = np.arange(np.floor(extent[0] / CELL_M) * CELL_M + CELL_M / 2,
                   np.ceil(extent[1] / CELL_M) * CELL_M, CELL_M)
    gy = np.arange(np.floor(extent[2] / CELL_M) * CELL_M + CELL_M / 2,
                   np.ceil(extent[3] / CELL_M) * CELL_M, CELL_M)
    xx, yy = np.meshgrid(gx, gy)
    with Dataset(mask_source) as source:
        sx = np.asarray(source["x"][:], dtype=float)
        sy = np.asarray(source["y"][:], dtype=float)
        xsel = (sx >= gx[0] - CELL_M / 2) & (sx < gx[-1] + CELL_M / 2)
        ysel = (sy >= gy[0] - CELL_M / 2) & (sy < gy[-1] + CELL_M / 2)
        ix = np.flatnonzero(xsel)
        iy = np.flatnonzero(ysel)
        mask = np.asarray(source["mask"][iy.min():iy.max() + 1,
                                         ix.min():ix.max() + 1])
        hx = sx[ix.min():ix.max() + 1]
        hy = sy[iy.min():iy.max() + 1]

    hxx, hyy = np.meshgrid(hx, hy)
    col = np.floor((hxx - (gx[0] - CELL_M / 2)) / CELL_M).astype(int)
    row = np.floor((hyy - (gy[0] - CELL_M / 2)) / CELL_M).astype(int)
    inside = ((col >= 0) & (col < len(gx)) &
              (row >= 0) & (row < len(gy)))
    flat_index = row[inside] * len(gx) + col[inside]
    total = np.bincount(flat_index, minlength=xx.size)
    count = np.bincount(
        flat_index, weights=(mask[inside] == 3).astype(float),
        minlength=xx.size)
    fraction = np.divide(
        count, total, out=np.zeros_like(count), where=total > 0).reshape(xx.shape)
    floating = (fraction > 0) & inside_roi(roi_path, xx, yy)
    return gx, gy, xx, yy, floating, fraction


def interpolate_shelves(cells, targets, winner):
    """Predição separada por plataforma; não mistura lados opostos de terra."""
    obs_xy = cells[["x", "y"]].to_numpy()
    nearest = cKDTree(obs_xy).query(targets[["x", "y"]].to_numpy(), k=1)
    targets = targets.copy()
    targets["shelf"] = cells.iloc[nearest[1]].shelf.to_numpy()
    targets["distance_observation_m"] = nearest[0]
    targets["basal_melt"] = np.nan
    targets["local_spread"] = np.nan
    targets["sigma_stat_nearest"] = cells.iloc[nearest[1]].sigma_stat.to_numpy()
    for shelf, query in targets.groupby("shelf"):
        source = cells[cells.shelf == shelf]
        if len(source) < 4:
            continue
        pred, dist, spread = idw(
            source[["x", "y"]].to_numpy(), source.basal_melt.to_numpy(),
            query[["x", "y"]].to_numpy(), k=int(winner["k"]),
            power=float(winner["power"]), return_spread=True)
        targets.loc[query.index, "basal_melt"] = pred
        targets.loc[query.index, "distance_observation_m"] = dist
        targets.loc[query.index, "local_spread"] = spread
    # Mantém a plataforma distante como extrapolação condicionada à máscara.
    # Ela não é confundida com observação: a classe de suporte e a incerteza
    # registram explicitamente a distância ao dado ATL06 mais próximo.
    # É uma incerteza de interpolação operacional, não a incerteza física total.
    targets["sigma_interpolation"] = np.sqrt(
        float(winner["rmse"]) ** 2
        + targets.local_spread.fillna(0).to_numpy() ** 2
        + targets.sigma_stat_nearest.fillna(0).to_numpy() ** 2
        + (targets.distance_observation_m / MAX_SUPPORT_M
           * float(winner["rmse"])) ** 2)
    targets["support_class"] = np.select(
        [targets.distance_observation_m <= 7_500,
         targets.distance_observation_m <= HIGH_SUPPORT_M,
         targets.distance_observation_m <= MAX_SUPPORT_M],
        ["observed_near", "interpolated", "weak_extrapolation"],
        default="mask_constrained_extrapolation")
    return targets


def highres_masked_surface(field: pd.DataFrame, cells: pd.DataFrame,
                           winner: dict, mask_source: Path, roi_path, extent):
    """Avalia o IDW na geometria flutuante exata de 500 m, por plataforma."""
    with Dataset(mask_source) as source:
        sx = np.asarray(source["x"][:], dtype=float)
        sy = np.asarray(source["y"][:], dtype=float)
        xsel = (sx >= extent[0]) & (sx <= extent[1])
        ysel = (sy >= extent[2]) & (sy <= extent[3])
        ix = np.flatnonzero(xsel)
        iy = np.flatnonzero(ysel)
        hx = sx[ix.min():ix.max() + 1]
        hy = sy[iy.min():iy.max() + 1]
        mask = np.asarray(source["mask"][iy.min():iy.max() + 1,
                                         ix.min():ix.max() + 1])
    hxx, hyy = np.meshgrid(hx, hy)
    floating = (mask == 3) & inside_roi(roi_path, hxx, hyy)
    values = np.full(mask.shape, np.nan, dtype=np.float32)
    query = np.column_stack((hxx[floating], hyy[floating]))
    if len(query):
        nearest = cKDTree(field[["x", "y"]].to_numpy()).query(query, k=1)[1]
        query_shelf = field.iloc[nearest].shelf.to_numpy()
        query_values = np.full(len(query), np.nan, dtype=float)
        for shelf in np.unique(query_shelf):
            take = query_shelf == shelf
            source = cells[cells.shelf == shelf]
            if len(source) < 4:
                continue
            query_values[take] = idw(
                source[["x", "y"]].to_numpy(),
                source.basal_melt.to_numpy(), query[take],
                k=int(winner["k"]), power=float(winner["power"]))
        values[floating] = query_values.astype(np.float32)
    return hx, hy, values


def draw_base(ax, cfg, roi_x, roi_y, roi_path, extent):
    draw_basemap(ax, cfg, *extent, target_px=850)
    draw_calving_fronts(ax, cfg, 2022.5)
    ax.plot(roi_x / 1000, roi_y / 1000, color="#202020", lw=1.0,
            ls=(0, (4, 2)), zorder=13)
    ax.set_xlim(extent[0] / 1000, extent[1] / 1000)
    ax.set_ylim(extent[2] / 1000, extent[3] / 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    add_scale_bar(ax, 100)


def plot_maps(field, cells, winner, cfg, roi_x, roi_y, roi_path, extent,
              output, mask_source):
    valid = field[np.isfinite(field.basal_melt)].copy()
    hx, hy, basal_surface = highres_masked_surface(
        field, cells, winner, mask_source, roi_path, extent)

    fig, ax = plt.subplots(figsize=(10.8, 8.8), constrained_layout=True)
    draw_base(ax, cfg, roi_x, roi_y, roi_path, extent)
    art = ax.pcolormesh(
        hx / 1000, hy / 1000, np.ma.masked_invalid(basal_surface),
        shading="nearest", cmap=BASAL_CMAP,
        norm=BoundaryNorm(BASAL_LEVELS, plt.get_cmap(BASAL_CMAP).N,
                          clip=False),
        zorder=8, rasterized=True)
    cbar = fig.colorbar(
        art, ax=ax, shrink=0.80, extend="both", pad=0.02,
        ticks=[-60, -40, -20, 0, 20, 40, 60],
        label="derretimento basal (m gelo/ano; positivo = derretimento)")
    cbar.ax.axhline(0, color="#202020", lw=0.9)
    ax.set_title(
        "a. Onde está derretendo? Campo basal JJA derivado de ATL06 v007\n"
        "2019–2025 · toda plataforma flutuante dentro da ROI · máscara de 500 m",
        loc="left", fontweight="bold")
    fig.savefig(output / "mapa_derretimento_basal_atl06_jja_continuo.png",
                dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 8.8), constrained_layout=True)
    draw_base(ax, cfg, roi_x, roi_y, roi_path, extent)
    vmax = float(max(np.nanquantile(valid.sigma_interpolation, 0.95), 2.0))
    art = ax.scatter(valid.x / 1000, valid.y / 1000,
                     c=valid.sigma_interpolation, s=37, marker="s",
                     edgecolor="none", cmap="magma",
                     norm=Normalize(vmin=0, vmax=vmax), zorder=8,
                     rasterized=True)
    fig.colorbar(art, ax=ax, shrink=0.80, extend="max", pad=0.02,
                 label="incerteza operacional da interpolação (m/ano)")
    ax.set_title(
        "b. Onde o mapa é sustentado? Incerteza e cobertura ATL06 JJA\n"
        "inclui erro de CV, dispersão local, σ estatístico mínimo e distância",
        loc="left", fontweight="bold")
    fig.savefig(output / "mapa_suporte_atl06_jja_continuo.png",
                dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    cfg = load_config("jja")
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="ase_jja_atl06_continuous")
    output = ROOT / "outputs" / "diagnostico_ase_jja"
    output.mkdir(parents=True, exist_ok=True)
    roi_x, roi_y, roi_path, extent = roi_polygon()

    records = pd.read_parquet(cfg.paths.dhdt_dir / "shelf_basal_melt.parquet")
    keep = inside_roi(roi_path, records.x_ref.to_numpy(), records.y_ref.to_numpy())
    records = records[keep & records.shelf.notna()].copy()
    cells = aggregate_cells(records)
    winner, cv = select_idw(cells)
    log.info(f"CV espacial IDW:\n{cv.to_string(index=False)}")
    log.info(f"vencedor: k={int(winner['k'])}, p={winner['power']:.1f}, "
             f"RMSE={winner['rmse']:.2f} m/ano")

    # Usa exatamente o mesmo BedMachine que desenha a plataforma no basemap.
    # A máscara reprojetada da arquitetura oceânica diferia em 66 células de
    # borda (37 ausentes e 29 excedentes), inclusive no setor sul da ROI.
    mask_source = sorted(
        cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))[0]
    gx, gy, xx, yy, floating, fraction = floating_grid(
        mask_source, roi_path, extent)
    target = pd.DataFrame({
        "x": xx[floating], "y": yy[floating],
        "floating_fraction": fraction[floating]})
    field = interpolate_shelves(cells, target, winner)
    field["cell_x"] = np.floor(field.x / CELL_M).astype(np.int64)
    field["cell_y"] = np.floor(field.y / CELL_M).astype(np.int64)
    field.to_parquet(output / "basal_melt_atl06_jja_continuous.parquet", index=False)
    plot_maps(field, cells, winner, cfg, roi_x, roi_y, roi_path, extent,
              output, mask_source)

    valid = np.isfinite(field.basal_melt)
    report = {
        "status": "EXPLORATORIO_ATL06_INTERPOLADO_NAO_REPLICA_MENG",
        "source": "ICESat-2 ATL06 v007, JJA 2019-2025",
        "method": {
            "input": "estimativas lagrangianas de derretimento basal por parcela",
            "aggregation": "mediana em células de 5 km",
            "interpolator": "IDW local separado por plataforma",
            "spatial_cv": "5 folds em blocos de 50 km",
            "winner": winner,
            "all_candidates": cv.to_dict(orient="records"),
            "mask": ("BedMachine Antarctica v4.1 original, classe 3 "
                     "(floating_ice), mesma fonte do basemap; fração calculada "
                     "a 500 m e recorte final na geometria nativa"),
            "maximum_support_distance_km": MAX_SUPPORT_M / 1000,
            "beyond_maximum_support": (
                "mantido como extrapolação condicionada à máscara, não dado "
                "observado"),
        },
        "coverage": {
            "observed_cells": int(len(cells)),
            "floating_mask_cells": int(len(field)),
            "mapped_cells": int(valid.sum()),
            "mapped_fraction_of_floating_mask": float(valid.mean()),
            "by_support_class": {
                str(k): int(v) for k, v in field.support_class.value_counts().items()
            },
        },
        "limitations": [
            "interpola m_b já calculado; Meng et al. ajustam primeiro campos de altura em intervalos de 3 meses",
            "a máscara BedMachine tem época nominal anterior ao período e não substitui frentes anuais",
            "valores além de 75 km são extrapolação condicionada à plataforma e não evidência observacional local",
            "a incerteza é operacional e não inclui todas as fontes físicas nem covariâncias",
            "JJA é uma estimativa sazonal de inverno e não representa a média anual",
            "o FAC/SMB local termina em 2022 e é extrapolado no pipeline basal atual",
        ],
    }
    report_path = output / "basal_melt_atl06_jja_continuous_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    log.info(f"campo -> {output / 'basal_melt_atl06_jja_continuous.parquet'}")
    log.info(f"mapas -> {output}")


if __name__ == "__main__":
    main()
