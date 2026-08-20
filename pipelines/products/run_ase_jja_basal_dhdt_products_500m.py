"""Mapas e animações JJA integrando dh/dt e derretimento basal no ASE.

Os dois campos usam a mesma paleta divergente ``RdBu_r``, mas permanecem em
escalas físicas independentes. A máscara BedMachine v4.1 original, a 500 m, é
usada tanto no cálculo quanto no desenho. O retângulo externo à ROI permanece
visível, sem esmaecimento.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from netCDF4 import Dataset
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.viz.basemap import (
    add_scale_bar,
    draw_calving_fronts,
    load_hillshade,
    mask_contours,
)
from pipelines.run_ase_jja_diagnostic_maps import inside_roi, roi_polygon
from pipelines.run_ase_jja_atl06_continuous import (
    aggregate_cells,
    floating_grid,
    idw,
    select_idw,
)


WINDOWS = ((2019, 2021), (2020, 2022), (2021, 2023),
           (2022, 2024), (2023, 2025))
CMAP = "RdBu_r"
DHDT_LEVELS = np.asarray([
    -3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5,
    -0.25, 0.0, 0.25, 0.5, 0.75, 1.0,
])
BASAL_LEVELS = np.arange(-60.0, 61.0, 10.0)
MAX_DHDT_SUPPORT_M = 75_000.0
KM = 1e-3
DPI = 220


def load_base_assets(cfg, extent, roi_path, target_px=850):
    """Carrega uma vez DEM, contornos e máscara científica nativa de 500 m."""
    hill_extent, hillshade, _ = load_hillshade(
        cfg, *extent, target_px=target_px)
    gx, gy, masks = mask_contours(cfg, *extent)
    bedmachine = sorted(
        cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))[0]
    with Dataset(bedmachine) as source:
        sx = np.asarray(source["x"][:], dtype=float)
        sy = np.asarray(source["y"][:], dtype=float)
        ix = np.flatnonzero((sx >= extent[0]) & (sx <= extent[1]))
        iy = np.flatnonzero((sy >= extent[2]) & (sy <= extent[3]))
        hx = sx[ix.min():ix.max() + 1]
        hy = sy[iy.min():iy.max() + 1]
        high_mask = np.asarray(source["mask"][iy.min():iy.max() + 1,
                                                ix.min():ix.max() + 1])
    hxx, hyy = np.meshgrid(hx, hy)
    native_roi = inside_roi(roi_path, hxx, hyy)
    dx = float(abs(np.median(np.diff(hx))))
    dy = float(abs(np.median(np.diff(hy))))
    return {
        "hill_extent": hill_extent,
        "hillshade": hillshade,
        "gx": gx,
        "gy": gy,
        "masks": masks,
        "bedmachine": bedmachine,
        "hx": hx,
        "hy": hy,
        "hxx": hxx,
        "hyy": hyy,
        "high_mask": high_mask,
        "native_roi": native_roi,
        "image_extent": (
            (hx.min() - dx / 2) * KM, (hx.max() + dx / 2) * KM,
            (hy.min() - dy / 2) * KM, (hy.max() + dy / 2) * KM),
        "image_origin": "upper" if hy[0] > hy[-1] else "lower",
    }


def draw_base(ax, cfg, assets, roi_x, roi_y, extent):
    """Oceano uniforme, continente com DEM e sem esmaecer fora da ROI."""
    gx, gy, masks = assets["gx"], assets["gy"], assets["masks"]
    ax.set_facecolor("#dfeaf4")
    ax.contourf(gx, gy, masks["ice"], levels=[0.5, 1.5],
                colors=["#b8b8b8"], zorder=0.5)
    ax.imshow(assets["hillshade"], extent=assets["hill_extent"], cmap="gray",
              vmin=0.05, vmax=1.30, origin="upper",
              interpolation="bilinear", zorder=1)
    ax.contourf(gx, gy, masks["ice"], levels=[-0.5, 0.5],
                colors=["#dfeaf4"], zorder=1.5)
    ax.contourf(gx, gy, masks["floating"], levels=[0.5, 1.5],
                colors=["#b9d6ec"], alpha=0.75, zorder=2)
    ax.contour(gx, gy, masks["ice"], levels=[0.5], colors="#1a1a1a",
               linewidths=0.8, zorder=4)
    ax.contour(gx, gy, masks["grounded"], levels=[0.5], colors="#8b0000",
               linewidths=0.75, linestyles="--", zorder=5)
    draw_calving_fronts(ax, cfg, 2022.5, color="#00509e", lw=0.9)
    ax.plot(roi_x * KM, roi_y * KM, color="#202020", lw=0.9,
            ls=(0, (4, 2)), zorder=13)
    ax.set_xlim(extent[0] * KM, extent[1] * KM)
    ax.set_ylim(extent[2] * KM, extent[3] * KM)
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    add_scale_bar(ax, 100)


def load_dhdt_full(cfg, roi_path):
    nodes = pd.read_parquet(
        cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
        columns=["x", "y", "dhdt", "mask_class", "reliability"])
    keep = inside_roi(roi_path, nodes.x.to_numpy(), nodes.y.to_numpy())
    nodes = nodes[keep & nodes.mask_class.eq(2)].copy()
    reliable = nodes.reliability.astype(str).str.startswith("confi", na=False)
    nodes = nodes[reliable & np.isfinite(nodes.dhdt)].copy()
    return nodes


def dhdt_moving_windows(cfg, roi_path, support_nodes):
    """OLS em cinco janelas de três anos, no suporte confiável comum."""
    series = pd.read_parquet(
        cfg.paths.dhdt_dir / "serie_anual.parquet",
        columns=["x", "y", "ano", "h"])
    keep = inside_roi(roi_path, series.x.to_numpy(), series.y.to_numpy())
    series = series[keep].copy()
    support = support_nodes[["x", "y"]].drop_duplicates()
    series = series.merge(support, on=["x", "y"], how="inner")
    years = sorted(int(year) for year in series.ano.unique())
    counts = series.groupby(["x", "y"]).ano.nunique()
    complete = counts[counts == len(years)].index
    series = series.set_index(["x", "y"])
    series = series.loc[series.index.isin(complete)].reset_index()
    pivot = series.pivot(index=["x", "y"], columns="ano", values="h")
    out = {}
    for start, end in WINDOWS:
        columns = list(range(start, end + 1))
        values = pivot[columns].to_numpy(dtype=float)
        t = np.asarray(columns, dtype=float)
        centered = t - t.mean()
        slope = (values @ centered) / float(np.sum(centered ** 2))
        frame = pivot.reset_index()[["x", "y"]].copy()
        frame["dhdt"] = slope
        out[(start, end)] = frame[np.isfinite(frame.dhdt)].copy()
    return out, int(len(pivot)), years


def basal_fields(cfg, roi_path, extent):
    """Campo global e campos móveis com partição de plataformas fixa."""
    records = pd.read_parquet(cfg.paths.dhdt_dir / "shelf_basal_melt.parquet")
    keep = inside_roi(roi_path, records.x_ref.to_numpy(), records.y_ref.to_numpy())
    records = records[keep & records.shelf.notna()].copy()
    global_cells = aggregate_cells(records)
    winner, cv = select_idw(global_cells)
    bedmachine = sorted(
        cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))[0]
    _, _, xx, yy, floating, fraction = floating_grid(
        bedmachine, roi_path, extent)
    target = pd.DataFrame({
        "x": xx[floating], "y": yy[floating],
        "floating_fraction": fraction[floating]})

    nearest = cKDTree(global_cells[["x", "y"]].to_numpy()).query(
        target[["x", "y"]].to_numpy(), k=1)[1]
    target["shelf"] = global_cells.iloc[nearest].shelf.to_numpy()

    def interpolate_fixed(source_cells, allow_global_fallback):
        field = target.copy()
        field["basal_melt"] = np.nan
        field["temporal_fallback"] = False
        fallback_shelves = []
        for shelf, query in field.groupby("shelf"):
            source = source_cells[source_cells.shelf == shelf]
            used_fallback = len(source) < 4
            if used_fallback and allow_global_fallback:
                source = global_cells[global_cells.shelf == shelf]
                fallback_shelves.append(str(shelf))
            if len(source) < 4:
                continue
            pred = idw(
                source[["x", "y"]].to_numpy(),
                source.basal_melt.to_numpy(),
                query[["x", "y"]].to_numpy(),
                k=int(winner["k"]), power=float(winner["power"]))
            field.loc[query.index, "basal_melt"] = pred
            field.loc[query.index, "temporal_fallback"] = used_fallback
        return field, fallback_shelves

    full, _ = interpolate_fixed(global_cells, allow_global_fallback=False)
    windows, observed, fallbacks = {}, {}, {}
    for window in WINDOWS:
        selected = records[
            records.window_start.eq(window[0]) &
            records.window_end.eq(window[1])]
        cells = aggregate_cells(selected)
        observed[window] = cells
        windows[window], fallbacks[window] = interpolate_fixed(
            cells, allow_global_fallback=True)
    return full, windows, observed, global_cells, winner, cv, fallbacks


def resample_to_native(frame, column, assets, mask_class, *,
                       max_distance_m=None):
    """Reamostragem IDW apenas para visualização na máscara nativa de 500 m."""
    values = np.full(assets["high_mask"].shape, np.nan, dtype=np.float32)
    domain = ((assets["high_mask"] == mask_class) & assets["native_roi"])
    valid = frame[np.isfinite(frame[column])]
    if not len(valid):
        return values
    query = np.column_stack((assets["hxx"][domain], assets["hyy"][domain]))
    source_xy = valid[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(source_xy)
    k = min(4, len(valid))
    dist, idx = tree.query(query, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    source_z = valid[column].to_numpy(dtype=float)[idx]
    weights = 1.0 / np.maximum(dist, 1.0) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    pred = np.sum(weights * source_z, axis=1)
    if max_distance_m is not None:
        pred[dist[:, 0] > max_distance_m] = np.nan
    values[domain] = pred.astype(np.float32)
    return values


def draw_surface(ax, surface, assets, norm, *, zorder):
    return ax.imshow(
        np.ma.masked_invalid(surface), extent=assets["image_extent"],
        origin=assets["image_origin"], interpolation="nearest", cmap=CMAP,
        norm=norm, zorder=zorder, rasterized=True)


def add_dhdt_colorbar(fig, artist, ax=None, cax=None, label=None):
    cbar = fig.colorbar(
        artist, ax=ax, cax=cax, shrink=0.80 if cax is None else 1.0,
        extend="both", pad=0.02 if cax is None else 0.0,
        ticks=[-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1])
    cbar.ax.axhline(0, color="#202020", lw=0.9)
    cbar.set_label(label or "dh/dt superficial (m/ano)")
    return cbar


def add_basal_colorbar(fig, artist, ax=None, cax=None, label=None):
    cbar = fig.colorbar(
        artist, ax=ax, cax=cax, shrink=0.80 if cax is None else 1.0,
        extend="both", pad=0.02 if cax is None else 0.0,
        ticks=[-60, -40, -20, 0, 20, 40, 60])
    cbar.ax.axhline(0, color="#202020", lw=0.9)
    cbar.set_label(
        label or "derretimento basal (m gelo/ano; positivo = derretimento)")
    return cbar


def static_dhdt(cfg, assets, roi_x, roi_y, extent, surface, norm, output):
    fig, ax = plt.subplots(figsize=(10.8, 8.8), constrained_layout=True)
    draw_base(ax, cfg, assets, roi_x, roi_y, extent)
    artist = draw_surface(ax, surface, assets, norm, zorder=7)
    add_dhdt_colorbar(fig, artist, ax=ax)
    ax.set_title(
        "Mudança de elevação superficial JJA derivada de ATL06 v007\n"
        "2019–2025 · máscara de gelo aterrado a 500 m",
        loc="left", fontweight="bold")
    path = output / "mapa_dhdt_atl06_jja_500m.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def add_two_colorbars(fig, dh_artist, basal_artist, *, animation=False):
    if animation:
        dh_ax = fig.add_axes([0.865, 0.535, 0.022, 0.285])
        basal_ax = fig.add_axes([0.865, 0.175, 0.022, 0.285])
    else:
        dh_ax = fig.add_axes([0.865, 0.555, 0.022, 0.285])
        basal_ax = fig.add_axes([0.865, 0.185, 0.022, 0.285])
    add_dhdt_colorbar(
        fig, dh_artist, cax=dh_ax,
        label="dh/dt (m/ano)" if animation else None)
    add_basal_colorbar(
        fig, basal_artist, cax=basal_ax,
        label="derretimento basal (m gelo/ano)" if animation else None)


def static_joint(cfg, assets, roi_x, roi_y, extent, dh_surface,
                 basal_surface, dh_norm, basal_norm, output):
    fig = plt.figure(figsize=(12.0, 8.5), facecolor="white")
    ax = fig.add_axes([0.075, 0.095, 0.745, 0.815])
    draw_base(ax, cfg, assets, roi_x, roi_y, extent)
    dh_artist = draw_surface(ax, dh_surface, assets, dh_norm, zorder=7)
    basal_artist = draw_surface(ax, basal_surface, assets, basal_norm, zorder=8)
    add_two_colorbars(fig, dh_artist, basal_artist)
    fig.suptitle(
        "dh/dt superficial + derretimento basal — ASE, JJA 2019–2025",
        fontsize=17, fontweight="semibold", y=0.965)
    fig.text(
        0.46, 0.025,
        "mesma paleta RdBu_r · escalas físicas independentes · "
        "gelo aterrado e flutuante",
        ha="center", fontsize=9, color="#4b5563")
    path = output / "mapa_conjunto_dhdt_derretimento_basal_jja.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path


def draw_timeline(fig, active):
    ax = fig.add_axes([0.235, 0.055, 0.48, 0.055])
    x = np.arange(len(WINDOWS), dtype=float)
    ax.plot(x, np.zeros_like(x), color="#b8b8b3", lw=1.2)
    ax.scatter(x, np.zeros_like(x), s=30, color="#d2d2ce", zorder=2)
    ax.scatter([x[active]], [0], s=88, color="#a8481c", zorder=3)
    for idx, (start, end) in enumerate(WINDOWS):
        ax.text(x[idx], -0.20, f"{start}–{end}", ha="center", va="top",
                fontsize=8.2, color="#1f2933" if idx == active else "#777772",
                fontweight="semibold" if idx == active else "normal")
    ax.set_xlim(-0.45, len(WINDOWS) - 0.55)
    ax.set_ylim(-0.55, 0.30)
    ax.axis("off")


def render_frame(cfg, assets, roi_x, roi_y, extent, window, active,
                 basal_surface, basal_norm, path, dh_surface=None,
                 dh_norm=None, fallback_shelves=None):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor="white")
    ax = fig.add_axes([0.055, 0.145, 0.765, 0.755])
    draw_base(ax, cfg, assets, roi_x, roi_y, extent)
    basal_artist = draw_surface(
        ax, basal_surface, assets, basal_norm, zorder=8)
    if dh_surface is None:
        cax = fig.add_axes([0.855, 0.245, 0.022, 0.50])
        add_basal_colorbar(fig, basal_artist, cax=cax)
        title = f"Derretimento basal JJA · janela {window[0]}–{window[1]}"
    else:
        dh_artist = draw_surface(ax, dh_surface, assets, dh_norm, zorder=7)
        # Recoloca o basal por cima do dh/dt; os domínios não se sobrepõem.
        basal_artist = draw_surface(
            ax, basal_surface, assets, basal_norm, zorder=8)
        add_two_colorbars(fig, dh_artist, basal_artist, animation=True)
        title = ("dh/dt superficial + derretimento basal JJA · "
                 f"janela {window[0]}–{window[1]}")
    fig.suptitle(title, fontsize=18, fontweight="semibold", y=0.965)
    if fallback_shelves:
        fig.text(
            0.50, 0.915,
            "Sem estimativa temporal local em " + ", ".join(fallback_shelves) +
            " — usada mediana espacial 2019–2025",
            ha="center", fontsize=8.2, color="#7a3d21")
    draw_timeline(fig, active)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def write_gif_mp4(frames, gif_path, mp4_path):
    images = [Image.open(path).convert("RGB") for path in frames]
    durations = [1700] * len(images)
    durations[-1] = 2800
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, disposal=2, optimize=False)
    for image in images:
        image.close()

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"quadro inválido: {frames[0]}")
    height, width = first.shape[:2]
    fps = 15.0
    writer = cv2.VideoWriter(
        str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (width, height))
    if not writer.isOpened():
        raise RuntimeError("codec MP4V indisponível")
    try:
        for index, path in enumerate(frames):
            frame = cv2.imread(str(path))
            seconds = 2.8 if index == len(frames) - 1 else 1.7
            for _ in range(round(seconds * fps)):
                writer.write(frame)
    finally:
        writer.release()


def animations(cfg, assets, roi_x, roi_y, extent, basal_surfaces,
               dhdt_surfaces, basal_norm, dh_norm, fallbacks, output):
    basal_dir = output / "animacao_basal_jja"
    joint_dir = output / "animacao_conjunta_jja"
    for directory in (basal_dir, joint_dir):
        (directory / "quadros").mkdir(parents=True, exist_ok=True)

    basal_frames, joint_frames = [], []
    for index, window in enumerate(WINDOWS):
        basal_path = basal_dir / "quadros" / f"basal_{index+1:02d}.png"
        render_frame(
            cfg, assets, roi_x, roi_y, extent, window, index,
            basal_surfaces[window], basal_norm, basal_path,
            fallback_shelves=fallbacks[window])
        basal_frames.append(basal_path)

        joint_path = joint_dir / "quadros" / f"conjunto_{index+1:02d}.png"
        render_frame(
            cfg, assets, roi_x, roi_y, extent, window, index,
            basal_surfaces[window], basal_norm, joint_path,
            dh_surface=dhdt_surfaces[window], dh_norm=dh_norm,
            fallback_shelves=fallbacks[window])
        joint_frames.append(joint_path)

    basal_gif = basal_dir / "derretimento_basal_jja_janelas_moveis.gif"
    basal_mp4 = basal_dir / "derretimento_basal_jja_janelas_moveis.mp4"
    joint_gif = joint_dir / "dhdt_derretimento_basal_jja_janelas_moveis.gif"
    joint_mp4 = joint_dir / "dhdt_derretimento_basal_jja_janelas_moveis.mp4"
    write_gif_mp4(basal_frames, basal_gif, basal_mp4)
    write_gif_mp4(joint_frames, joint_gif, joint_mp4)
    return basal_gif, basal_mp4, joint_gif, joint_mp4


def main():
    cfg = load_config("jja")
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="ase_jja_basal_dhdt_products_500m")
    output = ROOT / "outputs" / "diagnostico_ase_jja"
    output.mkdir(parents=True, exist_ok=True)
    roi_x, roi_y, roi_path, extent = roi_polygon()

    log.info("Carregando base cartográfica e máscara nativa de 500 m")
    assets = load_base_assets(cfg, extent, roi_path)
    dhdt_full = load_dhdt_full(cfg, roi_path)
    dhdt_windows, fixed_dhdt_nodes, years = dhdt_moving_windows(
        cfg, roi_path, dhdt_full)
    (basal_full, basal_windows, observed_windows, global_basal_cells,
     winner, cv, fallbacks) = basal_fields(cfg, roi_path, extent)

    dh_norm = BoundaryNorm(DHDT_LEVELS, plt.get_cmap(CMAP).N, clip=False)
    basal_norm = BoundaryNorm(BASAL_LEVELS, plt.get_cmap(CMAP).N, clip=False)

    log.info("Reamostrando campos para a máscara científica de 500 m")
    dh_surface = resample_to_native(
        dhdt_full, "dhdt", assets, 2,
        max_distance_m=MAX_DHDT_SUPPORT_M)
    basal_surface = resample_to_native(
        basal_full, "basal_melt", assets, 3)
    dhdt_surfaces = {
        window: resample_to_native(
            frame, "dhdt", assets, 2,
            max_distance_m=MAX_DHDT_SUPPORT_M)
        for window, frame in dhdt_windows.items()
    }
    basal_surfaces = {
        window: resample_to_native(frame, "basal_melt", assets, 3)
        for window, frame in basal_windows.items()
    }

    dhdt_path = static_dhdt(
        cfg, assets, roi_x, roi_y, extent, dh_surface, dh_norm, output)
    joint_path = static_joint(
        cfg, assets, roi_x, roi_y, extent, dh_surface, basal_surface,
        dh_norm, basal_norm, output)
    animation_paths = animations(
        cfg, assets, roi_x, roi_y, extent, basal_surfaces, dhdt_surfaces,
        basal_norm, dh_norm, fallbacks, output)

    grounded_domain = ((assets["high_mask"] == 2) & assets["native_roi"])
    floating_domain = ((assets["high_mask"] == 3) & assets["native_roi"])
    report = {
        "status": "PRODUTO_EXPLORATORIO_JJA_500M",
        "roi": "115W–95W; 77.5S–73S; EPSG:3031",
        "period": "2019-2025",
        "windows": [f"{start}-{end}" for start, end in WINDOWS],
        "common_style": {
            "colormap": CMAP,
            "native_mask_resolution_m": 500,
            "outside_roi_faded": False,
        },
        "dhdt": {
            "meaning": "mudança de elevação superficial no gelo aterrado",
            "static_source": "dhdt_nodes_qc.parquet, confiável, 2019-2025",
            "animation_source": "OLS de h anual em janelas móveis de 3 anos",
            "fixed_nodes_present_2019_2025": fixed_dhdt_nodes,
            "years": years,
            "color_levels": DHDT_LEVELS.tolist(),
            "maximum_visual_resampling_distance_km":
                MAX_DHDT_SUPPORT_M / 1000,
            "mapped_native_pixels": int(np.isfinite(dh_surface).sum()),
            "grounded_native_pixels": int(grounded_domain.sum()),
        },
        "basal_melt": {
            "meaning": "balanço basal inferido no gelo flutuante",
            "source": "shelf_basal_melt.parquet, ATL06 v007 + SMB/FAC/fluxo",
            "color_levels": BASAL_LEVELS.tolist(),
            "mapped_native_pixels": int(np.isfinite(basal_surface).sum()),
            "floating_native_pixels": int(floating_domain.sum()),
            "interpolation": {
                "method": "IDW separado por plataforma; partição espacial fixa",
                "winner": winner,
                "cv_candidates": cv.to_dict(orient="records"),
            },
            "temporal_fallback_shelves": {
                f"{start}-{end}": fallbacks[(start, end)]
                for start, end in WINDOWS
            },
        },
        "caveats": [
            "janelas consecutivas compartilham dois de três invernos e não são independentes",
            "dh/dt e derretimento basal têm significados físicos e escalas diferentes",
            "a reamostragem de 5 km para 500 m melhora a geometria de exibição, não cria resolução observacional de 500 m",
            "setores sem estimativa basal na janela usam a mediana espacial 2019–2025 e são identificados nos quadros",
            "JJA não representa o ciclo anual completo",
        ],
        "outputs": {
            "dhdt_static": str(dhdt_path),
            "joint_static": str(joint_path),
            "basal_gif": str(animation_paths[0]),
            "basal_mp4": str(animation_paths[1]),
            "joint_gif": str(animation_paths[2]),
            "joint_mp4": str(animation_paths[3]),
        },
    }
    report_path = output / "dhdt_basal_jja_products_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"mapa dh/dt -> {dhdt_path}")
    log.info(f"mapa conjunto -> {joint_path}")
    for path in animation_paths:
        log.info(f"animação -> {path}")
    log.info(f"relatório -> {report_path}")


if __name__ == "__main__":
    main()
