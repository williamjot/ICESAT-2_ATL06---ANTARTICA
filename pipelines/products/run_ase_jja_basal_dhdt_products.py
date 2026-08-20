"""Produtos integrados JJA de dh/dt e derretimento basal no ASE.

Gera um mapa estático conjunto e duas animações em cinco janelas móveis de
três invernos (2019–2021 ... 2023–2025). As grandezas permanecem separadas por
domínio, colormap e barra de cores:

* dh/dt: mudança da superfície no gelo aterrado;
* m_b: balanço basal inferido na plataforma flutuante.

O suporte espacial do dh/dt é fixo nos nós presentes nos sete anos. O suporte
basal animado é a interseção das células sustentadas em todas as janelas.
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
from PIL import Image

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
    interpolate_shelves,
    select_idw,
)


WINDOWS = ((2019, 2021), (2020, 2022), (2021, 2023),
           (2022, 2024), (2023, 2025))
# Paletas deliberadamente distintas. A direção segue a solicitação aprovada:
# dh/dt em RdBu_r e balanço basal em BrBG.
DHDT_CMAP = "RdBu_r"
BASAL_CMAP = "BrBG"
DHDT_LEVELS = np.asarray([
    -3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5,
    -0.25, 0.0, 0.25, 0.5, 0.75, 1.0,
])
BASAL_LEVELS = np.arange(-60.0, 61.0, 10.0)
KM = 1e-3
DPI = 220


def load_base_assets(cfg, extent, target_px=850):
    """Carrega REMA/BedMachine uma vez e reutiliza em todos os quadros."""
    hill_extent, hillshade, _ = load_hillshade(
        cfg, *extent, target_px=target_px)
    gx, gy, masks = mask_contours(cfg, *extent)
    return {
        "hill_extent": hill_extent,
        "hillshade": hillshade,
        "gx": gx,
        "gy": gy,
        "masks": masks,
    }


def draw_base(ax, cfg, assets, roi_x, roi_y, extent):
    """Mesmo padrão: oceano azul uniforme e continente com DEM sombreado."""
    gx, gy, masks = assets["gx"], assets["gy"], assets["masks"]
    ax.set_facecolor("#dfeaf4")
    ax.contourf(gx, gy, masks["ice"], levels=[0.5, 1.5],
                colors=["#b8b8b8"], zorder=0.5)
    ax.imshow(assets["hillshade"], extent=assets["hill_extent"], cmap="gray",
              vmin=0.05, vmax=1.30, origin="upper", interpolation="bilinear",
              zorder=1)
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


def grid_from_points(frame: pd.DataFrame, column: str):
    """Converte nós regulares de 5 km em grade 2-D com NaN nos vazios."""
    x = np.sort(frame.x.unique())
    y = np.sort(frame.y.unique())
    xi = {value: i for i, value in enumerate(x)}
    yi = {value: i for i, value in enumerate(y)}
    z = np.full((len(y), len(x)), np.nan)
    for px, py, value in frame[["x", "y", column]].itertuples(index=False):
        z[yi[py], xi[px]] = value
    return x, y, z


def load_dhdt_full(cfg, roi_path):
    """Campo de dh/dt de período completo já aprovado no QC."""
    nodes = pd.read_parquet(
        cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
        columns=["x", "y", "dhdt", "mask_class", "reliability"])
    keep = inside_roi(roi_path, nodes.x.to_numpy(), nodes.y.to_numpy())
    nodes = nodes[keep & nodes.mask_class.eq(2)].copy()
    if "reliability" in nodes:
        nodes = nodes[nodes.reliability.astype(str).str.startswith("confi", na=False)]
    return nodes[np.isfinite(nodes.dhdt)].copy()


def dhdt_moving_windows(cfg, roi_path):
    """OLS sobre medianas anuais, com o mesmo suporte em todas as janelas."""
    series = pd.read_parquet(
        cfg.paths.dhdt_dir / "serie_anual.parquet",
        columns=["x", "y", "ano", "h"])
    keep = inside_roi(roi_path, series.x.to_numpy(), series.y.to_numpy())
    series = series[keep].copy()
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


def basal_moving_windows(cfg, roi_path, extent):
    """Interpola cada janela com parâmetros fixos e suporte espacial comum."""
    records = pd.read_parquet(cfg.paths.dhdt_dir / "shelf_basal_melt.parquet")
    keep = inside_roi(roi_path, records.x_ref.to_numpy(), records.y_ref.to_numpy())
    records = records[keep & records.shelf.notna()].copy()
    global_cells = aggregate_cells(records)
    winner, cv = select_idw(global_cells)
    bedmachine = sorted(cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))[0]
    _, _, xx, yy, floating = floating_grid(bedmachine, roi_path, extent)
    target = pd.DataFrame({"x": xx[floating], "y": yy[floating]})

    fields, observed = {}, {}
    for window in WINDOWS:
        selected = records[
            records.window_start.eq(window[0]) & records.window_end.eq(window[1])]
        cells = aggregate_cells(selected)
        observed[window] = cells
        fields[window] = interpolate_shelves(cells, target, winner)

    common = np.logical_and.reduce([
        np.isfinite(fields[window].basal_melt.to_numpy()) for window in WINDOWS])
    for window in WINDOWS:
        fields[window] = fields[window].loc[common].copy()
    return records, global_cells, fields, observed, winner, cv, int(common.sum())


def color_norms(dhdt_full, dhdt_windows, global_basal, basal_windows):
    """Escalas discretas, fixas no tempo e com zero como fronteira de classe."""
    del dhdt_full, dhdt_windows, global_basal, basal_windows
    dh_cmap = plt.get_cmap(DHDT_CMAP)
    basal_cmap = plt.get_cmap(BASAL_CMAP)
    return (BoundaryNorm(DHDT_LEVELS, dh_cmap.N, clip=False),
            BoundaryNorm(BASAL_LEVELS, basal_cmap.N, clip=False))


def plot_layers(ax, dhdt, basal, dh_norm, basal_norm):
    dx, dy, dz = grid_from_points(dhdt, "dhdt")
    bx, by, bz = grid_from_points(basal, "basal_melt")
    dh_artist = ax.pcolormesh(dx * KM, dy * KM, dz, cmap=DHDT_CMAP,
                              norm=dh_norm, shading="auto", zorder=7,
                              rasterized=True)
    basal_artist = ax.pcolormesh(bx * KM, by * KM, bz, cmap=BASAL_CMAP,
                                 norm=basal_norm, shading="auto", zorder=8,
                                 rasterized=True)
    return dh_artist, basal_artist


def add_two_colorbars(fig, dh_artist, basal_artist, *, animation=False):
    if animation:
        dh_ax = fig.add_axes([0.865, 0.535, 0.022, 0.285])
        basal_ax = fig.add_axes([0.865, 0.175, 0.022, 0.285])
    else:
        dh_ax = fig.add_axes([0.865, 0.555, 0.022, 0.285])
        basal_ax = fig.add_axes([0.865, 0.185, 0.022, 0.285])
    c1 = fig.colorbar(dh_artist, cax=dh_ax, extend="both")
    c1.set_ticks([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1])
    c1.ax.axhline(0, color="#202020", lw=0.9)
    c1.set_label("dh/dt superficial (m/ano)", fontsize=9)
    c2 = fig.colorbar(basal_artist, cax=basal_ax, extend="both")
    c2.set_ticks([-60, -40, -20, 0, 20, 40, 60])
    c2.ax.axhline(0, color="#202020", lw=0.9)
    c2.set_label("derretimento basal (m gelo/ano)", fontsize=9)


def static_joint(cfg, assets, roi_x, roi_y, extent, dhdt_full, basal_full,
                 dh_norm, basal_norm, output):
    fig = plt.figure(figsize=(12.0, 8.5), facecolor="white")
    ax = fig.add_axes([0.075, 0.095, 0.745, 0.815])
    draw_base(ax, cfg, assets, roi_x, roi_y, extent)
    dh_artist, basal_artist = plot_layers(
        ax, dhdt_full, basal_full, dh_norm, basal_norm)
    add_two_colorbars(fig, dh_artist, basal_artist)
    fig.suptitle(
        "Mudança superficial e derretimento basal — ASE, JJA 2019–2025",
        fontsize=17, fontweight="semibold", y=0.965)
    fig.text(
        0.46, 0.025,
        "dh/dt no gelo aterrado · balanço basal inferido no gelo flutuante · "
        "escalas independentes",
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


def render_animation_frame(cfg, assets, roi_x, roi_y, extent, window, active,
                           basal, basal_norm, path, dhdt=None, dh_norm=None):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor="white")
    ax = fig.add_axes([0.055, 0.145, 0.765, 0.755])
    draw_base(ax, cfg, assets, roi_x, roi_y, extent)
    if dhdt is None:
        bx, by, bz = grid_from_points(basal, "basal_melt")
        basal_artist = ax.pcolormesh(
            bx * KM, by * KM, bz, cmap=BASAL_CMAP, norm=basal_norm,
            shading="auto", zorder=8, rasterized=True)
        cax = fig.add_axes([0.855, 0.245, 0.022, 0.50])
        cbar = fig.colorbar(basal_artist, cax=cax, extend="both")
        cbar.set_ticks([-60, -40, -20, 0, 20, 40, 60])
        cbar.ax.axhline(0, color="#202020", lw=0.9)
        cbar.set_label("derretimento basal (m gelo/ano)", fontsize=10)
        title = f"Derretimento basal JJA · janela {window[0]}–{window[1]}"
    else:
        dh_artist, basal_artist = plot_layers(
            ax, dhdt, basal, dh_norm, basal_norm)
        add_two_colorbars(fig, dh_artist, basal_artist, animation=True)
        title = ("Mudança superficial + derretimento basal JJA · "
                 f"janela {window[0]}–{window[1]}")
    fig.suptitle(title, fontsize=18, fontweight="semibold", y=0.965)
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
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
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


def animations(cfg, assets, roi_x, roi_y, extent, basal_windows, dhdt_windows,
               basal_norm, dh_norm, output):
    basal_dir = output / "animacao_basal_jja"
    joint_dir = output / "animacao_conjunta_jja"
    for directory in (basal_dir, joint_dir):
        (directory / "quadros").mkdir(parents=True, exist_ok=True)

    basal_frames, joint_frames = [], []
    for index, window in enumerate(WINDOWS):
        basal_path = basal_dir / "quadros" / f"basal_{index+1:02d}.png"
        render_animation_frame(
            cfg, assets, roi_x, roi_y, extent, window, index,
            basal_windows[window], basal_norm, basal_path)
        basal_frames.append(basal_path)

        joint_path = joint_dir / "quadros" / f"conjunto_{index+1:02d}.png"
        render_animation_frame(
            cfg, assets, roi_x, roi_y, extent, window, index,
            basal_windows[window], basal_norm, joint_path,
            dhdt=dhdt_windows[window], dh_norm=dh_norm)
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
                        run_name="ase_jja_basal_dhdt_products")
    output = ROOT / "outputs" / "diagnostico_ase_jja"
    output.mkdir(parents=True, exist_ok=True)
    roi_x, roi_y, roi_path, extent = roi_polygon()

    log.info("Carregando contexto cartográfico uma única vez")
    assets = load_base_assets(cfg, extent)
    dhdt_full = load_dhdt_full(cfg, roi_path)
    dhdt_windows, fixed_dhdt_nodes, years = dhdt_moving_windows(cfg, roi_path)
    (basal_records, global_basal_cells, basal_windows, observed_windows,
     winner, cv, common_basal_cells) = basal_moving_windows(
         cfg, roi_path, extent)

    full_target = pd.DataFrame({
        "x": np.concatenate([frame.x.to_numpy() for frame in basal_windows.values()]),
        "y": np.concatenate([frame.y.to_numpy() for frame in basal_windows.values()]),
        "basal_melt": np.concatenate([
            frame.basal_melt.to_numpy() for frame in basal_windows.values()]),
    })
    basal_full = full_target.groupby(["x", "y"], as_index=False).basal_melt.median()
    dh_norm, basal_norm = color_norms(
        dhdt_full, dhdt_windows, global_basal_cells, basal_windows)

    static_path = static_joint(
        cfg, assets, roi_x, roi_y, extent, dhdt_full, basal_full,
        dh_norm, basal_norm, output)
    animation_paths = animations(
        cfg, assets, roi_x, roi_y, extent, basal_windows, dhdt_windows,
        basal_norm, dh_norm, output)

    report = {
        "status": "PRODUTO_EXPLORATORIO_JJA",
        "roi": "115W–95W; 77.5S–73S; EPSG:3031",
        "period": "2019-2025",
        "windows": [f"{start}-{end}" for start, end in WINDOWS],
        "dhdt": {
            "meaning": "mudança de elevação superficial no gelo aterrado",
            "static_source": "dhdt_nodes_qc.parquet, ajuste 2019-2025",
            "animation_source": "OLS de h anual em cada janela de 3 anos",
            "fixed_nodes_present_2019_2025": fixed_dhdt_nodes,
            "years": years,
            "colormap": DHDT_CMAP,
            "color_levels": DHDT_LEVELS.tolist(),
        },
        "basal_melt": {
            "meaning": "balanço basal inferido no gelo flutuante",
            "source": "shelf_basal_melt.parquet, ATL06 v007 + SMB/FAC/fluxo",
            "fixed_cells_common_to_all_windows": common_basal_cells,
            "interpolation": {
                "method": "IDW separado por plataforma",
                "winner": winner,
                "cv_candidates": cv.to_dict(orient="records"),
            },
            "colormap": BASAL_CMAP,
            "color_levels": BASAL_LEVELS.tolist(),
        },
        "caveats": [
            "janelas consecutivas compartilham dois de três invernos e não são independentes",
            "dh/dt e derretimento basal têm significados físicos, suportes e incertezas diferentes",
            "o mapa basal interpola m_b já calculado; não replica o ajuste de altura de Meng et al.",
            "a animação mostra mudança entre estimativas correlacionadas, não prova aceleração",
            "JJA não representa o ciclo anual completo",
        ],
        "outputs": {
            "static": str(static_path),
            "basal_gif": str(animation_paths[0]),
            "basal_mp4": str(animation_paths[1]),
            "joint_gif": str(animation_paths[2]),
            "joint_mp4": str(animation_paths[3]),
        },
    }
    report_path = output / "dhdt_basal_jja_products_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    log.info(f"mapa -> {static_path}")
    for path in animation_paths:
        log.info(f"animação -> {path}")
    log.info(f"relatório -> {report_path}")


if __name__ == "__main__":
    main()
