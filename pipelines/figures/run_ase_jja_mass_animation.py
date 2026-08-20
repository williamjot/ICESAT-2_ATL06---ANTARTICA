"""Mapa e animação JJA da perda de massa inferida, no padrão cartográfico ASE."""

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
from matplotlib.path import Path as MplPath
from PIL import Image
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.viz.basemap import add_scale_bar, draw_basemap, draw_calving_fronts
from pipelines.run_figuras_massa import DIV_THIN

OUTPUT = ROOT / "outputs" / "mecanismo_oceanico_regional" / "animacao_massa_jja"
FRAMES = OUTPUT / "quadros"
RHO_ICE = 917.0
RHO_WATER = 1000.0
MASS_LEVELS = np.asarray([
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    -0.5, 0.0, 0.5, 1.0, 2.0, 3.0,
])


def roi_polygon(n=300):
    lon = np.r_[np.linspace(-115, -95, n), np.full(n, -95),
                np.linspace(-95, -115, n), np.full(n, -115), -115]
    lat = np.r_[np.full(n, -77.5), np.linspace(-77.5, -73, n),
                np.full(n, -73), np.linspace(-73, -77.5, n), -77.5]
    x, y = Transformer.from_crs(4326, 3031, always_xy=True).transform(lon, lat)
    vertices = np.column_stack([x, y])
    return (np.asarray(x), np.asarray(y), MplPath(vertices),
            (float(np.min(x)), float(np.max(x)),
             float(np.min(y)), float(np.max(y))))


def load_mass(cfg, roi_path):
    data = pd.read_parquet(
        cfg.paths.dhdt_dir / "serie_anual.parquet",
        columns=["x", "y", "ano", "anom"])
    keep = roi_path.contains_points(data[["x", "y"]].to_numpy(), radius=1.0)
    data = data[keep].copy()
    years = sorted(int(year) for year in data.ano.unique())
    count = data.groupby(["x", "y"]).ano.nunique()
    complete = count[count == len(years)].index
    indexed = data.set_index(["x", "y"])
    data = indexed.loc[indexed.index.isin(complete)].reset_index()
    data["mwe"] = data.anom * RHO_ICE / RHO_WATER
    return data, years, int(len(complete))


def to_grid(frame):
    x = np.sort(frame.x.unique())
    y = np.sort(frame.y.unique())
    pivot = frame.pivot(index="y", columns="x", values="mwe")
    pivot = pivot.reindex(index=y, columns=x)
    return x, y, pivot.to_numpy(dtype=float)


def base(ax, cfg, roi_x, roi_y, extent):
    draw_basemap(ax, cfg, *extent, target_px=850)
    draw_calving_fronts(ax, cfg, 2022.5, color="#00509e", lw=0.9)
    ax.plot(roi_x / 1000, roi_y / 1000, color="#202020", lw=0.9,
            ls=(0, (4, 2)), zorder=13)
    ax.set_xlim(extent[0] / 1000, extent[1] / 1000)
    ax.set_ylim(extent[2] / 1000, extent[3] / 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    add_scale_bar(ax, 100)


def timeline(fig, years, active):
    ax = fig.add_axes([0.255, 0.055, 0.48, 0.055])
    x = np.arange(len(years), dtype=float)
    ax.plot(x, np.zeros_like(x), color="#b8b8b3", lw=1.2)
    ax.scatter(x, np.zeros_like(x), s=30, color="#d2d2ce", zorder=2)
    ax.scatter([x[active]], [0], s=88, color="#a8481c", zorder=3)
    for index, year in enumerate(years):
        ax.text(x[index], -0.20, str(year), ha="center", va="top", fontsize=8.2,
                color="#1f2933" if index == active else "#777772",
                fontweight="semibold" if index == active else "normal")
    ax.set_xlim(-0.45, len(years) - 0.55)
    ax.set_ylim(-0.55, 0.30)
    ax.axis("off")


def frame(cfg, data, years, active, roi_x, roi_y, extent, path):
    year = years[active]
    selected = data[data.ano.eq(year)]
    x, y, values = to_grid(selected)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor="white")
    ax = fig.add_axes([0.09, 0.145, 0.70, 0.755])
    base(ax, cfg, roi_x, roi_y, extent)
    artist = ax.pcolormesh(
        x / 1000, y / 1000, values, cmap=DIV_THIN,
        norm=BoundaryNorm(MASS_LEVELS, DIV_THIN.N, clip=False),
        shading="auto", rasterized=True, zorder=8)
    cax = fig.add_axes([0.84, 0.225, 0.023, 0.56])
    cbar = fig.colorbar(artist, cax=cax, extend="both",
                        ticks=[-8, -6, -4, -2, 0, 1, 2, 3])
    cbar.ax.axhline(0, color="#202020", lw=0.9)
    cbar.set_label("variação acumulada (m de água equivalente)", fontsize=10)
    fig.suptitle(
        f"Perda de massa inferida JJA · 2019 → {year}",
        fontsize=18, fontweight="semibold", y=0.965)
    fig.text(
        0.44, 0.915,
        "anomalia relativa a 2019 · suporte espacial fixo · escala fixa",
        ha="center", fontsize=10, color="#4b5563")
    timeline(fig, years, active)
    fig.text(
        0.49, 0.018,
        "Massa inferida de altura × 917 kg m⁻³; não é medida gravimétrica.",
        ha="center", fontsize=8.5, color="#a8481c")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def write_animation(paths):
    images = [Image.open(path).convert("RGB") for path in paths]
    durations = [1500] * len(images)
    durations[0], durations[-1] = 1800, 2800
    gif = OUTPUT / "perda_massa_inferida_jja_2019_2025.gif"
    images[0].save(gif, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, disposal=2, optimize=False)
    for image in images:
        image.close()
    first = cv2.imread(str(paths[0]))
    height, width = first.shape[:2]
    mp4 = OUTPUT / "perda_massa_inferida_jja_2019_2025.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                             15.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("codec MP4V indisponível")
    try:
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            seconds = 2.8 if index == len(paths) - 1 else 1.5
            for _ in range(round(seconds * 15)):
                writer.write(image)
    finally:
        writer.release()
    return gif, mp4


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    cfg = load_config("jja")
    roi_x, roi_y, roi_path, extent = roi_polygon()
    data, years, nodes = load_mass(cfg, roi_path)
    paths = []
    for index, year in enumerate(years):
        path = FRAMES / f"massa_jja_{index + 1:02d}_{year}.png"
        frame(cfg, data, years, index, roi_x, roi_y, extent, path)
        paths.append(path)
    gif, mp4 = write_animation(paths)
    static = OUTPUT / "mapa_perda_massa_inferida_jja_2019_2025.png"
    static.write_bytes(paths[-1].read_bytes())
    report = {
        "status": "PRODUTO_JJA_CONCLUIDO",
        "years": years,
        "fixed_nodes": nodes,
        "density_kg_m3": RHO_ICE,
        "mass_levels_mwe": MASS_LEVELS.tolist(),
        "outputs": {"static": str(static), "gif": str(gif), "mp4": str(mp4)},
        "caveat": "massa inferida de altimetria e densidade; não é gravimetria",
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Mapa -> {static}")
    print(f"GIF -> {gif}")
    print(f"MP4 -> {mp4}")


if __name__ == "__main__":
    main()

