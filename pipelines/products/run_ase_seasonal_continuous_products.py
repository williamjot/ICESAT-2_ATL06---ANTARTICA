"""Produtos sazonais contínuos de dh/dt e derretimento basal no ASE.

Gera mapas e animações separados para JJA e DJF, todos em EPSG:3031, com
graticulado geográfico latitude/longitude reprojetado, paleta RdBu contínua
e máscara nativa BedMachine de 500 m. O dh/dt é reamostrado por IDW (k=8,
p=2, suporte de 30 km) a partir dos nós fitsec de 5 km; isso suaviza a
cartografia, mas não cria observações de 500 m.

Convenção cromática: vermelho sempre indica perda de gelo. Para o dh/dt
isso corresponde a valores negativos (afinamento); para o derretimento
basal, a valores positivos (perda pela base). Por isso o campo basal usa
RdBu_r, a mesma paleta do dh/dt com orientação invertida.

Nas animações conjuntas, as janelas de dh/dt (4 anos) e basal (3 anos) são
alinhadas pelo ano central e permanecem identificadas separadamente.
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
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from pipelines.run_ase_jja_basal_dhdt_products_500m import (
    add_basal_colorbar,
    add_dhdt_colorbar,
    basal_fields,
    draw_base,
    load_base_assets,
)
from pipelines.run_ase_jja_diagnostic_maps import inside_roi, roi_polygon


SEASONS = ("jja", "djf")
SEASON_LABELS = {
    "jja": "Inverno austral (JJA)",
    "djf": "Verão austral (DJF)",
}
DHDT_WINDOWS = (
    (2019, 2023),
    (2020, 2024),
    (2021, 2025),
    (2022, 2026),
)
DHDT_LABELS = {
    (2019, 2023): "2019–2023",
    (2020, 2024): "2020–2024",
    (2021, 2025): "2021–2025",
    (2022, 2026): "2022–2025,7",
}
# Alinhamento por centro temporal: 2021, 2022, 2023 e 2024.
JOINT_WINDOWS = (
    ((2019, 2023), (2020, 2022)),
    ((2020, 2024), (2021, 2023)),
    ((2021, 2025), (2022, 2024)),
    ((2022, 2026), (2023, 2025)),
)

# Mesma família RdBu nos dois campos; a orientação difere porque o sinal
# físico da perda de gelo é oposto entre eles.
CMAP = "RdBu"          # dh/dt: negativo (afinamento)      -> vermelho
CMAP_BASAL = "RdBu_r"  # basal: positivo (perda pela base) -> vermelho
DHDT_NORM = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=1.0)
BASAL_NORM = TwoSlopeNorm(vmin=-60.0, vcenter=0.0, vmax=60.0)
IDW_K = 8
IDW_POWER = 2.0
MAX_DHDT_SUPPORT_M = 30_000.0
KM = 1e-3
STATIC_DPI = 360
FRAME_DPI = 150
CHUNK = 200_000

# Graticulado geográfico desenhado sobre a projeção polar.
_TO_3031 = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
GRATICULE_LON = np.arange(-115.0, -94.9, 5.0)
GRATICULE_LAT = np.arange(-77.0, -72.9, 1.0)
GRATICULE_STYLE = dict(color="#2f2f2f", lw=0.5, ls=(0, (4, 3)),
                       alpha=0.55, zorder=9)
GRATICULE_LABEL_BOX = dict(boxstyle="round,pad=0.15", fc="white",
                           ec="none", alpha=0.72)


def load_dhdt_full(cfg, roi_path):
    """Nós fitsec/QC aterrados e finitos dentro da ROI."""
    nodes = pd.read_parquet(
        cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
        columns=["x", "y", "dhdt", "mask_class", "reliability"],
    )
    keep = inside_roi(roi_path, nodes.x.to_numpy(), nodes.y.to_numpy())
    finite = np.isfinite(nodes.x) & np.isfinite(nodes.y) & np.isfinite(nodes.dhdt)
    return nodes[keep & finite & nodes.mask_class.eq(2)].copy()


def load_dhdt_windows(cfg, roi_path, full_nodes):
    """Carrega janelas fitsec já processadas e mantém o suporte QC aterrado."""
    accepted = set(
        zip(
            np.rint(full_nodes.x).astype(np.int64),
            np.rint(full_nodes.y).astype(np.int64),
        )
    )
    report_path = cfg.paths.tables / "dhdt_janelas.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output = {}
    for window in DHDT_WINDOWS:
        key = f"{window[0]}-{window[1]}"
        info = report["janelas"][key]
        nodes = pd.read_parquet(
            cfg.paths.dhdt_dir / "janelas" / info["arquivo"],
            columns=["x", "y", "dhdt"],
        )
        finite = np.isfinite(nodes.x) & np.isfinite(nodes.y) & np.isfinite(nodes.dhdt)
        spatial = inside_roi(roi_path, nodes.x.to_numpy(), nodes.y.to_numpy())
        accepted_mask = np.fromiter(
            (
                (x, y) in accepted
                for x, y in zip(
                    np.rint(nodes.x).astype(np.int64),
                    np.rint(nodes.y).astype(np.int64),
                )
            ),
            dtype=bool,
            count=len(nodes),
        )
        output[window] = nodes[finite & spatial & accepted_mask].copy()
    return output


def idw_predict(source_xy, source_z, query_xy, *, k=IDW_K, power=IDW_POWER):
    """IDW em blocos para limitar memória sem mudar o resultado."""
    source_xy = np.asarray(source_xy, dtype=float)
    source_z = np.asarray(source_z, dtype=float)
    query_xy = np.asarray(query_xy, dtype=float)
    tree = cKDTree(source_xy)
    kk = min(int(k), len(source_xy))
    pred = np.empty(len(query_xy), dtype=np.float32)
    nearest = np.empty(len(query_xy), dtype=np.float32)
    for start in range(0, len(query_xy), CHUNK):
        end = min(start + CHUNK, len(query_xy))
        dist, idx = tree.query(query_xy[start:end], k=kk)
        if kk == 1:
            dist, idx = dist[:, None], idx[:, None]
        values = source_z[idx]
        weights = 1.0 / np.maximum(dist, 1.0) ** float(power)
        weights /= weights.sum(axis=1, keepdims=True)
        block = np.sum(weights * values, axis=1)
        exact = dist[:, 0] < 1e-6
        block[exact] = values[exact, 0]
        pred[start:end] = block.astype(np.float32)
        nearest[start:end] = dist[:, 0].astype(np.float32)
    return pred, nearest


def dhdt_to_native(nodes, assets):
    """Reamostra dh/dt somente no gelo aterrado e dentro da ROI."""
    surface = np.full(assets["high_mask"].shape, np.nan, dtype=np.float32)
    domain = (assets["high_mask"] == 2) & assets["native_roi"]
    query = np.column_stack((assets["hxx"][domain], assets["hyy"][domain]))
    pred, nearest = idw_predict(
        nodes[["x", "y"]].to_numpy(), nodes.dhdt.to_numpy(), query)
    pred[nearest > MAX_DHDT_SUPPORT_M] = np.nan
    surface[domain] = pred
    return surface


def basal_to_native(field, assets):
    """Suaviza o campo basal dentro de cada plataforma, sem cruzar partições."""
    surface = np.full(assets["high_mask"].shape, np.nan, dtype=np.float32)
    domain = (assets["high_mask"] == 3) & assets["native_roi"]
    query_xy = np.column_stack((assets["hxx"][domain], assets["hyy"][domain]))
    valid = field[np.isfinite(field.basal_melt)].copy()
    if not len(valid):
        return surface

    # Atribui cada pixel nativo à partição espacial de plataforma mais próxima.
    nearest_idx = cKDTree(valid[["x", "y"]].to_numpy()).query(query_xy, k=1)[1]
    query_shelf = valid.iloc[nearest_idx].shelf.astype(str).to_numpy()
    pred = np.full(len(query_xy), np.nan, dtype=np.float32)
    for shelf in sorted(valid.shelf.astype(str).unique()):
        source = valid[valid.shelf.astype(str).eq(shelf)]
        target = query_shelf == shelf
        if not target.any() or not len(source):
            continue
        values, _ = idw_predict(
            source[["x", "y"]].to_numpy(),
            source.basal_melt.to_numpy(),
            query_xy[target],
        )
        pred[target] = values
    surface[domain] = pred
    return surface


def draw_surface(ax, surface, assets, norm, *, zorder, cmap=CMAP):
    """Campo contínuo com suavização bilinear apenas de renderização."""
    return ax.imshow(
        np.ma.masked_invalid(surface),
        extent=assets["image_extent"],
        origin=assets["image_origin"],
        interpolation="bilinear",
        cmap=cmap,
        norm=norm,
        zorder=zorder,
        rasterized=True,
    )


def _axis_scale(ax):
    """Detecta se os eixos estão em metros ou em km (draw_base usa KM)."""
    span = max(abs(value) for value in ax.get_xlim() + ax.get_ylim())
    return 1.0 if span > 1e5 else KM


def _fmt_lon(value):
    return f"{abs(value):.0f}°{'O' if value < 0 else 'L'}"


def _fmt_lat(value):
    return f"{abs(value):.0f}°{'S' if value < 0 else 'N'}"


def draw_graticule(ax):
    """Meridianos e paralelos reprojetados em EPSG:3031, rotulados em graus."""
    scale = _axis_scale(ax)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0, x1 = min(xlim), max(xlim)
    y0, y1 = min(ylim), max(ylim)
    padx, pady = 0.015 * (x1 - x0), 0.015 * (y1 - y0)

    lat_dense = np.linspace(GRATICULE_LAT[0] - 1.5, GRATICULE_LAT[-1] + 1.5, 400)
    lon_dense = np.linspace(GRATICULE_LON[0] - 3.0, GRATICULE_LON[-1] + 3.0, 400)

    for lon in GRATICULE_LON:
        gx, gy = _TO_3031.transform(np.full_like(lat_dense, lon), lat_dense)
        gx, gy = gx * scale, gy * scale
        ax.plot(gx, gy, **GRATICULE_STYLE)
        inside = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
        if inside.any():  # rótulo do meridiano na borda inferior
            j = int(np.argmin(np.where(inside, gy, np.inf)))
            ax.text(gx[j], y0 + pady, _fmt_lon(lon), ha="center", va="bottom",
                    fontsize=7.6, color="#1f2933", zorder=10,
                    bbox=GRATICULE_LABEL_BOX)

    for lat in GRATICULE_LAT:
        gx, gy = _TO_3031.transform(lon_dense, np.full_like(lon_dense, lat))
        gx, gy = gx * scale, gy * scale
        ax.plot(gx, gy, **GRATICULE_STYLE)
        inside = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
        if inside.any():  # rótulo do paralelo na borda esquerda
            j = int(np.argmin(np.where(inside, gx, np.inf)))
            ax.text(x0 + padx, gy[j], _fmt_lat(lat), ha="left", va="center",
                    fontsize=7.6, color="#1f2933", zorder=10,
                    bbox=GRATICULE_LABEL_BOX)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def draw_map_base(ax, cfg, assets, roi_x, roi_y, extent):
    draw_base(ax, cfg, assets, roi_x, roi_y, extent)
    # Graticulado geográfico (lat/lon) reprojetado para EPSG:3031, sobre a
    # moldura cartesiana externa em quilômetros polares.
    ax.grid(False)
    draw_graticule(ax)
    ax.set_xlabel("X polar (km; EPSG:3031)")
    ax.set_ylabel("Y polar (km; EPSG:3031)")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#343434")


def add_two_colorbars(fig, dh_artist, basal_artist, *, animation=False):
    if animation:
        dh_ax = fig.add_axes([0.862, 0.535, 0.022, 0.285])
        basal_ax = fig.add_axes([0.862, 0.175, 0.022, 0.285])
    else:
        dh_ax = fig.add_axes([0.865, 0.555, 0.022, 0.285])
        basal_ax = fig.add_axes([0.865, 0.185, 0.022, 0.285])
    add_dhdt_colorbar(
        fig, dh_artist, cax=dh_ax, label="dh/dt superficial (m/ano)")
    add_basal_colorbar(
        fig, basal_artist, cax=basal_ax,
        label="derretimento basal (m gelo/ano; + = perda pela base)")


def static_dhdt(season, cfg, assets, roi_x, roi_y, extent, surface, output):
    fig, ax = plt.subplots(figsize=(12.0, 9.8), constrained_layout=True)
    draw_map_base(ax, cfg, assets, roi_x, roi_y, extent)
    artist = draw_surface(ax, surface, assets, DHDT_NORM, zorder=7)
    add_dhdt_colorbar(fig, artist, ax=ax)
    ax.set_title(
        f"dh/dt · {SEASON_LABELS[season]} · ATL06/fitsec\n"
        "IDW suavizado · suporte 30 km · escala contínua · "
        "ROI 115°W–95°W, 77,5°S–73°S",
        loc="left", fontweight="bold")
    path = output / f"mapa_dhdt_{season}_continuo_epsg3031.png"
    fig.savefig(path, dpi=STATIC_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def static_joint(season, cfg, assets, roi_x, roi_y, extent,
                 dh_surface, basal_surface, output):
    fig = plt.figure(figsize=(12.0, 8.5), facecolor="white")
    ax = fig.add_axes([0.075, 0.095, 0.745, 0.815])
    draw_map_base(ax, cfg, assets, roi_x, roi_y, extent)
    dh_artist = draw_surface(ax, dh_surface, assets, DHDT_NORM, zorder=7)
    basal_artist = draw_surface(
        ax, basal_surface, assets, BASAL_NORM, zorder=8, cmap=CMAP_BASAL)
    add_two_colorbars(fig, dh_artist, basal_artist)
    fig.suptitle(
        f"dh/dt superficial + derretimento basal · {SEASON_LABELS[season]}",
        fontsize=17, fontweight="semibold", y=0.965)
    fig.text(
        0.46, 0.025,
        "EPSG:3031 · ROI 115°W–95°W, 77,5°S–73°S · paleta RdBu contínua "
        "orientada para vermelho = perda de gelo nos dois campos · "
        "escalas físicas independentes",
        ha="center", fontsize=9, color="#4b5563")
    path = output / f"mapa_conjunto_dhdt_basal_{season}_epsg3031.png"
    fig.savefig(path, dpi=STATIC_DPI, facecolor="white")
    plt.close(fig)
    return path


def draw_timeline(fig, active, labels):
    ax = fig.add_axes([0.235, 0.047, 0.48, 0.060])
    x = np.arange(len(labels), dtype=float)
    ax.plot(x, np.zeros_like(x), color="#b8b8b3", lw=1.2)
    ax.scatter(x, np.zeros_like(x), s=30, color="#d2d2ce", zorder=2)
    ax.scatter([x[active]], [0], s=88, color="#a8481c", zorder=3)
    for idx, label in enumerate(labels):
        ax.text(x[idx], -0.20, label, ha="center", va="top", fontsize=8.2,
                color="#1f2933" if idx == active else "#777772",
                fontweight="semibold" if idx == active else "normal")
    ax.set_xlim(-0.45, len(labels) - 0.55)
    ax.set_ylim(-0.55, 0.30)
    ax.axis("off")


def render_dhdt_frame(season, cfg, assets, roi_x, roi_y, extent,
                      window, active, surface, output):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=FRAME_DPI, facecolor="white")
    ax = fig.add_axes([0.055, 0.145, 0.765, 0.755])
    draw_map_base(ax, cfg, assets, roi_x, roi_y, extent)
    artist = draw_surface(ax, surface, assets, DHDT_NORM, zorder=7)
    cax = fig.add_axes([0.855, 0.245, 0.022, 0.50])
    add_dhdt_colorbar(fig, artist, cax=cax)
    fig.suptitle(
        f"dh/dt · {SEASON_LABELS[season]} · janela {DHDT_LABELS[window]}",
        fontsize=18, fontweight="semibold", y=0.965)
    fig.text(0.44, 0.105,
             "janelas vizinhas compartilham três anos e não são independentes",
             ha="center", fontsize=8.2, color="#6b7280")
    draw_timeline(fig, active, [DHDT_LABELS[w] for w in DHDT_WINDOWS])
    path = output / "quadros" / f"dhdt_{season}_{active + 1:02d}.png"
    fig.savefig(path, dpi=FRAME_DPI, facecolor="white")
    plt.close(fig)
    return path


def render_joint_frame(season, cfg, assets, roi_x, roi_y, extent,
                       dh_window, basal_window, active, dh_surface,
                       basal_surface, fallback_shelves, output):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=FRAME_DPI, facecolor="white")
    ax = fig.add_axes([0.055, 0.145, 0.765, 0.755])
    draw_map_base(ax, cfg, assets, roi_x, roi_y, extent)
    dh_artist = draw_surface(ax, dh_surface, assets, DHDT_NORM, zorder=7)
    basal_artist = draw_surface(
        ax, basal_surface, assets, BASAL_NORM, zorder=8, cmap=CMAP_BASAL)
    add_two_colorbars(fig, dh_artist, basal_artist, animation=True)
    fig.suptitle(
        f"dh/dt + derretimento basal · {SEASON_LABELS[season]}",
        fontsize=18, fontweight="semibold", y=0.972)
    fig.text(
        0.44, 0.915,
        f"dh/dt: {DHDT_LABELS[dh_window]} · basal: "
        f"{basal_window[0]}–{basal_window[1]} · mesmo centro temporal",
        ha="center", fontsize=9.1, color="#374151")
    if fallback_shelves:
        fig.text(
            0.44, 0.885,
            "Fallback basal 2019–2025: " + ", ".join(fallback_shelves),
            ha="center", fontsize=7.7, color="#7a3d21")
    centers = [str(int((d[0] + d[1]) / 2)) for d, _ in JOINT_WINDOWS]
    draw_timeline(fig, active, [f"centro {value}" for value in centers])
    path = output / "quadros" / f"conjunto_{season}_{active + 1:02d}.png"
    fig.savefig(path, dpi=FRAME_DPI, facecolor="white")
    plt.close(fig)
    return path


def write_animation(frames, gif_path, mp4_path):
    durations = [1800] * len(frames)
    durations[-1] = 3000
    images = [Image.open(path).convert("RGB") for path in frames]
    images[0].save(
        gif_path, save_all=True, append_images=images[1:],
        duration=durations, loop=0, disposal=2, optimize=False)
    for image in images:
        image.close()

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"quadro inválido: {frames[0]}")
    height, width = first.shape[:2]
    fps = 15.0
    writer = cv2.VideoWriter(
        str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("codec MP4V indisponível")
    try:
        for index, path in enumerate(frames):
            frame = cv2.imread(str(path))
            seconds = 3.0 if index == len(frames) - 1 else 1.8
            if frame is None or frame.shape[:2] != (height, width):
                raise RuntimeError(f"quadro inválido: {path}")
            for _ in range(round(seconds * fps)):
                writer.write(frame)
    finally:
        writer.release()


def prepare_directories(root):
    paths = {
        "dhdt_static": {},
        "dhdt_animation": {},
        "joint_static": {},
        "joint_animation": {},
    }
    for season in SEASONS:
        paths["dhdt_static"][season] = root / "01_mapas_dhdt" / season
        paths["dhdt_animation"][season] = root / "02_animacoes_dhdt" / season
        paths["joint_static"][season] = root / "03_mapas_conjuntos" / season
        paths["joint_animation"][season] = root / "04_animacoes_conjuntas" / season
        for category in paths:
            paths[category][season].mkdir(parents=True, exist_ok=True)
        (paths["dhdt_animation"][season] / "quadros").mkdir(exist_ok=True)
        (paths["joint_animation"][season] / "quadros").mkdir(exist_ok=True)
    return paths


def main():
    output_root = ROOT / "outputs" / "produtos_sazonais_ase_3031"
    paths = prepare_directories(output_root)
    cfgs = {season: load_config(season) for season in SEASONS}
    log = setup_logging(
        cfgs["jja"].paths.logs,
        level=cfgs["jja"].logging.level,
        run_name="ase_seasonal_continuous_products",
    )
    roi_x, roi_y, roi_path, extent = roi_polygon()
    log.info("Carregando uma base cartográfica comum em EPSG:3031")
    assets = load_base_assets(cfgs["jja"], extent, roi_path, target_px=1400)

    report = {
        "crs": "EPSG:3031",
        "geographic_graticule": {
            "meridians_deg": [float(v) for v in GRATICULE_LON],
            "parallels_deg": [float(v) for v in GRATICULE_LAT],
            "description": (
                "meridianos a cada 5° e paralelos a cada 1°, reprojetados de "
                "EPSG:4326 para EPSG:3031 e rotulados em graus nas bordas "
                "inferior e esquerda; moldura cartesiana externa em km polares"
            ),
        },
        "colormap": {
            "dhdt": CMAP,
            "basal_melt": CMAP_BASAL,
            "convention": (
                "vermelho = perda de gelo em ambos os campos "
                "(dh/dt < 0; derretimento basal > 0)"
            ),
        },
        "continuous_norms": {
            "dhdt_m_per_year": [-3.0, 0.0, 1.0],
            "basal_melt_m_ice_per_year": [-60.0, 0.0, 60.0],
        },
        "dhdt_interpolation": {
            "method": "IDW após fitsec",
            "k": IDW_K,
            "power": IDW_POWER,
            "maximum_support_km": MAX_DHDT_SUPPORT_M / 1000,
            "effective_source_spacing_km": 5,
            "native_display_mask_m": 500,
        },
        "joint_window_alignment": [
            {
                "dhdt": DHDT_LABELS[d],
                "basal": f"{b[0]}–{b[1]}",
                "center_year": int((d[0] + d[1]) / 2),
            }
            for d, b in JOINT_WINDOWS
        ],
        "seasons": {},
        "caveats": [
            "A máscara de 500 m e a exportação em alta resolução não transformam o dh/dt fitsec de 5 km em observação de 500 m.",
            "Janelas móveis vizinhas de dh/dt compartilham três de quatro anos e não constituem estimativas independentes de aceleração.",
            "As janelas basais disponíveis têm três anos; nas animações conjuntas elas são alinhadas às janelas de dh/dt pelo centro temporal.",
            "Os verões DJF das extremidades do registro são parciais; dezembro pertence ao verão seguinte em análises agregadas por ano sazonal.",
            "As duas barras de cor compartilham a família RdBu com orientações opostas: a leitura cromática é comparável, mas as escalas físicas não são.",
        ],
    }

    for season in SEASONS:
        cfg = cfgs[season]
        log.info(f"[{season.upper()}] carregando dh/dt completo e janelas")
        full_nodes = load_dhdt_full(cfg, roi_path)
        window_nodes = load_dhdt_windows(cfg, roi_path, full_nodes)
        log.info(f"[{season.upper()}] reamostrando dh/dt completo")
        dh_full_surface = dhdt_to_native(full_nodes, assets)
        dh_window_surfaces = {}
        for window in DHDT_WINDOWS:
            log.info(f"[{season.upper()}] reamostrando dh/dt {DHDT_LABELS[window]}")
            dh_window_surfaces[window] = dhdt_to_native(
                window_nodes[window], assets)

        log.info(f"[{season.upper()}] preparando derretimento basal")
        (basal_full, basal_windows, _observed, _global_cells,
         basal_winner, basal_cv, fallbacks) = basal_fields(cfg, roi_path, extent)
        basal_full_surface = basal_to_native(basal_full, assets)
        needed_basal = sorted({b for _, b in JOINT_WINDOWS})
        basal_window_surfaces = {}
        for window in needed_basal:
            log.info(f"[{season.upper()}] reamostrando basal {window[0]}–{window[1]}")
            basal_window_surfaces[window] = basal_to_native(
                basal_windows[window], assets)

        log.info(f"[{season.upper()}] mapas estáticos")
        dh_static = static_dhdt(
            season, cfg, assets, roi_x, roi_y, extent,
            dh_full_surface, paths["dhdt_static"][season])
        joint_static_path = static_joint(
            season, cfg, assets, roi_x, roi_y, extent,
            dh_full_surface, basal_full_surface,
            paths["joint_static"][season])

        log.info(f"[{season.upper()}] animação dh/dt")
        dh_frames = []
        for index, window in enumerate(DHDT_WINDOWS):
            dh_frames.append(render_dhdt_frame(
                season, cfg, assets, roi_x, roi_y, extent,
                window, index, dh_window_surfaces[window],
                paths["dhdt_animation"][season]))
        dh_gif = paths["dhdt_animation"][season] / f"dhdt_{season}_janelas_moveis.gif"
        dh_mp4 = paths["dhdt_animation"][season] / f"dhdt_{season}_janelas_moveis.mp4"
        write_animation(dh_frames, dh_gif, dh_mp4)

        log.info(f"[{season.upper()}] animação conjunta")
        joint_frames = []
        for index, (dh_window, basal_window) in enumerate(JOINT_WINDOWS):
            joint_frames.append(render_joint_frame(
                season, cfg, assets, roi_x, roi_y, extent,
                dh_window, basal_window, index,
                dh_window_surfaces[dh_window],
                basal_window_surfaces[basal_window],
                fallbacks.get(basal_window, []),
                paths["joint_animation"][season]))
        joint_gif = (
            paths["joint_animation"][season]
            / f"dhdt_basal_{season}_janelas_moveis.gif")
        joint_mp4 = (
            paths["joint_animation"][season]
            / f"dhdt_basal_{season}_janelas_moveis.mp4")
        write_animation(joint_frames, joint_gif, joint_mp4)

        grounded = (assets["high_mask"] == 2) & assets["native_roi"]
        floating = (assets["high_mask"] == 3) & assets["native_roi"]
        report["seasons"][season] = {
            "label": SEASON_LABELS[season],
            "dhdt_nodes_full": int(len(full_nodes)),
            "dhdt_window_nodes": {
                DHDT_LABELS[w]: int(len(window_nodes[w])) for w in DHDT_WINDOWS
            },
            "dhdt_coverage_fraction": float(
                np.isfinite(dh_full_surface).sum() / grounded.sum()),
            "basal_coverage_fraction": float(
                np.isfinite(basal_full_surface).sum() / floating.sum()),
            "basal_idw_cv_winner": basal_winner,
            "basal_idw_cv_candidates": basal_cv.to_dict(orient="records"),
            "outputs": {
                "dhdt_static": str(dh_static),
                "dhdt_gif": str(dh_gif),
                "dhdt_mp4": str(dh_mp4),
                "joint_static": str(joint_static_path),
                "joint_gif": str(joint_gif),
                "joint_mp4": str(joint_mp4),
            },
        }
        log.info(f"[{season.upper()}] concluído")

    report_path = output_root / "relatorio_produtos_sazonais_ase_3031.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Relatório -> {report_path}")


if __name__ == "__main__":
    main()