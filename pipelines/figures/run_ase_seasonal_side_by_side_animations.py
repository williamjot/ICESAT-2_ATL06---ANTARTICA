"""Animações sazonais lado a lado: JJA e DJF no mesmo quadro.

Dois produtos, ambos em GIF:

dh/dt
    quatro janelas móveis de quatro anos (2019–2023 … 2022–2025,7), com JJA
    à esquerda e DJF à direita, escala de cor compartilhada.
derretimento basal
    cinco janelas móveis de três anos (2019–2021 … 2023–2025), mesmo arranjo.

O campo é desenhado na resolução em que foi estimado — a grade de 5 km dos
nós fitsec (dh/dt) e das células de plataforma (basal). Não há reamostragem
para 500 m: aqui a célula visível é a célula observada.

Os dois GIFs compartilham a mesma moldura cartográfica, para que a comparação
entre eles seja geométrica e não só cromática.

O layout é deliberadamente enxuto — dois títulos de painel, uma barra de cor
e a linha do tempo. Sem título geral, sem graticulado, sem barra de escala,
sem notas de rodapé. As ressalvas metodológicas (janelas que compartilham
anos, plataformas em fallback temporal) ficam no relatório JSON, não no
quadro; ver a seção "caveats".
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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.viz.produtos import carregar_bedmachine, para_grade

from pipelines.run_produto_figuras import (
    DIV_MELT,
    DIV_THIN,
    INK,
    KM,
    contorno_gl,
    eixo_mapa,
)
from pipelines.run_ase_jja_basal_dhdt_products_500m import (
    WINDOWS as BASAL_WINDOWS,
    basal_fields,
)
from pipelines.run_ase_jja_diagnostic_maps import roi_polygon
from pipelines.run_ase_seasonal_continuous_products import (
    DHDT_LABELS,
    DHDT_WINDOWS,
    SEASONS,
    SEASON_LABELS,
    load_dhdt_full,
    load_dhdt_windows,
)

DHDT_NORM = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=1.0)
BASAL_NORM = TwoSlopeNorm(vmin=-60.0, vcenter=0.0, vmax=60.0)
PAD_M = 8_000.0
DPI = 150
DHDT_HOLD_MS = (2100, 2100, 2100, 3200)
BASAL_HOLD_MS = (1900, 1900, 1900, 1900, 3000)
BASAL_LABELS = tuple(f"{start}–{end}" for start, end in BASAL_WINDOWS)


def map_extent_km(frames):
    """Moldura comum: envelope dos dados desenhados, com folga de 8 km."""
    xs = np.concatenate([frame.x.to_numpy() for frame in frames])
    ys = np.concatenate([frame.y.to_numpy() for frame in frames])
    return ((xs.min() - PAD_M) * KM, (xs.max() + PAD_M) * KM,
            (ys.min() - PAD_M) * KM, (ys.max() + PAD_M) * KM)


def load_bedmachine_mask(cfg, extent_km):
    candidates = sorted(cfg.paths.data_dir.glob("*BedMachine*.nc"))
    if not candidates:
        raise FileNotFoundError(
            f"BedMachine não encontrado em {cfg.paths.data_dir}")
    return carregar_bedmachine(
        candidates[0],
        extent_km[0] / KM, extent_km[1] / KM,
        extent_km[2] / KM, extent_km[3] / KM,
        vars=("mask",))


def draw_timeline(fig, active, labels):
    ax = fig.add_axes([0.26, 0.032, 0.48, 0.058])
    x = np.arange(len(labels), dtype=float)
    ax.plot(x, np.zeros_like(x), color="#b8b8b3", lw=1.3, zorder=1)
    ax.scatter(x, np.zeros_like(x), s=38, color="#d2d2ce", zorder=2)
    ax.scatter([x[active]], [0], s=94, color="#a8481c", zorder=3)
    for index, label in enumerate(labels):
        ax.text(x[index], -0.20, label, ha="center", va="top", fontsize=8.2,
                color=INK if index == active else "#777772",
                fontweight="semibold" if index == active else "normal")
    ax.set_xlim(-0.35, len(labels) - 0.65)
    ax.set_ylim(-0.55, 0.30)
    ax.axis("off")


def render_frame(fields, column, cmap, norm, colorbar_label, bedmachine,
                 extent_km, active, labels, path):
    """Um quadro 1920×1080 com JJA à esquerda e DJF à direita."""
    bx, by, bed = bedmachine
    fig = plt.figure(figsize=(12.8, 7.2), dpi=DPI, facecolor="#fcfcfb")
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.045), left=0.055,
                            right=0.945, bottom=0.150, top=0.935, wspace=0.10)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    color_ax = fig.add_subplot(grid[0, 2])
    image = None

    for index, season in enumerate(SEASONS):
        ax = axes[index]
        # Painéis ancorados no topo: sem título geral, o espaço livre da
        # figura fica embaixo, junto da linha do tempo.
        ax.set_anchor("N")
        xs, ys, field = para_grade(fields[season], column)
        image = ax.pcolormesh(xs * KM, ys * KM, field, cmap=cmap, norm=norm,
                              shading="auto", rasterized=True)
        contorno_gl(ax, bx, by, bed["mask"])
        eixo_mapa(ax, extent_km)
        ax.set_title(SEASON_LABELS[season], fontsize=13,
                     fontweight="semibold", pad=9)
        if index == 1:
            ax.set_ylabel("")

    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.set_label(colorbar_label, fontsize=10.5)
    colorbar.ax.axhline(0, color=INK, lw=0.9)

    draw_timeline(fig, active, labels)
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_gif(frames, durations, path):
    images = [Image.open(frame).convert("RGB") for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=list(durations), loop=0, disposal=2,
                   optimize=False)
    for image in images:
        image.close()
    return path


def load_dhdt(cfgs, roi_path, log):
    """Nós fitsec de 5 km recortados ao suporte QC aterrado da ROI."""
    nodes, support = {}, {}
    for season in SEASONS:
        full = load_dhdt_full(cfgs[season], roi_path)
        support[season] = int(len(full))
        nodes[season] = load_dhdt_windows(cfgs[season], roi_path, full)
        log.info(f"[{season.upper()}] dh/dt: {support[season]} nós no suporte")
    return nodes, support


def load_basal(cfgs, roi_path, extent, log):
    """Células de plataforma de 5 km, uma partição espacial por plataforma."""
    fields, fallbacks, selection = {}, {}, {}
    for season in SEASONS:
        log.info(f"[{season.upper()}] derretimento basal: janelas móveis")
        _full, windows, _observed, _cells, winner, _cv, season_fallbacks = (
            basal_fields(cfgs[season], roi_path, extent))
        fields[season] = windows
        fallbacks[season] = season_fallbacks
        selection[season] = dict(winner)
    return fields, fallbacks, selection


def animate(fields_by_window, windows, labels, column, cmap, norm, label,
            bedmachine, extent_km, holds, stem, output, log):
    frames_dir = output / "quadros"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for active, window in enumerate(windows):
        log.info(f"quadro {stem} {labels[active]}")
        frames.append(render_frame(
            {season: fields_by_window[season][window] for season in SEASONS},
            column, cmap, norm, label, bedmachine, extent_km, active,
            list(labels), frames_dir / f"{stem}_{active + 1:02d}.png"))
    gif = write_gif(frames, holds, output / f"{stem}_janelas_moveis.gif")
    return gif, frames


def main():
    root = ROOT / "outputs" / "produtos_sazonais_ase_3031"
    dhdt_dir = root / "02_animacoes_dhdt" / "conjunta_sazonal"
    basal_dir = root / "04_animacoes_conjuntas" / "conjunta_sazonal"
    for directory in (dhdt_dir, basal_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cfgs = {season: load_config(season) for season in SEASONS}
    log = setup_logging(cfgs["jja"].paths.logs, level=cfgs["jja"].logging.level,
                        run_name="ase_seasonal_side_by_side_animations")
    _roi_x, _roi_y, roi_path, extent = roi_polygon()

    dhdt_nodes, dhdt_support = load_dhdt(cfgs, roi_path, log)
    basal_cells, fallbacks, selection = load_basal(cfgs, roi_path, extent, log)

    extent_km = map_extent_km(
        [dhdt_nodes[season][window]
         for season in SEASONS for window in DHDT_WINDOWS] +
        [basal_cells[season][window]
         for season in SEASONS for window in BASAL_WINDOWS])
    bedmachine = load_bedmachine_mask(cfgs["jja"], extent_km)

    dhdt_gif, dhdt_frames = animate(
        dhdt_nodes, DHDT_WINDOWS,
        tuple(DHDT_LABELS[window] for window in DHDT_WINDOWS), "dhdt",
        DIV_THIN, DHDT_NORM, "dh/dt (m ano⁻¹)", bedmachine, extent_km,
        DHDT_HOLD_MS, "dhdt_jja_djf", dhdt_dir, log)
    basal_gif, basal_frames = animate(
        basal_cells, BASAL_WINDOWS, BASAL_LABELS, "basal_melt",
        DIV_MELT, BASAL_NORM, "derretimento basal (m gelo ano⁻¹)", bedmachine,
        extent_km, BASAL_HOLD_MS, "derretimento_basal_jja_djf", basal_dir, log)

    report = {
        "crs": "EPSG:3031",
        "roi": "115°W–95°W; 77,5°S–73°S",
        "layout": (
            "JJA à esquerda e DJF à direita, escala de cor compartilhada, "
            "linha do tempo discreta; sem título geral, graticulado, barra "
            "de escala ou rodapé"
        ),
        "rendering_resolution_m": 5000,
        "shared_extent_km": [float(value) for value in extent_km],
        "colormaps": {
            "dhdt": "DIV_THIN (adelgaçamento em vermelho)",
            "basal_melt": "DIV_MELT (derretimento em vermelho)",
        },
        "norms": {
            "dhdt_m_per_year": [-3.0, 0.0, 1.0],
            "basal_melt_m_ice_per_year": [-60.0, 0.0, 60.0],
        },
        "dhdt": {
            "gif": str(dhdt_gif),
            "frames": [str(frame) for frame in dhdt_frames],
            "windows": [DHDT_LABELS[window] for window in DHDT_WINDOWS],
            "support_nodes": dhdt_support,
            "nodes_per_window": {
                season: {
                    DHDT_LABELS[window]: int(len(dhdt_nodes[season][window]))
                    for window in DHDT_WINDOWS
                }
                for season in SEASONS
            },
        },
        "basal_melt": {
            "gif": str(basal_gif),
            "frames": [str(frame) for frame in basal_frames],
            "windows": list(BASAL_LABELS),
            "cells_per_window": {
                season: {
                    label: int(np.isfinite(
                        basal_cells[season][window].basal_melt).sum())
                    for label, window in zip(BASAL_LABELS, BASAL_WINDOWS)
                }
                for season in SEASONS
            },
            "idw_selection": {
                season: {key: float(value)
                         for key, value in selection[season].items()}
                for season in SEASONS
            },
            # Plataformas sem estimativa própria na janela: o quadro mostra a
            # mediana espacial 2019–2025 e o layout enxuto não sinaliza isso.
            "temporal_fallback_shelves": {
                season: {label: fallbacks[season][window]
                         for label, window in zip(BASAL_LABELS, BASAL_WINDOWS)}
                for season in SEASONS
            },
        },
        "caveats": [
            "Janelas vizinhas de dh/dt compartilham três dos quatro anos e as de derretimento basal dois dos três; a sequência não mede aceleração.",
            "As duas séries têm comprimentos de janela diferentes (4 anos em dh/dt, 3 no basal) e não estão alinhadas entre os dois GIFs.",
            "A janela final de dh/dt termina em 2025,7: é mais curta que as demais, e o rótulo 2022–2025,7 registra isso.",
            "Os verões DJF das extremidades do registro são parciais.",
            "Plataformas sem estimativa basal própria na janela usam a mediana espacial 2019–2025; o layout enxuto não as sinaliza no quadro, só em temporal_fallback_shelves.",
            "O derretimento basal é um balanço inferido no gelo flutuante, com incerteza estrutural não capturada pela escala de cor.",
            "dh/dt sem correção de GIA e sem propagação formal de incerteza nestes quadros.",
        ],
    }
    report_path = root / "relatorio_animacoes_sazonais_lado_a_lado.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    log.info(f"GIF dh/dt -> {dhdt_gif}")
    log.info(f"GIF basal -> {basal_gif}")
    log.info(f"Relatório -> {report_path}")


if __name__ == "__main__":
    main()
