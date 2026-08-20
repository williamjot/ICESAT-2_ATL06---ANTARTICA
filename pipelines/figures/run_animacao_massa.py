"""
Gera uma animação 16:9 da evolução da massa acumulada da Figura F9.

Cada quadro compara JJA e DJF no mesmo ano, de 2019 a 2025. Para impedir que
mudanças de cobertura sejam confundidas com mudanças físicas, os mapas de cada
estação usam somente nós presentes em TODOS os sete anos — o mesmo suporte
espacial usado para calcular a série regional de massa.

A massa é inferida da anomalia de altura multiplicada pela densidade do gelo
(917 kg m⁻³); não é uma medida gravimétrica. A escala cartográfica permanece
fixa em toda a animação.

Entradas
--------
data/<estacao>/dhdt/serie_anual.parquet
outputs/<estacao>/tables/serie_massa.json

Saídas
------
outputs/comparacao_sazonal/animacao_massa/F9_animacao_massa.gif
outputs/comparacao_sazonal/animacao_massa/F9_animacao_massa.mp4
outputs/comparacao_sazonal/animacao_massa/quadros/*.png

Uso: python pipelines/run_animacao_massa.py
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.viz.produtos import carregar_bedmachine, para_grade

# Mesma convenção visual e cartográfica da F9 estática.
from pipelines.run_figuras_massa import (
    DIV_THIN,
    INK,
    INK2,
    KM,
    contornos,
)


OUT = ROOT / "outputs" / "comparacao_sazonal" / "animacao_massa"
FRAMES = OUT / "quadros"
SEASONS = ("jja", "djf")
SEASON_LABEL = {
    "jja": "Inverno austral (JJA)",
    "djf": "Verão austral (DJF)",
}
RHO_ICE = 917.0
RHO_WATER = 1000.0


def _load_data() -> tuple[dict, dict, list[int]]:
    """Carrega as séries e fixa o suporte espacial dentro de cada estação."""
    configs = {"jja": load_config(), "djf": load_config("djf")}
    maps: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    common_years: set[int] | None = None

    for season, cfg in configs.items():
        series = pd.read_parquet(
            cfg.paths.dhdt_dir / "serie_anual.parquet",
            columns=["x", "y", "ano", "anom"],
        )
        years = {int(year) for year in series["ano"].unique()}
        common_years = years if common_years is None else common_years & years

        # A série regional usa apenas nós vistos em todos os anos. Repetimos o
        # mesmo critério no mapa animado para manter a cobertura constante.
        n_years = len(years)
        counts = series.groupby(["x", "y"])["ano"].nunique()
        complete = counts[counts == n_years].index
        indexed = series.set_index(["x", "y"])
        series = indexed.loc[indexed.index.isin(complete)].reset_index()
        series["mwe"] = series["anom"] * RHO_ICE / RHO_WATER
        maps[season] = series

        report = json.loads(
            (cfg.paths.tables / "serie_massa.json").read_text(encoding="utf-8")
        )
        report["por_ano"] = {int(item["ano"]): item for item in report["serie"]}
        reports[season] = report

        if series.groupby(["x", "y"])["ano"].nunique().nunique() != 1:
            raise RuntimeError(f"o suporte espacial de {season} não ficou constante")
        if series.groupby(["x", "y"]).ngroups != report["n_nos_completos"]:
            raise RuntimeError(
                f"nós completos de {season} divergem do relatório de massa"
            )

    if common_years is None:
        raise RuntimeError("nenhum ano encontrado nas séries de massa")
    return configs, maps, sorted(common_years)


def _spatial_context(configs: dict) -> tuple[np.ndarray, np.ndarray, dict, tuple]:
    """Reproduz a extensão espacial e o recorte do BedMachine usados na F9."""
    grid = pd.read_parquet(
        configs["jja"].paths.interim / "dhdt_grid.parquet", columns=["x", "y"]
    )
    x0, x1 = grid["x"].min() - 8e3, grid["x"].max() + 8e3
    y0, y1 = grid["y"].min() - 8e3, grid["y"].max() + 8e3
    extent_km = (x0 * KM, x1 * KM, y0 * KM, y1 * KM)

    candidates = sorted(configs["jja"].paths.data_dir.glob("*BedMachine*.nc"))
    if not candidates:
        raise FileNotFoundError(
            f"BedMachine não encontrado em {configs['jja'].paths.data_dir}"
        )
    bx, by, bed = carregar_bedmachine(
        candidates[0], x0, x1, y0, y1, vars=("mask",)
    )
    return bx, by, bed, extent_km


def _draw_timeline(fig: plt.Figure, years: list[int], active: int) -> None:
    ax = fig.add_axes([0.22, 0.071, 0.56, 0.055])
    x = np.arange(len(years), dtype=float)
    ax.plot(x, np.zeros_like(x), color="#b8b8b3", lw=1.3, zorder=1)
    ax.scatter(x, np.zeros_like(x), s=34, color="#d2d2ce", zorder=2)
    ax.scatter([x[active]], [0], s=94, color="#a8481c", zorder=3)
    for index, year in enumerate(years):
        ax.text(
            x[index],
            -0.20,
            str(year),
            ha="center",
            va="top",
            fontsize=8.2,
            color=INK if index == active else "#777772",
            fontweight="semibold" if index == active else "normal",
        )
    ax.set_xlim(-0.45, len(years) - 0.55)
    ax.set_ylim(-0.55, 0.30)
    ax.axis("off")


def _render_frame(
    year: int,
    active: int,
    years: list[int],
    maps: dict,
    bx: np.ndarray,
    by: np.ndarray,
    bed: dict,
    extent_km: tuple,
) -> Path:
    """Renderiza um quadro 1920×1080 com o acumulado JJA e DJF."""
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor="#fcfcfb")
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=(1, 1, 0.045),
        left=0.055,
        right=0.945,
        bottom=0.205,
        top=0.835,
        wspace=0.10,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    color_ax = fig.add_subplot(gs[0, 2])
    image = None

    for column, season in enumerate(SEASONS):
        ax = axes[column]
        annual = maps[season].loc[maps[season]["ano"] == year]
        xs, ys, field = para_grade(annual, "mwe")
        image = ax.pcolormesh(
            xs * KM,
            ys * KM,
            field,
            cmap=DIV_THIN,
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-8.0, vmax=3.0),
            shading="auto",
            rasterized=True,
        )
        contornos(ax, bx, by, bed["mask"])
        ax.set_xlim(*extent_km[:2])
        ax.set_ylim(*extent_km[2:])
        ax.set_aspect("equal")
        ax.set_xlabel("x (km, EPSG:3031)")
        ax.set_ylabel("y (km)")
        ax.set_title(SEASON_LABEL[season], fontsize=13, fontweight="semibold", pad=9)
        if column == 1:
            ax.set_ylabel("")

    assert image is not None
    colorbar = fig.colorbar(image, cax=color_ax, extend="both")
    colorbar.set_label("variação acumulada (m de água equivalente)", fontsize=10.3)
    colorbar.ax.axhline(0, color=INK, lw=0.9)

    fig.suptitle(
        f"Evolução da perda de massa · 2019 → {year}",
        fontsize=19,
        fontweight="semibold",
        y=0.965,
        color=INK,
    )
    fig.text(
        0.5,
        0.910,
        "Anomalia relativa a 2019 · suporte espacial fixo · mesma escala em todos os quadros",
        ha="center",
        fontsize=10.7,
        color=INK2,
    )
    _draw_timeline(fig, years, active)
    fig.text(
        0.5,
        0.019,
        "Massa INFERIDA de variação de altura × 917 kg m⁻³, não medida por gravimetria; "
        "a propagação formal de incerteza ainda não está incorporada.",
        ha="center",
        fontsize=8.8,
        color="#a8481c",
    )

    path = FRAMES / f"F9_quadro_{active + 1:02d}_{year}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _write_gif(frame_paths: list[Path]) -> Path:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    output = OUT / "F9_animacao_massa.gif"
    durations = [1500] * len(images)
    durations[0] = 1800
    durations[-1] = 2800
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False,
    )
    for image in images:
        image.close()
    return output


def _write_mp4(frame_paths: list[Path]) -> Path:
    """Cria MP4 sem depender de um executável ffmpeg externo."""
    fps = 15.0
    holds = [1.5] * len(frame_paths)
    holds[0] = 1.8
    holds[-1] = 2.8

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"não foi possível abrir {frame_paths[0]}")
    height, width = first.shape[:2]
    output = OUT / "F9_animacao_massa.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("o codec MP4V não pôde ser inicializado pelo OpenCV")
    try:
        for path, seconds in zip(frame_paths, holds):
            frame = cv2.imread(str(path))
            if frame is None or frame.shape[:2] != (height, width):
                raise RuntimeError(f"quadro inválido: {path}")
            for _ in range(round(seconds * fps)):
                writer.write(frame)
    finally:
        writer.release()
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)

    print("Carregando séries anuais e fixando o suporte espacial...")
    configs, maps, years = _load_data()
    print("Carregando contexto cartográfico...")
    bx, by, bed, extent_km = _spatial_context(configs)

    frame_paths = []
    for index, year in enumerate(years):
        print(f"Renderizando acumulado até {year}...")
        frame_paths.append(
            _render_frame(
                year,
                index,
                years,
                maps,
                bx,
                by,
                bed,
                extent_km,
            )
        )

    gif = _write_gif(frame_paths)
    mp4 = _write_mp4(frame_paths)
    print(f"GIF -> {gif}")
    print(f"MP4 -> {mp4}")
    print(f"Quadros -> {FRAMES}")


if __name__ == "__main__":
    main()
