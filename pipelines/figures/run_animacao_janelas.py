"""Gera animações comparáveis de dh/dt para JJA e DJF.

Modos disponíveis
-----------------
moveis
    Janelas de quatro anos com passo anual: 2019–2023, 2020–2024,
    2021–2025 e 2022–2025,7.
inicio_fixo
    Períodos expansivos com início em 2019: 2019–2023, 2019–2024,
    2019–2025 e 2019–2025,7.

Os dois modos usam a mesma extensão, máscara, QC e escala de cores. Quadros
consecutivos reutilizam observações e não constituem estimativas independentes
de aceleração.

Uso
---
python pipelines/run_animacao_janelas.py --modo moveis
python pipelines/run_animacao_janelas.py --modo inicio_fixo
"""

from __future__ import annotations

import argparse
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

from pipelines.run_produto_figuras import (
    DIV_THIN,
    INK,
    KM,
    contorno_gl,
    eixo_mapa,
)

SEASONS = ("jja", "djf")
SEASON_LABEL = {
    "jja": "Inverno austral (JJA)",
    "djf": "Verão austral (DJF)",
}

MODES = {
    "moveis": {
        "windows": ("2019-2023", "2020-2024", "2021-2025", "2022-2026"),
        "labels": {
            "2019-2023": "2019–2023",
            "2020-2024": "2020–2024",
            "2021-2025": "2021–2025",
            "2022-2026": "2022–2025,7",
        },
        "data_subdir": "janelas",
        "report": "dhdt_janelas.json",
        "output_subdir": "animacao_janelas_moveis",
        "stem": "F6_dhdt_janelas_moveis",
    },
    "inicio_fixo": {
        "windows": ("2019-2023", "2019-2024", "2019-2025", "2019-2026"),
        "labels": {
            "2019-2023": "2019–2023",
            "2019-2024": "2019–2024",
            "2019-2025": "2019–2025",
            "2019-2026": "2019–2025,7",
        },
        "data_subdir": "janelas_inicio_fixo",
        "report": "dhdt_janelas_inicio_fixo.json",
        "output_subdir": "animacao_inicio_fixo",
        "stem": "F6_dhdt_inicio_fixo",
    },
}


def _load_window_data(mode: dict) -> tuple[dict, dict]:
    """Carrega as janelas e as recorta aos nós aceitos no QC sazonal."""
    configs = {"jja": load_config(), "djf": load_config("djf")}
    data: dict[str, dict[str, pd.DataFrame]] = {season: {} for season in SEASONS}

    for season, cfg in configs.items():
        report_path = cfg.paths.tables / mode["report"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        qc = pd.read_parquet(
            cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet", columns=["x", "y"]
        )
        accepted = set(
            zip(
                np.rint(qc["x"]).astype(np.int64),
                np.rint(qc["y"]).astype(np.int64),
            )
        )

        for window in mode["windows"]:
            info = report["janelas"][window]
            nodes = pd.read_parquet(
                cfg.paths.dhdt_dir / mode["data_subdir"] / info["arquivo"],
                columns=["x", "y", "dhdt"],
            )
            keep = np.fromiter(
                (
                    (x, y) in accepted
                    for x, y in zip(
                        np.rint(nodes["x"]).astype(np.int64),
                        np.rint(nodes["y"]).astype(np.int64),
                    )
                ),
                dtype=bool,
                count=len(nodes),
            )
            data[season][window] = nodes.loc[keep].copy()

    return configs, data


def _spatial_context(configs: dict) -> tuple[np.ndarray, np.ndarray, dict, tuple]:
    """Usa a mesma extensão espacial da figura comparativa estática."""
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
    bx, by, bed = carregar_bedmachine(candidates[0], x0, x1, y0, y1)
    return bx, by, bed, extent_km


def _draw_timeline(
    fig: plt.Figure, active: int, windows: tuple[str, ...], labels: dict[str, str]
) -> None:
    """Desenha marcadores discretos, sem sugerir interpolação temporal."""
    ax = fig.add_axes([0.26, 0.030, 0.48, 0.060])
    x = np.arange(len(windows), dtype=float)
    ax.plot(x, np.zeros_like(x), color="#b8b8b3", lw=1.3, zorder=1)
    ax.scatter(x, np.zeros_like(x), s=38, color="#d2d2ce", zorder=2)
    ax.scatter([x[active]], [0], s=94, color="#a8481c", zorder=3)
    for idx, window in enumerate(windows):
        ax.text(
            x[idx],
            -0.20,
            labels[window],
            ha="center",
            va="top",
            fontsize=8.2,
            color=INK if idx == active else "#777772",
            fontweight="semibold" if idx == active else "normal",
        )
    ax.set_xlim(-0.35, len(windows) - 0.65)
    ax.set_ylim(-0.55, 0.30)
    ax.axis("off")


def _render_frame(
    window: str,
    active: int,
    data: dict,
    bx: np.ndarray,
    by: np.ndarray,
    bed: dict,
    extent_km: tuple,
    mode: dict,
    frames_dir: Path,
) -> Path:
    """Renderiza um quadro 1920×1080 com JJA e DJF lado a lado."""
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor="#fcfcfb")
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=(1, 1, 0.045),
        left=0.055,
        right=0.945,
        bottom=0.170,
        top=0.885,
        wspace=0.10,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    color_ax = fig.add_subplot(gs[0, 2])
    image = None

    for column, season in enumerate(SEASONS):
        ax = axes[column]
        xs, ys, field = para_grade(data[season][window], "dhdt")
        image = ax.pcolormesh(
            xs * KM,
            ys * KM,
            field,
            cmap=DIV_THIN,
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-3.0, vmax=1.0),
            shading="auto",
            rasterized=True,
        )
        contorno_gl(ax, bx, by, bed["mask"])
        eixo_mapa(ax, extent_km)
        ax.set_title(SEASON_LABEL[season], fontsize=13, fontweight="semibold", pad=9)
        if column == 1:
            ax.set_ylabel("")

    assert image is not None
    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.set_label("dh/dt (m ano⁻¹)", fontsize=10.5)
    colorbar.ax.axhline(0, color=INK, lw=0.9)

    fig.suptitle(
        f"Mudança espacial da taxa de elevação · {mode['labels'][window]}",
        fontsize=19,
        fontweight="semibold",
        y=0.965,
        color=INK,
    )
    _draw_timeline(fig, active, mode["windows"], mode["labels"])

    path = frames_dir / f"{mode['stem']}_quadro_{active + 1:02d}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _write_gif(frame_paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=[2100, 2100, 2100, 3200],
        loop=0,
        disposal=2,
        optimize=False,
    )
    for image in images:
        image.close()


def _write_mp4(frame_paths: list[Path], output: Path) -> None:
    """Cria MP4 sem depender de executável ffmpeg externo."""
    fps = 15.0
    holds = (2.1, 2.1, 2.1, 3.2)
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"não foi possível abrir {frame_paths[0]}")
    height, width = first.shape[:2]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Animações temporais de dh/dt.")
    parser.add_argument("--modo", choices=tuple(MODES), default="moveis")
    args = parser.parse_args()
    mode = MODES[args.modo]

    out = ROOT / "outputs" / "comparacao_sazonal" / mode["output_subdir"]
    frames_dir = out / "quadros"
    out.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Carregando dados do modo {args.modo} e nós de QC...")
    configs, data = _load_window_data(mode)
    print("Carregando contexto cartográfico...")
    bx, by, bed, extent_km = _spatial_context(configs)

    frame_paths = []
    for index, window in enumerate(mode["windows"]):
        print(f"Renderizando {window}...")
        frame_paths.append(
            _render_frame(
                window, index, data, bx, by, bed, extent_km, mode, frames_dir
            )
        )

    gif = out / f"{mode['stem']}.gif"
    mp4 = out / f"{mode['stem']}.mp4"
    _write_gif(frame_paths, gif)
    _write_mp4(frame_paths, mp4)
    print(f"GIF -> {gif}")
    print(f"MP4 -> {mp4}")
    print(f"Quadros -> {frames_dir}")


if __name__ == "__main__":
    main()
