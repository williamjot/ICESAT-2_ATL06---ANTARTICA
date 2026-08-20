"""
Gera histogramas separados de dh/dt para inverno (JJA) e verão (DJF).

As barras usam a mesma paleta divergente dos mapas de dh/dt: vermelho para
afinamento, cinza no zero e azul para espessamento. Os dois painéis usam os
mesmos limites e classes de 0,1 m/ano, permitindo comparação direta.

Entradas
--------
data/<estacao>/dhdt/dhdt_nodes_qc.parquet

Saídas
------
outputs/comparacao_sazonal/histograma_dhdt_inverno.{png,svg}
outputs/comparacao_sazonal/histograma_dhdt_verao.{png,svg}

Uso: python pipelines/run_histogramas_dhdt_individuais.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from pipelines.run_produto_figuras import DIV_THIN


OUT = ROOT / "outputs" / "comparacao_sazonal"
INK = "#1A1A1A"
INK2 = "#4A4A4A"
MUTED = "#777777"
GRID = "#D9D9D6"
BG = "#FCFCFB"
STAT = "#0D47A1"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "legend.frameon": True,
        "svg.fonttype": "none",
    }
)


def _load() -> dict[str, np.ndarray]:
    configs = {"jja": load_config(), "djf": load_config("djf")}
    result = {}
    for season, cfg in configs.items():
        values = pd.read_parquet(
            cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet", columns=["dhdt"]
        )["dhdt"].to_numpy(float)
        result[season] = values[np.isfinite(values)]
    return result


def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    svg = OUT / f"{stem}.svg"
    fig.savefig(png, dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(svg, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png, svg


def _histogram(
    values: np.ndarray,
    bins: np.ndarray,
    xlim: tuple[float, float],
    season_label: str,
    season_code: str,
    stem: str,
) -> tuple[Path, Path]:
    mean = float(np.mean(values))
    median = float(np.median(values))
    density, edges = np.histogram(values, bins=bins, density=True)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-3.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    fig.subplots_adjust(left=0.095, right=0.965, bottom=0.145, top=0.82)

    for index, height in enumerate(density):
        left, right = edges[index], edges[index + 1]
        midpoint = 0.5 * (left + right)
        color = DIV_THIN(np.clip(norm(midpoint), 0.0, 1.0))
        ax.bar(
            left,
            height,
            width=right - left,
            align="edge",
            color=color,
            edgecolor="#333333",
            linewidth=0.48,
            zorder=2,
        )

    ax.axvline(0, color=INK, linewidth=1.0, linestyle="--", alpha=0.65, zorder=3)
    mean_line = ax.axvline(
        mean,
        color=STAT,
        linewidth=2.2,
        linestyle="-",
        label=f"média {mean:+.3f} m/ano",
        zorder=4,
    )
    median_line = ax.axvline(
        median,
        color=STAT,
        linewidth=1.8,
        linestyle=":",
        label=f"mediana {median:+.3f} m/ano",
        zorder=4,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Taxa de mudança de elevação, dh/dt (m ano⁻¹)")
    ax.set_ylabel("Densidade de probabilidade")
    ax.grid(axis="y", color=GRID, linewidth=0.75, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[mean_line, median_line],
        loc="upper left",
        title=f"n = {len(values):,} nós",
        fontsize=10.2,
        title_fontsize=10.2,
        handlelength=2.6,
        borderpad=0.65,
        labelspacing=0.5,
        facecolor=BG,
        edgecolor="#B9B9B5",
        framealpha=0.96,
    )

    fig.suptitle(
        f"Distribuição de dh/dt — {season_label}",
        fontsize=21,
        fontweight="semibold",
        y=0.955,
        color=INK,
    )
    fig.text(
        0.5,
        0.895,
        f"Amundsen Sea Embayment · ICESat-2 · {season_code} 2019–2025 · nós validados após controle de qualidade",
        ha="center",
        fontsize=11.5,
        color=INK2,
    )
    fig.text(
        0.095,
        0.055,
        "Vermelho = afinamento · azul = espessamento · classes de 0,1 m/ano",
        ha="left",
        fontsize=9.2,
        color=INK2,
    )
    return _save(fig, stem)


def main() -> None:
    values = _load()
    lower = np.floor(min(values["jja"].min(), values["djf"].min()) * 10) / 10
    upper = np.ceil(max(values["jja"].max(), values["djf"].max()) * 10) / 10
    bins = np.arange(lower, upper + 0.1001, 0.1)
    xlim = (float(lower), float(upper))

    outputs = []
    outputs.extend(
        _histogram(
            values["jja"],
            bins,
            xlim,
            "inverno austral",
            "JJA",
            "histograma_dhdt_inverno",
        )
    )
    outputs.extend(
        _histogram(
            values["djf"],
            bins,
            xlim,
            "verão austral",
            "DJF",
            "histograma_dhdt_verao",
        )
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
