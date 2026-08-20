"""
Figuras comparativas independentes para apresentação e publicação.

1. Histograma conjunto de dh/dt para JJA e DJF, usando os nós validados após QC,
   classes idênticas e densidade de probabilidade.
2. Série profissional de massa acumulada para JJA e DJF, baseada nos relatórios
   de massa já produzidos pelo pipeline.

Nenhuma incerteza é inventada: a série de massa não contém propagação formal de
erro e essa limitação é informada na própria figura.

Saídas
------
outputs/comparacao_sazonal/histograma_dhdt_JJA_DJF.{png,svg}
outputs/comparacao_sazonal/massa_acumulada_profissional.{png,svg}

Uso: python pipelines/run_figuras_comparativas.py
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
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config


OUT = ROOT / "outputs" / "comparacao_sazonal"
BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#1A1A1A"
INK2 = "#4A4A4A"
MUTED = "#777777"
GRID = "#D9D9D6"
BG = "#FCFCFB"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 17,
        "axes.labelsize": 12,
        "axes.edgecolor": "#777777",
        "axes.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }
)


def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    svg = OUT / f"{stem}.svg"
    fig.savefig(png, dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(svg, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png, svg


def _load_dhdt() -> dict[str, np.ndarray]:
    configs = {"JJA": load_config(), "DJF": load_config("djf")}
    values = {}
    for season, cfg in configs.items():
        path = cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet"
        data = pd.read_parquet(path, columns=["dhdt"])["dhdt"].to_numpy(float)
        values[season] = data[np.isfinite(data)]
    return values


def histograma_dhdt() -> tuple[Path, Path]:
    values = _load_dhdt()
    jja, djf = values["JJA"], values["DJF"]

    lower = np.floor(min(jja.min(), djf.min()) * 10) / 10
    upper = np.ceil(max(jja.max(), djf.max()) * 10) / 10
    bins = np.arange(lower, upper + 0.1001, 0.1)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    fig.subplots_adjust(left=0.095, right=0.965, bottom=0.145, top=0.80)

    styles = {
        "JJA": (jja, BLUE, "-", "JJA · inverno austral"),
        "DJF": (djf, ORANGE, (0, (4, 2)), "DJF · verão austral"),
    }
    for season, (data, color, linestyle, label) in styles.items():
        density, edges = np.histogram(data, bins=bins, density=True)
        ax.stairs(density, edges, fill=True, color=color, alpha=0.10)
        ax.stairs(
            density,
            edges,
            color=color,
            linewidth=2.2,
            linestyle=linestyle,
            label=(
                f"{label}  ·  n={len(data):,}  ·  média={np.mean(data):+.3f}  ·  "
                f"mediana={np.median(data):+.3f} m/ano"
            ),
        )
        ax.axvline(
            np.median(data),
            color=color,
            linewidth=1.35,
            linestyle=linestyle,
            alpha=0.95,
        )

    ax.axvline(0, color=INK, linewidth=1.0, linestyle=":", alpha=0.75)
    ax.text(
        0.01,
        0.97,
        "afinamento",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#A8481C",
    )
    ax.text(
        0.99,
        0.97,
        "espessamento",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=BLUE,
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Taxa de mudança de elevação, dh/dt (m ano⁻¹)")
    ax.set_ylabel("Densidade de probabilidade")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.13), fontsize=9.4, handlelength=3.4)

    fig.suptitle(
        "Distribuição de dh/dt no inverno e no verão austral",
        fontsize=21,
        fontweight="semibold",
        y=0.955,
        color=INK,
    )
    fig.text(
        0.5,
        0.895,
        "Amundsen Sea Embayment · ICESat-2 · 2019–2025 · nós validados após controle de qualidade",
        ha="center",
        fontsize=11.5,
        color=INK2,
    )
    thin_jja = 100 * np.mean(jja < 0)
    thin_djf = 100 * np.mean(djf < 0)
    fig.text(
        0.095,
        0.065,
        f"Fração dos nós com afinamento: JJA {thin_jja:.0f}% · DJF {thin_djf:.0f}%  |  "
        f"assimetria: JJA {stats.skew(jja):+.2f} · DJF {stats.skew(djf):+.2f}",
        ha="left",
        fontsize=9.2,
        color=INK2,
    )
    fig.text(
        0.095,
        0.030,
        "JJA e DJF são janelas de amostragem distintas; a comparação não isola, por si só, sazonalidade física do gelo.",
        ha="left",
        fontsize=8.8,
        color="#A8481C",
    )
    return _save(fig, "histograma_dhdt_JJA_DJF")


def _load_mass() -> dict[str, tuple[pd.DataFrame, dict]]:
    configs = {"JJA": load_config(), "DJF": load_config("djf")}
    result = {}
    for season, cfg in configs.items():
        report = json.loads(
            (cfg.paths.tables / "serie_massa.json").read_text(encoding="utf-8")
        )
        result[season] = (pd.DataFrame(report["serie"]), report)
    return result


def massa_acumulada() -> tuple[Path, Path]:
    mass = _load_mass()
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    fig.subplots_adjust(left=0.105, right=0.855, bottom=0.155, top=0.82)

    plotted = {}
    for season, color, label in (
        ("JJA", BLUE, "JJA · inverno austral"),
        ("DJF", ORANGE, "DJF · verão austral"),
    ):
        series, report = mass[season]
        years = series["ano"].to_numpy(float)
        values = series["massa_acumulada_Gt"].to_numpy(float)
        rate = float(np.polyfit(years, values, 1)[0])
        plotted[season] = (series, report, rate)
        ax.plot(
            years,
            values,
            color=color,
            linewidth=3.0,
            marker="o",
            markersize=7.2,
            markerfacecolor=BG,
            markeredgecolor=color,
            markeredgewidth=2.2,
            label=f"{label}  ·  taxa linear observada {rate:+.1f} Gt/ano",
            zorder=3,
        )

    ax.axhline(0, color=INK, linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_xlim(2018.8, 2026.05)
    ax.set_ylim(-600, 35)
    ax.set_xticks(np.arange(2019, 2026))
    ax.set_xlabel("Ano")
    ax.set_ylabel("Variação acumulada de massa (Gt; relativa a 2019)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=10.2, handlelength=3.0)

    for season, color, offset in (("JJA", BLUE, 13), ("DJF", ORANGE, -13)):
        series, _, _ = plotted[season]
        last = series.iloc[-1]
        ax.annotate(
            f"{season}  {last['massa_acumulada_Gt']:+.1f} Gt",
            xy=(last["ano"], last["massa_acumulada_Gt"]),
            xytext=(2025.18, last["massa_acumulada_Gt"] + offset),
            color=color,
            fontsize=11.5,
            fontweight="semibold",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=1.3),
            annotation_clip=False,
        )

    jja_report = plotted["JJA"][1]
    djf_report = plotted["DJF"][1]
    fig.suptitle(
        "Perda de massa acumulada no Amundsen Sea Embayment",
        fontsize=21,
        fontweight="semibold",
        y=0.955,
        color=INK,
    )
    fig.text(
        0.5,
        0.895,
        "ICESat-2 · referência em 2019 · massa inferida da variação de altura × densidade do gelo",
        ha="center",
        fontsize=11.5,
        color=INK2,
    )
    fig.text(
        0.105,
        0.074,
        f"Suporte espacial fixo: JJA {jja_report['area_km2']:,.0f} km² "
        f"({jja_report['n_nos_completos']:,} nós) · DJF {djf_report['area_km2']:,.0f} km² "
        f"({djf_report['n_nos_completos']:,} nós)",
        ha="left",
        fontsize=9.2,
        color=INK2,
    )
    fig.text(
        0.105,
        0.035,
        "Grandeza inferida, não medida por gravimetria. A propagação formal de incerteza ainda não está incorporada à série.",
        ha="left",
        fontsize=8.8,
        color="#A8481C",
    )
    return _save(fig, "massa_acumulada_profissional")


def main() -> None:
    hist_png, hist_svg = histograma_dhdt()
    mass_png, mass_svg = massa_acumulada()
    print(f"Histograma PNG -> {hist_png}")
    print(f"Histograma SVG -> {hist_svg}")
    print(f"Massa PNG -> {mass_png}")
    print(f"Massa SVG -> {mass_svg}")


if __name__ == "__main__":
    main()
