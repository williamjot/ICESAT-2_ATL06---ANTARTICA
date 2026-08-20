"""
pipelines/run_previsao_mapa.py
==============================
Mapas de projeção de elevação, com o horizonte que o hindcast validou.

    -> outputs/comparacao_sazonal/F8_projecao.png

O horizonte é 3 anos, e não é escolha
-------------------------------------
O registro tem 7 anos. Para testar a extrapolação é preciso treinar num trecho e
verificar no resto, e o ajuste exige `dhdt.dt_min_years` = 3 anos de vão. Treinar
em 4 e verificar em 3 é a única partição possível: com 5 de treino sobram 2 de
teste, com 3 o treino já não qualifica. Logo **3 anos é o horizonte máximo
verificável com este registro** — qualquer projeção além disso é extrapolação da
extrapolação, sem nada com que confrontá-la.

O que o hindcast mediu (treino t < 2023, verificação 2023–2025)
--------------------------------------------------------------
Ganho sobre persistência de 55–60% no conjunto, mas MUITO desigual no espaço:

    |dh/dt| < 0,25 m/ano  ->  ganho 20%   (ruído domina o sinal)
    0,25 – 0,5            ->  35–43%
    0,5 – 1              ->  53–56%
    1 – 2                ->  67%
    > 2                  ->  60–66%

Por isso a projeção vem acompanhada do mapa de competência: onde a taxa é
pequena, estender a reta quase não é melhor do que supor que nada muda, e o
número projetado ali não deve ser usado.

Uso: python pipelines/run_previsao_mapa.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from thwaites import load_config
from thwaites.uncertainty.mass_balance import apply_coverage_mask
from thwaites.viz.produtos import carregar_bedmachine, para_grade, amostrar

INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e3e3e0"
_P = ["#08306b", "#3573b9", "#9dc2e0", "#e6e4e1", "#f2b48c", "#d1552a", "#7f2704"]
DIV_THIN = LinearSegmentedColormap.from_list("thin", _P[::-1])
SKILL = LinearSegmentedColormap.from_list(
    "skill", ["#e6e4e1", "#cfe0d6", "#9ec9b4", "#5da88a", "#2b7d5f", "#0d5138"])
OUT = ROOT / "outputs" / "comparacao_sazonal"
KM = 1e-3
HORIZONTE = 3.0

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": .7,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "grid.color": GRID, "grid.linewidth": .6,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
})


def contornos(ax, bx, by, mask):
    gl = np.where(mask == 2, 1.0, np.where(mask == 3, -1.0, np.nan))
    ax.contour(bx * KM, by * KM, gl, levels=[0.0], colors="#8b0000",
               linewidths=.9, zorder=6)
    ax.contour(bx * KM, by * KM, np.isin(mask, (1, 2, 3, 4)).astype(float),
               levels=[.5], colors="#333333", linewidths=.6, zorder=6)
    if (mask == 1).any():
        ax.contourf(bx * KM, by * KM, (mask == 1).astype(float), levels=[.5, 1.5],
                    colors=["#9a9186"], zorder=5)


def eixo(ax, ext):
    ax.set_xlim(*ext[:2]); ax.set_ylim(*ext[2:]); ax.set_aspect("equal")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")


def main():
    cfgs = {"jja": load_config(), "djf": load_config("djf")}
    g0 = pd.read_parquet(cfgs["jja"].paths.interim / "dhdt_grid.parquet",
                         columns=["x", "y"])
    x0, x1 = g0.x.min() - 8e3, g0.x.max() + 8e3
    y0, y1 = g0.y.min() - 8e3, g0.y.max() + 8e3
    ext = (x0 * KM, x1 * KM, y0 * KM, y1 * KM)
    bmp = sorted(cfgs["jja"].paths.data_dir.glob("*BedMachine*.nc"))[0]
    bx, by, B = carregar_bedmachine(bmp, x0, x1, y0, y1, vars=("mask",))

    fig, ax = plt.subplots(2, 2, figsize=(10.4, 9.2))
    rel = {"horizonte_anos": HORIZONTE, "estacoes": {}}

    for k, (l, lab) in enumerate((("jja", "JJA · inverno"), ("djf", "DJF · verão"))):
        c = cfgs[l]
        g = pd.read_parquet(c.paths.interim / "dhdt_grid.parquet")
        nq = pd.read_parquet(c.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
                             columns=["x", "y"])
        g = apply_coverage_mask(g, nq, c.mass_balance.coverage_dist_m)
        cls = amostrar(B["mask"].astype(int), bx, by, g.x.to_numpy(), g.y.to_numpy())
        g = g[cls == 2].copy()
        g["proj"] = g["pred"] * HORIZONTE

        xs, ys, M = para_grade(g, "proj")
        a = ax[0, k]
        im = a.pcolormesh(xs * KM, ys * KM, M, cmap=DIV_THIN, shading="auto",
                          norm=TwoSlopeNorm(vcenter=0, vmin=-10, vmax=3),
                          rasterized=True)
        cb = fig.colorbar(im, ax=a, shrink=.86, pad=.02)
        cb.set_label(f"variação projetada em {HORIZONTE:.0f} anos (m)")
        cb.ax.axhline(0, color=INK, lw=.8)
        contornos(a, bx, by, B["mask"]); eixo(a, ext)
        a.set_title(f"{'ab'[k]}. Projeção a {HORIZONTE:.0f} anos — {lab}")
        v = M[np.isfinite(M)]
        a.text(.03, .04, f"mediana {np.median(v):+.2f} m\np10 {np.percentile(v,10):+.2f}",
               transform=a.transAxes, fontsize=7.6, color=INK2, va="bottom",
               bbox=dict(fc="#fcfcfb", ec=GRID, alpha=.93, pad=2.6))

        # competência local, do hindcast
        H = pd.read_parquet(c.paths.dhdt_dir / "previsao_nodes.parquet")
        H = H[H.ano == H.ano.max()].copy()
        H["ganho"] = 1 - H.rms_extrap / H.rms_persist.replace(0, np.nan)
        xs2, ys2, S = para_grade(H, "ganho")
        b = ax[1, k]
        im = b.pcolormesh(xs2 * KM, ys2 * KM, np.clip(S, 0, .9), cmap=SKILL,
                          vmin=0, vmax=.9, shading="auto", rasterized=True)
        fig.colorbar(im, ax=b, shrink=.86, pad=.02).set_label(
            "ganho sobre persistência")
        contornos(b, bx, by, B["mask"]); eixo(b, ext)
        b.set_title(f"{'cd'[k]}. Onde a projeção tem competência — {lab}")
        s = S[np.isfinite(S)]
        b.text(.03, .04, f"{100*np.mean(s > .4):.0f}% dos nós com ganho > 40%",
               transform=b.transAxes, fontsize=7.6, color=INK2, va="bottom",
               bbox=dict(fc="#fcfcfb", ec=GRID, alpha=.93, pad=2.6))

        sk = json.loads((c.paths.tables / "previsao_skill.json").read_text(
            encoding="utf-8"))
        rel["estacoes"][l] = {
            "projecao_mediana_m": float(np.median(v)),
            "projecao_p10_m": float(np.percentile(v, 10)),
            "frac_nos_ganho_acima_40pct": float(np.mean(s > .4)),
            "skill": sk["por_avanco"],
        }

    fig.tight_layout(rect=[0, .028, 1, .93])
    fig.suptitle("Projeção de variação de elevação e onde ela tem competência",
                 fontsize=12.5, fontweight="semibold", y=.985, color=INK)
    fig.text(.5, .945,
             f"horizonte de {HORIZONTE:.0f} anos — o MÁXIMO que 7 anos de registro "
             f"permitem verificar · ganho medido por hindcast (treino < 2023)",
             ha="center", fontsize=8.6, color=INK2)
    fig.text(.5, .008,
             "Projeção linear: supõe a taxa constante. As janelas móveis mostram "
             "que ela varia ±25%, e onde o ganho é baixo o valor projetado não deve "
             "ser usado.", ha="center", fontsize=8, color="#a8481c")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "F8_projecao.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    (OUT / "projecao.json").write_text(json.dumps(rel, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print("  F8_projecao.png")


if __name__ == "__main__":
    main()
