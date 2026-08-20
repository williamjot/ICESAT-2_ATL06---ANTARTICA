"""
pipelines/run_figuras_massa.py
==============================
Duas figuras no formato do produto GRACE, mas com ICESat-2 e no escopo da ROI.

    -> outputs/comparacao_sazonal/F9_serie_massa.png
    -> outputs/comparacao_sazonal/F10_perfil_por_latitude.png

F9 — série acumulada + mapa em água equivalente
-----------------------------------------------
Equivale aos dois painéis da figura de balanço de massa antártico do GRACE, com
uma diferença de escopo que precisa ficar explícita na própria figura: o GRACE
cobre a Antártica inteira desde 2002 e mede massa pela deformação do campo
gravitacional; aqui a cobertura é o Amundsen Sea Embayment desde 2019 e a massa
é INFERIDA de altura vezes densidade suposta.

F10 — rebaixamento por latitude
-------------------------------
Mesma trilha, uma curva por ano, referenciada ao PRIMEIRO ano: 2019 fica em zero
e os anos seguintes descem. É mais legível que a anomalia contra o perfil médio
(usada na F7) porque o eixo passa a ser lido diretamente como "quanto baixou
desde o início".

Uso: python pipelines/run_figuras_massa.py
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
from thwaites.viz.produtos import carregar_bedmachine, para_grade, amostrar

CAT = ["#0072B2", "#D55E00"]
INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e3e3e0"
_P = ["#08306b", "#3573b9", "#9dc2e0", "#e6e4e1", "#f2b48c", "#d1552a", "#7f2704"]
DIV_THIN = LinearSegmentedColormap.from_list("thin", _P[::-1])
ANOS = LinearSegmentedColormap.from_list(
    "anos", ["#8f6fb5", "#7f9fd0", "#63b8c9", "#4fbfa3", "#6cbf6e", "#a8bd4e",
             "#c9a227"])
OUT = ROOT / "outputs" / "comparacao_sazonal"
KM = 1e-3
# IMBIE / NASA GRACE-FO: perda média da Antártica inteira, para contexto.
# Entra na figura como REFERÊNCIA EXTERNA, nunca como dado nosso.
ANTARTICA_GT_ANO = -135.0
ANTARTICA_AREA_KM2 = 13.98e6

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


def fig9(cfgs, bx, by, B, ext):
    fig = plt.figure(figsize=(13.0, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])

    a = fig.add_subplot(gs[0, 0])
    txt = []
    for l, lab, col in (("jja", "JJA · inverno", CAT[0]), ("djf", "DJF · verão", CAT[1])):
        r = json.loads((cfgs[l].paths.tables / "serie_massa.json").read_text(
            encoding="utf-8"))
        s = pd.DataFrame(r["serie"])
        a.plot(s.ano, s.massa_acumulada_Gt, marker="o", ms=6, lw=2.2, color=col,
               label=lab)
        # taxa média do período, por regressão sobre a própria curva
        k = np.polyfit(s.ano, s.massa_acumulada_Gt, 1)[0]
        txt.append((lab, k, float(s.massa_acumulada_Gt.iloc[-1]), r["area_km2"],
                    r["n_nos_completos"]))
    a.axhline(0, color=MUTED, lw=.9, ls=":")
    a.set_xlabel("ano"); a.set_ylabel("massa acumulada (Gt, relativa a 2019)")
    a.set_title("a. Massa acumulada no Amundsen Sea Embayment")
    a.grid(alpha=.65); a.set_axisbelow(True)
    a.legend(fontsize=8, labelcolor=INK2, loc="lower left")
    linhas = "\n".join(f"{lab}: {k:+.1f} Gt/ano" for lab, k, *_ in txt)
    a.text(.975, .96, linhas, transform=a.transAxes, ha="right", va="top",
           fontsize=9, color=INK, linespacing=1.5,
           bbox=dict(fc="#fcfcfb", ec=GRID, alpha=.95, pad=4))
    a.text(.975, .06,
           f"Antártica inteira: {ANTARTICA_GT_ANO:.0f} Gt/ano (GRACE-FO)\n"
           f"esta região = {100*txt[0][3]/ANTARTICA_AREA_KM2:.1f}% da área do continente",
           transform=a.transAxes, ha="right", va="bottom", fontsize=7.8,
           color=INK2, linespacing=1.4,
           bbox=dict(fc="#fcfcfb", ec=GRID, alpha=.95, pad=3.4))

    b = fig.add_subplot(gs[0, 1])
    c = cfgs["jja"]
    S = pd.read_parquet(c.paths.dhdt_dir / "serie_anual.parquet")
    ult = S[S.ano == S.ano.max()].copy()
    ult["mwe"] = ult.anom * 917.0 / 1000.0
    xs, ys, M = para_grade(ult, "mwe")
    im = b.pcolormesh(xs * KM, ys * KM, M, cmap=DIV_THIN, shading="auto",
                      norm=TwoSlopeNorm(vcenter=0, vmin=-8, vmax=3), rasterized=True)
    cb = fig.colorbar(im, ax=b, shrink=.88, pad=.02)
    cb.set_label("variação acumulada (m de água equivalente)")
    cb.ax.axhline(0, color=INK, lw=.8)
    contornos(b, bx, by, B["mask"])
    b.set_xlim(*ext[:2]); b.set_ylim(*ext[2:]); b.set_aspect("equal")
    b.set_xlabel("x (km, EPSG:3031)"); b.set_ylabel("y (km)")
    b.set_title(f"b. Acumulado 2019 → {int(S.ano.max())} — inverno")

    fig.tight_layout(rect=[0, .03, 1, .90])
    fig.suptitle("Perda de massa no Amundsen Sea Embayment — ICESat-2",
                 fontsize=12.5, fontweight="semibold", y=.985, color=INK)
    fig.text(.5, .935,
             "mesma leitura dos painéis do GRACE, em escopo regional · a massa aqui é "
             "INFERIDA de altura × densidade, não medida por gravimetria",
             ha="center", fontsize=8.6, color=INK2)
    fig.text(.5, .008,
             "O GRACE mede a Antártica inteira desde 2002; este produto cobre 1,4% da "
             "área do continente desde 2019. Os totais não são intercambiáveis.",
             ha="center", fontsize=8, color="#a8481c")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "F9_serie_massa.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  F9_serie_massa.png")
    return txt


def fig10(cfgs):
    """Rebaixamento por latitude, referenciado ao primeiro ano."""
    fig, ax = plt.subplots(2, 1, figsize=(11.0, 8.2), sharex=True)
    for k, (l, lab) in enumerate((("jja", "JJA · inverno austral"),
                                  ("djf", "DJF · verão austral"))):
        c = cfgs[l]
        d = pd.read_parquet(c.paths.dhdt_dir / "serie_anual.parquet")
        # posição -> latitude, pelo nó mais próximo da tabela de nós
        nq = pd.read_parquet(c.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
                             columns=["x", "y", "lat", "lon"])
        d = d.merge(nq, on=["x", "y"], how="inner")
        # faixa longitudinal estreita: um corte quase meridional pelo tronco
        faixa = d[(d.lon > -107.5) & (d.lon < -105.5)]
        if len(faixa) < 200:
            faixa = d[(d.lon > -109) & (d.lon < -104)]
        anos = sorted(faixa.ano.unique())
        cores = {a: ANOS(i / max(len(anos) - 1, 1)) for i, a in enumerate(anos)}
        a = ax[k]
        for an in anos:
            g = faixa[faixa.ano == an].sort_values("lat")
            if len(g) < 8:
                continue
            b = np.round(g.lat * 20) / 20            # caixas de 0,05°
            q = g.assign(_b=b).groupby("_b").anom.median()
            a.plot(q.index, q.values, lw=1.7, color=cores[an], label=str(an))
        a.axhline(0, color=MUTED, lw=.9, ls=":")
        a.set_ylabel("mudança de elevação (m)")
        a.set_title(f"{'ab'[k]}. {lab}")
        a.grid(alpha=.6); a.set_axisbelow(True)
        a.invert_xaxis()
        if k == 0:
            a.legend(fontsize=7.6, labelcolor=INK2, ncol=4, loc="lower left",
                     title="ano", title_fontsize=7.6, frameon=True,
                     facecolor="#fcfcfb", edgecolor=GRID, framealpha=.93)
    ax[1].set_xlabel("latitude (°)")
    fig.tight_layout(rect=[0, .012, 1, .925])
    fig.suptitle("Rebaixamento por latitude — mesma faixa, uma curva por ano",
                 fontsize=12.5, fontweight="semibold", y=.985, color=INK)
    fig.text(.5, .942,
             "cada curva é a mediana por caixa de 0,05° de latitude, referenciada ao "
             "primeiro ano de cada nó · faixa longitudinal estreita sobre o tronco",
             ha="center", fontsize=8.6, color=INK2)
    fig.savefig(OUT / "F10_perfil_por_latitude.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  F10_perfil_por_latitude.png")


def main():
    cfgs = {"jja": load_config(), "djf": load_config("djf")}
    g0 = pd.read_parquet(cfgs["jja"].paths.interim / "dhdt_grid.parquet",
                         columns=["x", "y"])
    x0, x1 = g0.x.min() - 8e3, g0.x.max() + 8e3
    y0, y1 = g0.y.min() - 8e3, g0.y.max() + 8e3
    ext = (x0 * KM, x1 * KM, y0 * KM, y1 * KM)
    bmp = sorted(cfgs["jja"].paths.data_dir.glob("*BedMachine*.nc"))[0]
    bx, by, B = carregar_bedmachine(bmp, x0, x1, y0, y1, vars=("mask",))
    print("figuras:")
    txt = fig9(cfgs, bx, by, B, ext)
    fig10(cfgs)
    for lab, k, tot, area, n in txt:
        print(f"  {lab}: {k:+.1f} Gt/ano | acumulado {tot:+.1f} Gt | "
              f"{n:,} nós | {area:,.0f} km²")


if __name__ == "__main__":
    main()
