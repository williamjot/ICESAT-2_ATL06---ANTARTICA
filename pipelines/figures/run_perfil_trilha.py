"""
pipelines/run_perfil_trilha.py
==============================
Perfil de REPETIÇÃO DE TRILHA: elevação ao longo de um corredor, uma curva por
ano, para ver quanto aquela linha baixou.

    data/<estação>/tiles/*.parquet
        -> outputs/comparacao_sazonal/F7_perfil_trilha_{JJA,DJF,ambos}.png
        -> outputs/comparacao_sazonal/perfil_trilha.json

Por que um CORREDOR e não `track_id`
------------------------------------
`track_id` é rótulo de PASSAGEM — muda a cada troca de feixe ou intervalo de
tempo (ver `thwaites/qc/filttrack.py`). Duas passagens do mesmo terreno em anos
diferentes recebem ids diferentes, então ele não liga repetições. O que liga é
a geometria: o ICESat-2 aponta para repetir a mesma trilha de referência, e as
repetições caem dentro de algumas dezenas de metros umas das outras.

Então a trilha é definida por uma RETA de referência, ajustada por componente
principal sobre uma passagem real, e o corredor é a faixa de `meia_largura` em
torno dela. Tudo que cai dentro é repetição da mesma linha, venha de que ano
vier.

O que a figura mostra
---------------------
A elevação absoluta é dominada pela topografia — centenas de metros de relevo
escondem o sinal de ~1 m/ano. Por isso o painel principal é a ANOMALIA em
relação ao perfil médio do corredor: sobra o rebaixamento, e as curvas se
separam por ano na ordem do tempo quando há adelgaçamento.

Uso: python pipelines/run_perfil_trilha.py
"""

import argparse
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
from thwaites.logging import setup_logging
from thwaites.grid.tiles import load_manifest
from thwaites.uncertainty.mass_balance import apply_coverage_mask
from thwaites.viz.produtos import carregar_bedmachine, para_grade

INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e3e3e0"
_P = ["#08306b", "#3573b9", "#9dc2e0", "#e6e4e1", "#f2b48c", "#d1552a", "#7f2704"]
DIV_THIN = LinearSegmentedColormap.from_list("thin", _P[::-1])
# rampa temporal de hue única: ano é ordinal, e uma sequência clara->escura
# deixa a ordem legível sem legenda; arco-íris embaralharia a leitura
ANOS = LinearSegmentedColormap.from_list(
    "anos", ["#bcd7ea", "#7fb0d0", "#4a86b4", "#2a5f8c", "#123f63", "#06243c"])
OUT = ROOT / "outputs" / "comparacao_sazonal"
KM = 1e-3

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": .7,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "grid.color": GRID, "grid.linewidth": .6,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
})


# ------------------------------------------------------------------ corredor
def polilinha_de_referencia(x, y, passo_m=200.0):
    """
    Geometria REAL da passagem, reamostrada a passo constante.

    A trilha do ICESat-2 é um arco de órbita e, projetada em EPSG:3031, curva
    visivelmente ao longo de dezenas de quilômetros. Aproximá-la por uma reta
    afastaria o corredor nas pontas e capturaria topografia transversal.

    Devolve (px, py, s) com a distância acumulada ao longo da própria trilha.
    """
    o = np.argsort(y)                     # a órbita é quase meridional aqui
    x, y = x[o], y[o]
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    novo = np.arange(0.0, d[-1], passo_m)
    return np.interp(novo, d, x), np.interp(novo, d, y), novo


def escolher_referencia(cfg, log, min_pontos=3000):
    """
    Passagem longa de UM feixe, que define a geometria do corredor.

    O feixe importa: dentro de um par os feixes distam ~90 m e entre pares
    ~3,3 km. Um corredor estreito em torno de UM feixe captura as repetições
    daquele feixe e só delas.
    """
    man = load_manifest(cfg)
    alvo = [e for e in man
            if -1.66e6 <= e["x_min"] <= -1.44e6 and -6.0e5 <= e["y_min"] <= -2.0e5]
    melhor = None
    for e in alvo:
        d = pd.read_parquet(cfg.paths.tiles_dir / e["file"],
                            columns=["track_id", "x", "y"])
        for tid, g in d.groupby("track_id"):
            if len(g) < min_pontos:
                continue
            L = float(np.hypot(g.x.max() - g.x.min(), g.y.max() - g.y.min()))
            if melhor is None or L > melhor[0]:
                melhor = (L, g.x.to_numpy(), g.y.to_numpy(), int(tid), e["tile"])
    if melhor is None:
        raise RuntimeError("nenhuma passagem longa encontrada")
    L, gx, gy, tid, tile = melhor
    px, py, s = polilinha_de_referencia(gx, gy)
    log.info(f"referência: passagem {tid} em {tile}, {L/1e3:.1f} km, "
             f"{len(px)} vértices")
    return px, py, s


def coletar(cfg, px, py, s_ref, meia_largura=100.0, log=None):
    """
    Pontos a menos de `meia_largura` da polilinha de referência.

    A distância ao longo do corredor vem do vértice mais próximo — assim ela
    segue a CURVA real da trilha, e não uma projeção sobre uma reta.
    """
    from scipy.spatial import cKDTree

    arv = cKDTree(np.c_[px, py])
    x0, x1 = px.min() - 5e3, px.max() + 5e3
    y0, y1 = py.min() - 5e3, py.max() + 5e3
    man = load_manifest(cfg)
    partes = []
    for e in man:
        if e["x_max"] < x0 or e["x_min"] > x1 or e["y_max"] < y0 or e["y_min"] > y1:
            continue
        d = pd.read_parquet(cfg.paths.tiles_dir / e["file"],
                            columns=["x", "y", "t_year", "h_corr", "h_res", "beam"])
        dist, idx = arv.query(np.c_[d.x.to_numpy(), d.y.to_numpy()], k=1,
                              distance_upper_bound=meia_largura)
        m = np.isfinite(dist)
        if m.any():
            g = d[m].copy()
            g["s_km"] = s_ref[idx[m]] * KM
            g["desvio_m"] = dist[m]
            partes.append(g)
    if not partes:
        raise RuntimeError("corredor vazio")
    out = pd.concat(partes, ignore_index=True).drop_duplicates(
        subset=["x", "y", "t_year"])
    out["ano"] = np.floor(out.t_year).astype(int)
    if log:
        log.info(f"corredor: {len(out):,} pontos, {out.ano.nunique()} anos, "
                 f"{out.s_km.min():.0f} a {out.s_km.max():.0f} km")
    return out


def perfil_por_ano(df, passo_km=1.0, min_pts=8):
    """Mediana de h por caixa de `passo_km` ao longo do corredor, por ano."""
    b = np.floor(df.s_km / passo_km).astype(int)
    g = df.assign(_b=b).groupby(["ano", "_b"]).agg(
        s=("s_km", "median"), h=("h_corr", "median"),
        hres=("h_res", "median"), n=("h_corr", "size")).reset_index()
    return g[g.n >= min_pts]


# ------------------------------------------------------------------- figuras
def _mapa_corredor(fig, a, cfg, px, py, bx, by, B, titulo):
    g = pd.read_parquet(cfg.paths.interim / "dhdt_grid.parquet")
    nq = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
                         columns=["x", "y"])
    g = apply_coverage_mask(g, nq, cfg.mass_balance.coverage_dist_m)
    xs, ys, M = para_grade(g, "pred")
    im = a.pcolormesh(xs * KM, ys * KM, M, cmap=DIV_THIN, shading="auto",
                      norm=TwoSlopeNorm(vcenter=0, vmin=-3.5, vmax=1.0),
                      rasterized=True)
    cb = fig.colorbar(im, ax=a, shrink=.86, pad=.02)
    cb.set_label("dh/dt (m ano⁻¹)"); cb.ax.axhline(0, color=INK, lw=.8)
    a.contour(bx * KM, by * KM, (B["mask"] == 2).astype(float), levels=[.5],
              colors="#8b0000", linewidths=.9)
    a.contour(bx * KM, by * KM, np.isin(B["mask"], (1, 2, 3, 4)).astype(float),
              levels=[.5], colors="#333333", linewidths=.6)
    a.plot(px * KM, py * KM, color="#111111", lw=2.4, solid_capstyle="round",
           zorder=8)
    for i, lab in ((0, "0 km"), (-1, f"{(len(px)-1)*0.2:.0f} km")):
        a.plot(px[i] * KM, py[i] * KM, "o", ms=6, color="#111111", zorder=9)
        a.text(px[i] * KM + 14, py[i] * KM, lab, fontsize=7.6, color=INK,
               va="center", zorder=9,
               bbox=dict(fc="#fcfcfb", ec="none", alpha=.9, pad=1.6))
    a.set_aspect("equal"); a.set_xlabel("x (km)"); a.set_ylabel("y (km)")
    a.set_title(titulo)


def figura(cfg, dados, px, py, bx, by, B, nome, titulo, sub):
    pf = perfil_por_ano(dados)
    anos = sorted(pf.ano.unique())
    cores = {a: ANOS(i / max(len(anos) - 1, 1)) for i, a in enumerate(anos)}
    # `h_res` = h_corr − REMA, isto é, a altura menos o relevo AMOSTRADO NO
    # PRÓPRIO PONTO. Subtrair um perfil médio por caixa preservaria relevo
    # transversal e introduziria dispersão que não representa mudança temporal.
    ref = pf.groupby("_b").hres.median()

    fig = plt.figure(figsize=(13.2, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.55], hspace=.32, wspace=.22)

    a = fig.add_subplot(gs[:, 0])
    _mapa_corredor(fig, a, cfg, px, py, bx, by, B,
                   "a. Corredor sobre o campo de dh/dt")

    b = fig.add_subplot(gs[0, 1])
    for an in anos:
        q = pf[pf.ano == an]
        b.plot(q.s, q.h, lw=1.1, color=cores[an])
    b.set_ylabel("elevação (m)")
    b.set_title("b. Elevação absoluta — a topografia domina")
    b.grid(alpha=.6); b.set_axisbelow(True); b.set_xticklabels([])

    c = fig.add_subplot(gs[1, 1])
    for an in anos:
        q = pf[pf.ano == an]
        c.plot(q.s, q.hres - ref.reindex(q._b).to_numpy(), lw=1.5,
               color=cores[an], label=str(an))
    c.axhline(0, color=MUTED, lw=.8, ls=":")
    c.set_xlabel("distância ao longo do corredor (km)")
    c.set_ylabel("anomalia de elevação (m)")
    c.set_title("c. Anomalia de altura relativa ao REMA — o rebaixamento")
    c.grid(alpha=.6); c.set_axisbelow(True)
    c.legend(fontsize=7.4, labelcolor=INK2, ncol=4, loc="lower right",
             title="ano", title_fontsize=7.4, frameon=True, facecolor="#fcfcfb",
             edgecolor=GRID, framealpha=.93)

    fig.suptitle(titulo, fontsize=12.5, fontweight="semibold", y=.985, color=INK)
    fig.text(.5, .948, sub, ha="center", fontsize=8.6, color=INK2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / nome, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  {nome}")
    return pf, ref


def main():
    ap = argparse.ArgumentParser(description="Perfil de repetição de trilha.")
    ap.add_argument("--meia-largura-m", type=float, default=100.0)
    args = ap.parse_args()

    cfg_j = load_config()
    log = setup_logging(cfg_j.paths.logs, level=cfg_j.logging.level,
                        run_name="perfil_trilha")
    cfg_d = load_config("djf")

    px, py, s_ref = escolher_referencia(cfg_j, log)

    bmp = sorted(cfg_j.paths.data_dir.glob("*BedMachine*.nc"))[0]
    g = pd.read_parquet(cfg_j.paths.interim / "dhdt_grid.parquet", columns=["x", "y"])
    bx, by, B = carregar_bedmachine(bmp, g.x.min() - 8e3, g.x.max() + 8e3,
                                    g.y.min() - 8e3, g.y.max() + 8e3)

    rel = {"corredor": {"comprimento_km": float(s_ref[-1] * KM),
                        "n_vertices": int(len(px)),
                        "meia_largura_m": args.meia_largura_m}}
    dados = {}
    print("figuras:")
    for lab, cfg in (("JJA", cfg_j), ("DJF", cfg_d)):
        d = coletar(cfg, px, py, s_ref, args.meia_largura_m, log)
        dados[lab] = d
        est = "inverno austral" if lab == "JJA" else "verão austral"
        pf, _ = figura(cfg, d, px, py, bx, by, B,
                       f"F7_perfil_trilha_{lab}.png",
                       f"Perfil de repetição de trilha — {lab} · {est}",
                       f"corredor de ±{args.meia_largura_m:.0f} m · "
                       f"mediana por caixa de 1 km · uma curva por ano")
        rel[lab] = {"n_pontos": int(len(d)), "anos": sorted(map(int, d.ano.unique())),
                    "desvio_mediano_m": float(d.desvio_m.abs().median())}

    # ---- as duas estações no mesmo corredor -------------------------------
    pj = perfil_por_ano(dados["JJA"])
    pd_ = perfil_por_ano(dados["DJF"])
    ref = pd.concat([pj, pd_]).groupby("_b").hres.median()
    anos = sorted(set(pj.ano) | set(pd_.ano))
    cores = {a: ANOS(i / max(len(anos) - 1, 1)) for i, a in enumerate(anos)}

    fig, ax = plt.subplots(2, 1, figsize=(12.4, 7.6), sharex=True)
    for k, (lab, p, ls) in enumerate((("JJA", pj, "-"), ("DJF", pd_, "--"))):
        for an in anos:
            q = p[p.ano == an]
            if not len(q):
                continue
            ax[0].plot(q.s, q.hres - ref.reindex(q._b).to_numpy(), lw=1.4, ls=ls,
                       color=cores[an], label=str(an) if k == 0 else None)
    ax[0].axhline(0, color=MUTED, lw=.8, ls=":")
    ax[0].set_ylabel("anomalia de elevação (m)")
    ax[0].set_title("a. Inverno (linha cheia) e verão (tracejada) no mesmo corredor")
    ax[0].grid(alpha=.6); ax[0].set_axisbelow(True)
    ax[0].legend(fontsize=7.4, labelcolor=INK2, ncol=7, loc="upper center",
                 bbox_to_anchor=(.5, -.02), title="ano", title_fontsize=7.4,
                 frameon=True, facecolor="#fcfcfb", edgecolor=GRID, framealpha=.93)

    # Rebaixamento acumulado: primeiro ano contra último.
    #
    # As caixas usadas são as presentes nos QUATRO conjuntos (primeiro e último
    # ano de cada estação). Sem essa interseção cada estação cobriria um trecho
    # diferente do corredor, fazendo as curvas descreverem lugares distintos.
    pontas = {}
    for lab, p in (("JJA", pj), ("DJF", pd_)):
        a0, a1 = min(p.ano), max(p.ano)
        pontas[lab] = (a0, a1,
                       p[p.ano == a0].set_index("_b").hres,
                       p[p.ano == a1].set_index("_b").hres)
    com = None
    for _, _, h0, h1 in pontas.values():
        ix = h0.index.intersection(h1.index)
        com = ix if com is None else com.intersection(ix)
    s_pos = pd.concat([pj, pd_]).set_index("_b").s.groupby(level=0).first()
    for (lab, (a0, a1, h0, h1)), col, ls in zip(
            pontas.items(), ("#0072B2", "#D55E00"), ("-", "--")):
        ax[1].plot(s_pos.reindex(com), (h1 - h0).reindex(com), lw=1.9, ls=ls,
                   color=col, label=f"{lab}: {a0} → {a1}")
    ax[1].text(.985, .06, f"{len(com)} km comuns às duas estações",
               transform=ax[1].transAxes, ha="right", fontsize=7.6, color=INK2)
    ax[1].axhline(0, color=MUTED, lw=.8, ls=":")
    ax[1].set_xlabel("distância ao longo do corredor (km)")
    ax[1].set_ylabel("variação total (m)")
    ax[1].set_title("b. Rebaixamento acumulado entre o primeiro e o último ano")
    ax[1].grid(alpha=.6); ax[1].set_axisbelow(True)
    ax[1].legend(fontsize=7.8, labelcolor=INK2)

    fig.suptitle("Rebaixamento ao longo de uma trilha — inverno e verão",
                 fontsize=12.5, fontweight="semibold", y=.985, color=INK)
    fig.text(.5, .945, "mesmo corredor nas duas estações · anomalia em relação ao "
                       "perfil médio conjunto", ha="center", fontsize=8.6, color=INK2)
    fig.tight_layout(rect=[0, .01, 1, .935])
    fig.savefig(OUT / "F7_perfil_trilha_ambos.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  F7_perfil_trilha_ambos.png")

    (OUT / "perfil_trilha.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
