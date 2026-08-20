"""
pipelines/run_produto_figuras.py
================================
Figuras de PRODUTO com dado real, comparando inverno (JJA) e verão (DJF).

    -> outputs/comparacao_sazonal/*.png

Ênfase cartográfica: das cinco figuras, quatro são mapas. As duas séries
temporais que existem estão subordinadas a um mapa que mostra ONDE elas foram
medidas — uma curva de fluxo sem o portão desenhado no mapa é ilegível.

Convenção de cor (mesma do catálogo, validada com o validador da skill dataviz)
------------------------------------------------------------------------------
* dh/dt e DH/Dt: divergente com adelgaçamento em VERMELHO (convenção da
  criosfera), meio cinza — não branco, que sumiria sobre o relevo.
* ṁ_b: divergente com derretimento em VERMELHO, recongelamento em azul.
* HAF, tempo até desaterrar, velocidade: magnitudes sem polaridade -> hue única.
* Nunca arco-íris.

Uso: python pipelines/run_produto_figuras.py
"""

import json
import sys
from pathlib import Path

# O console do Windows usa cp1252 e não codifica "ṁ"; sem isto o script morre
# ao IMPRIMIR o diagnóstico, depois de já ter feito o trabalho.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.uncertainty.mass_balance import apply_coverage_mask
from thwaites.viz.produtos import (carregar_bedmachine, altura_de_flutuacao,
                                   tempo_ate_desaterrar, carregar_velocidade_anual,
                                   amostrar, para_grade, fluxo_por_portao)

CAT = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e3e3e0"
_P = ["#08306b", "#3573b9", "#9dc2e0", "#e6e4e1", "#f2b48c", "#d1552a", "#7f2704"]
DIV_MELT = LinearSegmentedColormap.from_list("melt", _P)
DIV_THIN = LinearSegmentedColormap.from_list("thin", _P[::-1])
SEQ = LinearSegmentedColormap.from_list(
    "mag", ["#f2f7fa", "#c2dbe9", "#84b5d2", "#4a86b4", "#245d8f", "#0b3055"])
URG = LinearSegmentedColormap.from_list(          # tempo: curto = alarmante
    "urg", ["#7f2704", "#d1552a", "#f2b48c", "#c2dbe9", "#245d8f"])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": .7,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "grid.color": GRID, "grid.linewidth": .6,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
})
OUT = ROOT / "outputs" / "comparacao_sazonal"
KM = 1e-3


def fim(fig, nome, titulo, sub=""):
    # O espaçamento do cabeçalho é em POLEGADAS convertidas em fração, não em
    # fração fixa: numa figura baixa e larga (F4 tem 5 pol de altura) uma folga
    # de 0,03 vale 0,15 pol e o subtítulo sobe por cima do título.
    h = fig.get_figheight()
    dy = 0.30 / h
    fig.tight_layout(rect=[0, .012, 1, 1 - (2.1 * dy if sub else 1.1 * dy)])
    fig.suptitle(titulo, fontsize=12.5, fontweight="semibold",
                 y=1 - .28 * dy, color=INK)
    if sub:
        fig.text(.5, 1 - 1.25 * dy, sub, ha="center", fontsize=8.6, color=INK2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / nome, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  {nome}")


def mapa(ax, xs, ys, M, cmap, norm=None, **kw):
    return ax.pcolormesh(xs * KM, ys * KM, M, cmap=cmap, norm=norm,
                         shading="auto", rasterized=True, **kw)


def contorno_gl(ax, bx, by, mask, rocha=True):
    """
    Contornos do BedMachine.

    A linha de aterramento é somente a fronteira entre gelo aterrado e
    plataforma. `contour(mask == 2)` também contornaria afloramentos rochosos do
    interior; por isso a rocha recebe estilo próprio e o vermelho fica reservado
    à fronteira grounded/floating.
    """
    # NaN em tudo que não é aterrado nem plataforma. Com rocha e oceano
    # valendo 0 num campo (+1, 0, −1), o contorno de nível 0 passava POR CIMA
    # deles e continuava circulando cada afloramento. Marcados como NaN, ficam
    # invisíveis ao contorno e só sobra a fronteira aterrado/plataforma — que é
    # a definição da linha de aterramento no BedMachine.
    gl = np.where(mask == 2, 1.0, np.where(mask == 3, -1.0, np.nan))
    ax.contour(bx * KM, by * KM, gl, levels=[0.0], colors="#8b0000",
               linewidths=.9, zorder=6)
    ax.contour(bx * KM, by * KM, np.isin(mask, (1, 2, 3, 4)).astype(float),
               levels=[.5], colors="#333333", linewidths=.6, zorder=6)
    if rocha and (mask == 1).any():
        ax.contourf(bx * KM, by * KM, (mask == 1).astype(float), levels=[.5, 1.5],
                    colors=["#9a9186"], zorder=5)


def eixo_mapa(ax, ext):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (km, EPSG:3031)"); ax.set_ylabel("y (km)")


def _n_celulas_do_balanco(c) -> int:
    """Nº de células do balanço de massa VIGENTE da estação."""
    cands = sorted((c.paths.outputs_dir / "experiments").glob("*mass*/mass_balance.json"))
    cands += sorted(c.paths.tables.glob("mass_balance.json"))
    if not cands:
        raise FileNotFoundError(f"sem mass_balance.json em {c.paths.outputs_dir}")
    return int(json.loads(cands[-1].read_text(encoding="utf-8"))["n_cells"])


def _grid_do_experimento(c, padrao, arquivo, n_alvo):
    """
    Escolhe o experimento cujo grid tem EXATAMENTE `n_alvo` células.

    Selecionar por nome (`sorted(...)[-1]`) pode eleger um experimento obsoleto e
    comparar estações com domínios diferentes sem falhar na leitura.

    Amarrar ao `n_cells` do balanço de massa liga a figura ao NÚMERO PUBLICADO
    por construção. Se nenhum bater, é erro, não escolha silenciosa.
    """
    achados = []
    for p in sorted((c.paths.outputs_dir / "experiments").glob(f"{padrao}/{arquivo}")):
        n = len(pd.read_parquet(p, columns=["x"]))
        achados.append((p, n))
        if n == n_alvo:
            return pd.read_parquet(p), p
    raise RuntimeError(
        f"nenhum {arquivo} com {n_alvo} células em {c.paths.outputs_dir}; "
        f"achei {[(p.parent.name, n) for p, n in achados]}")


def _so_aterrado(g, cfg):
    """Descarta células cujo centro não cai em gelo aterrado no BedMachine."""
    from thwaites.viz.produtos import carregar_bedmachine, amostrar as _am
    bmp = sorted(cfg.paths.data_dir.glob("*BedMachine*.nc"))[0]
    bx, by, B = carregar_bedmachine(bmp, g.x.min() - 5e3, g.x.max() + 5e3,
                                    g.y.min() - 5e3, g.y.max() + 5e3,
                                    vars=("mask",))
    cls = _am(B["mask"].astype(int), bx, by, g.x.to_numpy(), g.y.to_numpy())
    fora = int((cls != 2).sum())
    if fora:
        print(f"    recorte ao aterrado: -{fora} células fora da classe 2")
    return g[cls == 2].copy()


def carregar():
    cfg = {l: load_config(None if l == "jja" else l) for l in ("jja", "djf")}
    D = {}
    for l, c in cfg.items():
        n = _n_celulas_do_balanco(c)
        g = pd.read_parquet(c.paths.interim / "dhdt_grid.parquet")
        f, fp = _grid_do_experimento(c, "*firn*", "firn_corrected_grid.parquet", n)
        # a advecção parte do mesmo grid de nós, mas antes do recorte de
        # cobertura — casa com o grid bruto, não com o do balanço
        a, ap = _grid_do_experimento(c, "*adv*", "dhdt_lagrangian_grid.parquet",
                                     _n_advec(c))
        # RECORTE DE COBERTURA. `dhdt_grid.parquet` traz as 15.375 células da
        # grade regular, mas o interpolador PREENCHE tudo — inclusive onde não
        # há nó nenhum por perto. Mapear a grade bruta mostra extrapolação com
        # a mesma tinta do dado observado, e o mapa passa a afirmar cobertura
        # que o levantamento não tem. É o mesmo recorte que o balanço de massa
        # aplica antes de somar (daí as ~8.02k células do resultado publicado).
        nq = pd.read_parquet(c.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
                             columns=["x", "y"])
        g = apply_coverage_mask(g, nq, c.mass_balance.coverage_dist_m)
        # 48 das 8.024 células caíam sobre classes NÃO aterradas (7 em rocha,
        # 5 em oceano, 36 em plataforma) — o centro da célula interpolada cai
        # num pixel não-aterrado ainda que os nós que a alimentam sejam
        # válidos. São 0,6% e não movem estatística, mas pintar dh/dt de gelo
        # aterrado sobre rocha exposta é afirmar o que a máscara nega.
        g = _so_aterrado(g, c)
        bm = pd.read_parquet(c.paths.dhdt_dir / "shelf_basal_melt.parquet")
        print(f"  {l.upper()}: firn={fp.parent.name} ({len(f):,}) | "
              f"adv={ap.parent.name} ({len(a):,}) | ṁ_b={len(bm):,}")
        D[l] = {"cfg": c, "grid": g, "firn": f, "adv": a, "melt": bm}
    return D


def _n_advec(c) -> int:
    """Maior grid de advecção com a extensão da ROI atual (a antiga é menor)."""
    melhor = 0
    for p in sorted((c.paths.outputs_dir / "experiments").glob("*adv*/dhdt_lagrangian_grid.parquet")):
        d = pd.read_parquet(p, columns=["x", "y"])
        if d.x.max() - d.x.min() > 5e5:       # ROI expandida (ASE), não Thwaites
            melhor = max(melhor, len(d))
    if not melhor:
        raise RuntimeError(f"sem grid de advecção na ROI atual em {c.paths.outputs_dir}")
    return melhor


# ==================================================================== FIG 0
LAB = {"jja": "JJA · inverno austral", "djf": "DJF · verão austral"}


def _dhdt_painel(fig, a, D, l, bx, by, B, ext, vmin=-3.5, vmax=1.0,
                 titulo=None, barra=True):
    """Um painel de dh/dt: campo, linha de aterramento, costa e barra de escala."""
    xs, ys, R = para_grade(D[l]["grid"], "pred")
    im = mapa(a, xs, ys, R, DIV_THIN,
              norm=TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
    if barra:
        cb = fig.colorbar(im, ax=a, shrink=.86, pad=.02)
        cb.set_label("dh/dt (m ano⁻¹)")
        cb.ax.axhline(0, color=INK, lw=.8)
    contorno_gl(a, bx, by, B["mask"]); eixo_mapa(a, ext)
    a.set_title(titulo if titulo else LAB[l])
    return im


def _escala(a, ext, km=100):
    """
    Barra de escala, no canto inferior DIREITO.

    Ficava à esquerda e caía em cima da caixa de estatísticas — as duas
    ancoradas no mesmo canto.
    """
    x0 = ext[1] - .30 * (ext[1] - ext[0])
    y0 = ext[2] + .055 * (ext[3] - ext[2])
    a.plot([x0, x0 + km], [y0, y0], color=INK, lw=3, solid_capstyle="butt", zorder=10)
    a.text(x0 + km / 2, y0 + .012 * (ext[3] - ext[2]), f"{km} km", ha="center",
           va="bottom", fontsize=8, color=INK, zorder=10)


def fig0_individuais(D, bx, by, B, ext):
    """Um mapa por estação, em folha própria — o produto principal do projeto."""
    for l in ("jja", "djf"):
        fig, a = plt.subplots(figsize=(7.6, 7.8))
        _dhdt_painel(fig, a, D, l, bx, by, B, ext,
                     titulo="Taxa de elevação da superfície")
        _escala(a, ext)
        a.plot([], [], color="#8b0000", lw=1.1, label="linha de aterramento")
        a.plot([], [], color="#333333", lw=.8, label="limite do gelo")
        a.legend(fontsize=7.4, labelcolor=INK2, frameon=True, facecolor="#fcfcfb",
                 edgecolor=GRID, framealpha=.93, loc="upper right")
        fim(fig, f"F0_dhdt_{l.upper()}.png",
            f"Amundsen Sea Embayment — {LAB[l]}",
            "ICESat-2/ATL06 v007 · 2019–2025 · gelo aterrado, nós validados · "
            "interpolação por validação cruzada em blocos")


def fig0c_comparacao(D, bx, by, B, ext):
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 5.6))
    for k, l in enumerate(("jja", "djf")):
        _dhdt_painel(fig, ax[k], D, l, bx, by, B, ext,
                     titulo=f"{'ab'[k]}. {LAB[l]}")
    xj, yj, Rj = para_grade(D["jja"]["grid"], "pred")
    _, _, Rd = para_grade(D["djf"]["grid"], "pred")
    dif = Rd - Rj
    c = ax[2]
    im = mapa(c, xj, yj, dif, DIV_THIN,
              norm=TwoSlopeNorm(vcenter=0, vmin=-.6, vmax=.6))
    cb = fig.colorbar(im, ax=c, shrink=.86, pad=.02)
    cb.set_label("DJF − JJA (m ano⁻¹)"); cb.ax.axhline(0, color=INK, lw=.8)
    contorno_gl(c, bx, by, B["mask"]); eixo_mapa(c, ext)
    c.set_title("c. Diferença entre as janelas")
    fim(fig, "F0_dhdt_comparacao.png",
        "Taxa de elevação — inverno, verão e a diferença entre as janelas",
        "mesma cadeia, mesma máscara e mesmos limiares nas duas estações · "
        "a diferença NÃO é sazonalidade do gelo, é efeito da janela de amostragem")


# ==================================================================== FIG 1
def fig1_haf(D, bx, by, B, ext):
    haf = altura_de_flutuacao(B["bed"], B["thickness"])
    haf = np.where(B["mask"] == 2, haf, np.nan)          # só gelo aterrado

    fig, ax = plt.subplots(2, 2, figsize=(10.2, 9.0))
    a = ax[0, 0]
    im = mapa(a, bx, by, haf, SEQ, norm=LogNorm(20, 3000))
    fig.colorbar(im, ax=a, shrink=.85, pad=.02).set_label("HAF (m de gelo)")
    contorno_gl(a, bx, by, B["mask"]); eixo_mapa(a, ext)
    a.set_title("a. Altura acima da flutuação")

    for k, (l, lab) in enumerate((("jja", "JJA (inverno)"), ("djf", "DJF (verão)"))):
        g = D[l]["firn"]
        xs, ys, R = para_grade(g, "dhdt_ice")
        # HAF amostrado nas células da grade de dh/dt
        gx, gy = np.meshgrid(xs, ys)
        h_g = amostrar(haf, bx, by, gx.ravel(), gy.ravel()).reshape(gx.shape)
        T = tempo_ate_desaterrar(h_g, R)
        b = ax[0, 1] if k == 0 else ax[1, 0]
        im = mapa(b, xs, ys, np.clip(T, 0, 2000), URG, norm=LogNorm(20, 2000))
        fig.colorbar(im, ax=b, shrink=.85, pad=.02).set_label("anos até HAF = 0")
        contorno_gl(b, bx, by, B["mask"]); eixo_mapa(b, ext)
        n = np.isfinite(T)
        b.set_title(f"{'b' if k == 0 else 'c'}. Tempo até desaterrar — {lab}")
        b.text(.03, .04, f"mediana {np.nanmedian(T[n]):.0f} anos\n"
                         f"{100*np.mean(T[n] < 200):.1f}% < 200 anos",
               transform=b.transAxes, fontsize=7.6, color=INK2, va="bottom",
               bbox=dict(fc="#fcfcfb", ec=GRID, alpha=.92, pad=2.4))

    d = ax[1, 1]
    im = mapa(d, bx, by, np.where(haf < 200, haf, np.nan), URG, vmin=0, vmax=200)
    fig.colorbar(im, ax=d, shrink=.85, pad=.02).set_label("HAF (m)")
    contorno_gl(d, bx, by, B["mask"]); eixo_mapa(d, ext)
    d.set_title("d. Zona crítica — HAF < 200 m")

    fim(fig, "F1_altura_de_flutuacao.png",
        "Altura acima da flutuação e proximidade do desaterramento",
        "HAF = H − (ρ_mar/ρ_gelo)·max(−leito, 0)  ·  BedMachine v4.1 + dh/dt corrigido de firn  ·  "
        "extrapolação linear da taxa observada, NÃO previsão")


# ==================================================================== FIG 2
def fig2_portoes(D, bx, by, B, ext, vx, vy, vxs, vys, anos):
    gxg, gyg = np.meshgrid(vxs, vys)
    Hv = amostrar(B["thickness"], bx, by, gxg.ravel(), gyg.ravel()).reshape(gxg.shape)
    Mv = amostrar(B["mask"].astype(float), bx, by, gxg.ravel(), gyg.ravel()).reshape(gxg.shape)
    Hv = np.where(np.isin(Mv, (2, 3)), Hv, np.nan)

    # portões perpendiculares ao fluxo principal, a montante da GL
    portoes = [("P1", -1560e3, -560e3, -1560e3, -380e3),
               ("P2", -1490e3, -600e3, -1490e3, -400e3),
               ("P3", -1420e3, -640e3, -1420e3, -420e3)]

    fig = plt.figure(figsize=(11.0, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
    a = fig.add_subplot(gs[0, 0])
    v_ult = np.hypot(vx[-1], vy[-1])
    im = mapa(a, vxs, vys, v_ult, SEQ, norm=LogNorm(10, 4000))
    fig.colorbar(im, ax=a, shrink=.85, pad=.02).set_label("velocidade (m ano⁻¹)")
    contorno_gl(a, bx, by, B["mask"]); eixo_mapa(a, ext)
    for nome, x0, y0, x1, y1 in portoes:
        a.plot([x0 * KM, x1 * KM], [y0 * KM, y1 * KM], color=CAT[1], lw=2.4,
               solid_capstyle="round", zorder=8)
        a.text(x0 * KM, y1 * KM + 12, nome, ha="center", fontsize=8, color=CAT[1],
               fontweight="semibold", zorder=9,
               bbox=dict(fc="#fcfcfb", ec="none", alpha=.85, pad=1.4))
    a.set_title(f"a. Velocidade em {anos[-1]:.0f} e portões de fluxo")

    b = fig.add_subplot(gs[0, 1])
    dv = np.hypot(vx[-1], vy[-1]) - np.hypot(vx[0], vy[0])
    im = mapa(b, vxs, vys, dv, DIV_MELT, norm=TwoSlopeNorm(vcenter=0, vmin=-300, vmax=300))
    cb = fig.colorbar(im, ax=b, shrink=.85, pad=.02)
    cb.set_label("Δ velocidade (m ano⁻¹)"); cb.ax.axhline(0, color=INK, lw=.8)
    contorno_gl(b, bx, by, B["mask"]); eixo_mapa(b, ext)
    b.set_title(f"b. Mudança de velocidade {anos[0]:.0f} → {anos[-1]:.0f}")

    c = fig.add_subplot(gs[1, :])
    tab = {}
    for k, (nome, x0, y0, x1, y1) in enumerate(portoes):
        q = [fluxo_por_portao(x0, y0, x1, y1, vxs, vys, vx[i], vy[i], Hv)
             for i in range(len(anos))]
        tab[nome] = q
        c.plot(anos, q, marker="o", ms=5, lw=2, color=CAT[k], label=nome)
        c.text(anos[-1] + .08, q[-1], nome, color=CAT[k], fontsize=8.4,
               va="center", fontweight="semibold")
    c.set_xlabel("ano"); c.set_ylabel("descarga (Gt ano⁻¹)")
    c.set_title("c. Descarga através de cada portão")
    c.grid(alpha=.7); c.set_axisbelow(True)
    c.set_xlim(anos[0] - .2, anos[-1] + .6)

    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "fluxo_portoes.json").write_text(json.dumps(
        {"anos": anos.tolist(), "Gt_por_ano": tab,
         "nota": ("Q = integral de H·(v·n̂) dl · ρ_gelo; só a componente NORMAL "
                  "ao portão atravessa. H do BedMachine (época ~2015), fixo no "
                  "tempo — a variação mostrada é de VELOCIDADE, não de espessura.")},
        indent=2, ensure_ascii=False), encoding="utf-8")

    fim(fig, "F2_fluxo_por_portao.png",
        "Descarga por portão de fluxo e evolução da velocidade",
        "ITS_LIVE anual (120 m) · espessura BedMachine fixa no tempo · "
        "Q = ∫H(v·n̂)dl·ρ — só a componente normal ao portão")


# ==================================================================== FIG 3
def fig3_lagrangiano(D, bx, by, B, ext):
    fig, ax = plt.subplots(2, 2, figsize=(10.2, 9.0))
    for k, (l, lab) in enumerate((("jja", "JJA (inverno)"), ("djf", "DJF (verão)"))):
        a = D[l]["adv"]
        xs, ys, L = para_grade(a, "dhdt_lagrangian")
        _, _, Av = para_grade(a, "advection")
        u = ax[0, k]
        im = mapa(u, xs, ys, L, DIV_THIN, norm=TwoSlopeNorm(vcenter=0, vmin=-3.5, vmax=1.0))
        cb = fig.colorbar(im, ax=u, shrink=.85, pad=.02)
        cb.set_label("DH/Dt (m ano⁻¹)"); cb.ax.axhline(0, color=INK, lw=.8)
        contorno_gl(u, bx, by, B["mask"]); eixo_mapa(u, ext)
        u.set_title(f"{'a' if k == 0 else 'b'}. dh/dt lagrangiano — {lab}")

        d = ax[1, k]
        im = mapa(d, xs, ys, Av, DIV_THIN, norm=TwoSlopeNorm(vcenter=0, vmin=-1.2, vmax=1.2))
        cb = fig.colorbar(im, ax=d, shrink=.85, pad=.02)
        cb.set_label("v·∇h (m ano⁻¹)"); cb.ax.axhline(0, color=INK, lw=.8)
        contorno_gl(d, bx, by, B["mask"]); eixo_mapa(d, ext)
        d.set_title(f"{'c' if k == 0 else 'd'}. Termo de advecção — {lab}")

    fim(fig, "F3_lagrangiano.png",
        "Taxa de elevação no referencial da parcela e o termo de advecção",
        "DH/Dt = ∂h/∂t + v·∇h  ·  a advecção é o que separa o que a parcela vive "
        "do que o ponto fixo vê")


# ==================================================================== FIG 4
def fig4_basal(D, bx, by, B, ext):
    # Enquadramento pelas PARCELAS: usar a ROI inteira deixaria dois terços do
    # painel vazios e o produto de plataforma ilegível.
    todos = pd.concat([D[l]["melt"][["x_ref", "y_ref"]] for l in ("jja", "djf")])
    m = 25e3
    ext4 = ((todos.x_ref.min() - m) * KM, (todos.x_ref.max() + m) * KM,
            (todos.y_ref.min() - m) * KM, (todos.y_ref.max() + m) * KM)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 5.8))
    med = {}
    for k, (l, lab) in enumerate((("jja", "JJA (inverno)"), ("djf", "DJF (verão)"))):
        d = D[l]["melt"]
        sel = np.isfinite(d["basal_melt"])
        if "reliable" in d.columns:
            sel &= d["reliable"].to_numpy(bool)
        g = d[sel]
        med[l] = g
        a = ax[k]
        s = a.scatter(g["x_ref"] * KM, g["y_ref"] * KM, c=g["basal_melt"], s=7,
                      cmap=DIV_MELT, norm=TwoSlopeNorm(vcenter=0, vmin=-8, vmax=32), lw=0,
                      rasterized=True)
        cb = fig.colorbar(s, ax=a, shrink=.86, pad=.02)
        cb.set_label("ṁ_b (m ano⁻¹)"); cb.ax.axhline(0, color=INK, lw=.8)
        contorno_gl(a, bx, by, B["mask"]); eixo_mapa(a, ext4)
        a.set_title(f"{'a' if k == 0 else 'b'}. Derretimento basal — {lab}")
        a.text(.03, .04, f"n = {len(g):,}\nmediana {g['basal_melt'].median():+.2f} m/ano",
               transform=a.transAxes, fontsize=7.6, color=INK2, va="bottom",
               bbox=dict(fc="#fcfcfb", ec=GRID, alpha=.92, pad=2.4))

    c = ax[2]
    for l, lab, col in (("jja", "JJA", CAT[0]), ("djf", "DJF", CAT[1])):
        v = med[l]["basal_melt"].to_numpy()
        v = v[np.isfinite(v) & (v > -20) & (v < 60)]
        c.hist(v, bins=70, histtype="step", lw=1.8, color=col, label=lab,
               density=True)
        c.axvline(np.median(v), color=col, lw=1, ls="--")
    c.axvline(0, color=MUTED, lw=.9)
    c.set_xlabel("ṁ_b (m ano⁻¹)"); c.set_ylabel("densidade")
    c.set_title("c. Distribuição")
    c.legend(fontsize=8, labelcolor=INK2); c.grid(alpha=.7); c.set_axisbelow(True)

    fim(fig, "F4_derretimento_basal.png",
        "Derretimento basal por parcela — inverno e verão",
        "ṁ_b = a_s − DH/Dt − H·∇·v (Adusumilli et al. 2018) · referencial "
        "lagrangiano · positivo = derretimento · produto EXPLORATÓRIO (máscara estática)")


# ==================================================================== FIG 5
def fig5_transversais(D, bx, by, B, ext, vx, vy, vxs, vys):
    """
    Cortes PERPENDICULARES ao tronco, centrados no máximo local de velocidade.

    A posição dos cortes vem do dado: para cada latitude, o centro é o pixel
    mais rápido daquela linha. Isso evita cortes subjetivos sobre o interior
    lento em vez do tronco glacial.

    Também não usa eixo duplo. Velocidade e dh/dt têm escalas e unidades
    diferentes, e sobrepô-las num só painel com dois eixos y deixa a
    comparação entre elas à mercê do intervalo escolhido para cada eixo — dois
    painéis empilhados que dividem o eixo x dizem a mesma coisa sem essa
    arbitrariedade.
    """
    v_ult = np.hypot(vx[-1], vy[-1])
    meia = 115e3
    cortes = []
    for nome, yalvo in (("T1", -300e3), ("T2", -450e3), ("T3", -500e3)):
        i = int(np.argmin(np.abs(vys - yalvo)))
        lin = v_ult[i]
        if not np.isfinite(lin).any():
            continue
        xc = vxs[int(np.nanargmax(lin))]
        larg = np.nansum(lin > .5 * np.nanmax(lin)) * abs(vxs[1] - vxs[0]) / 1e3
        cortes.append((nome, xc - meia, xc + meia, vys[i], larg))

    fig = plt.figure(figsize=(11.8, 10.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.75, .85, .85], hspace=.42)
    a = fig.add_subplot(gs[0, :])
    im = mapa(a, vxs, vys, v_ult, SEQ, norm=LogNorm(10, 4000))
    fig.colorbar(im, ax=a, shrink=.9, pad=.015).set_label("velocidade (m ano⁻¹)")
    contorno_gl(a, bx, by, B["mask"]); eixo_mapa(a, ext)
    for k, (nome, x0, x1, y, larg) in enumerate(cortes):
        a.plot([x0 * KM, x1 * KM], [y * KM, y * KM], color=CAT[k], lw=2.6,
               solid_capstyle="round", zorder=8)
        a.text(x1 * KM + 10, y * KM, nome, color=CAT[k], fontsize=9,
               va="center", fontweight="semibold", zorder=9,
               bbox=dict(fc="#fcfcfb", ec="none", alpha=.85, pad=1.4))
    a.set_title("a. Cortes perpendiculares ao tronco, centrados no máximo local "
                "de velocidade")

    grids = {l: para_grade(D[l]["firn"], "dhdt_ice") for l in ("jja", "djf")}
    for k, (nome, x0, x1, y, larg) in enumerate(cortes):
        px = np.linspace(x0, x1, 240)
        py = np.full_like(px, y)
        d_km = (px - (x0 + x1) / 2) * KM          # distância ao eixo do tronco

        b = fig.add_subplot(gs[1, k])
        vv = amostrar(v_ult, vxs, vys, px, py)
        b.fill_between(d_km, 0, vv, color=CAT[k], alpha=.20, lw=0)
        b.plot(d_km, vv, color=CAT[k], lw=1.9)
        b.set_ylabel("velocidade (m ano⁻¹)" if k == 0 else "")
        b.set_title(f"{'bcd'[k]}. {nome}  ·  y = {y*KM:.0f} km  ·  "
                    f"zona rápida {larg:.0f} km", fontsize=9)
        b.set_xticklabels([]); b.grid(alpha=.6); b.set_axisbelow(True)
        b.set_xlim(d_km[0], d_km[-1])

        c = fig.add_subplot(gs[2, k])
        for l, lab, col in (("jja", "JJA (inverno)", CAT[0]), ("djf", "DJF (verão)", CAT[1])):
            xs, ys, R = grids[l]
            c.plot(d_km, amostrar(R, xs, ys, px, py), color=col, lw=1.9, label=lab)
        c.axhline(0, color=MUTED, lw=.8, ls=":")
        c.set_ylabel("dh/dt (m ano⁻¹)" if k == 0 else "")
        c.set_xlabel("distância ao eixo do tronco (km)")
        c.grid(alpha=.6); c.set_axisbelow(True); c.set_xlim(d_km[0], d_km[-1])
        if k == 0:
            c.legend(fontsize=7.6, labelcolor=INK2, loc="lower left")

    fim(fig, "F5_perfis_transversais.png",
        "Cortes transversais ao tronco — alargamento da zona rápida",
        "posição de cada corte definida pelo MÁXIMO de velocidade da sua latitude · "
        "largura = trecho acima de 50% do máximo · dh/dt corrigido de firn")


# ==================================================================== FIG 6
def fig6_janelas(D, bx, by, B, ext):
    """
    Evolução espacial da taxa em janelas móveis de 4 anos, passo de 1.

    Os nós das janelas saem de `compute_tile_dhdt` sobre os tiles brutos e não
    passam pelo filtro de `run_qc_report`; por isso são recortados ao conjunto
    validado da estação antes de virar mapa.
    """
    import json as _json

    fig, ax = plt.subplots(2, 4, figsize=(15.0, 8.4))
    for r, l in enumerate(("jja", "djf")):
        c = D[l]["cfg"]
        rel = _json.loads((c.paths.tables / "dhdt_janelas.json").read_text(encoding="utf-8"))
        qc = pd.read_parquet(c.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
                             columns=["x", "y"])
        chave = set(zip(np.rint(qc.x).astype(np.int64), np.rint(qc.y).astype(np.int64)))
        for k, (nome, info) in enumerate(sorted(rel["janelas"].items())):
            nd = pd.read_parquet(c.paths.dhdt_dir / "janelas" / info["arquivo"],
                                 columns=["x", "y", "dhdt"])
            manter = np.fromiter(
                ((x, y) in chave for x, y in zip(np.rint(nd.x).astype(np.int64),
                                                 np.rint(nd.y).astype(np.int64))),
                bool, len(nd))
            nd = nd[manter]
            xs, ys, M = para_grade(nd, "dhdt")
            a = ax[r, k]
            im = mapa(a, xs, ys, M, DIV_THIN,
                      norm=TwoSlopeNorm(vcenter=0, vmin=-3.0, vmax=1.0))
            contorno_gl(a, bx, by, B["mask"]); eixo_mapa(a, ext)
            fim_lab = nome.split("-")[1]
            rot = nome.replace("-", "–")
            if fim_lab == "2026":
                rot += "*"
            a.set_title(f"{LAB[l].split(chr(183))[0].strip()} · {rot}", fontsize=9.4)
            if k:
                a.set_ylabel("")
            if r == 0:
                a.set_xlabel("")

    cb = fig.colorbar(im, ax=ax, shrink=.62, pad=.015, aspect=34)
    cb.set_label("dh/dt (m ano⁻¹)"); cb.ax.axhline(0, color=INK, lw=.8)

    fig.suptitle("Evolução da taxa de elevação em janelas móveis",
                 fontsize=12.5, fontweight="semibold", y=.985, color=INK)
    fig.text(.5, .952,
             "janelas de 4 anos com passo de 1 · nós recortados ao conjunto validado · "
             "* a última janela termina em 2025,7 — o registro não chega a 2026",
             ha="center", fontsize=8.6, color=INK2)
    fig.text(.5, .012,
             "Janelas vizinhas compartilham 3 dos 4 anos: mapas consecutivos NÃO são "
             "amostras independentes e a sequência não quantifica aceleração.",
             ha="center", fontsize=8, color="#a8481c")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "F6_janelas_moveis.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  F6_janelas_moveis.png")


def main():
    cfg0 = load_config()
    setup_logging(cfg0.paths.logs, level=cfg0.logging.level, run_name="produto_figuras")
    print("carregando produtos...")
    D = carregar()
    g = D["jja"]["grid"]
    x0, x1 = g.x.min() - 8e3, g.x.max() + 8e3
    y0, y1 = g.y.min() - 8e3, g.y.max() + 8e3
    ext = (x0 * KM, x1 * KM, y0 * KM, y1 * KM)

    bmp = sorted(cfg0.paths.data_dir.glob("*BedMachine*.nc"))[0]
    bx, by, B = carregar_bedmachine(bmp, x0, x1, y0, y1)
    print(f"  BedMachine: {B['bed'].shape}")

    vpath = cfg0.paths.data_dir / "velocity_itslive_annual.nc"
    vxs, vys, anos, vx, vy, _ = carregar_velocidade_anual(vpath, x0, x1, y0, y1,
                                                          decimar=4)
    print(f"  ITS_LIVE: {vx.shape} | épocas {anos[0]:.0f}-{anos[-1]:.0f}")

    print("figuras:")
    fig0_individuais(D, bx, by, B, ext)
    fig0c_comparacao(D, bx, by, B, ext)
    fig1_haf(D, bx, by, B, ext)
    fig2_portoes(D, bx, by, B, ext, vx, vy, vxs, vys, anos)
    fig3_lagrangiano(D, bx, by, B, ext)
    fig4_basal(D, bx, by, B, ext)
    fig5_transversais(D, bx, by, B, ext, vx, vy, vxs, vys)
    fig6_janelas(D, bx, by, B, ext)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
