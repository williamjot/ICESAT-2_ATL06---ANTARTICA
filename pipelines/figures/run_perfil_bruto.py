"""
pipelines/run_perfil_bruto.py
=============================
Rebaixamento por latitude direto dos segmentos ATL06 — **sem ajuste de nó, sem
interpolação, sem krigagem**.

    data/<estação>/tiles/*.parquet
        -> data/<estação>/interim/celulas_bruto_<cell>m.parquet   (cache)
        -> outputs/comparacao_sazonal/F11_perfil_bruto_por_latitude.png
        -> outputs/comparacao_sazonal/F12_bruto_vs_nos.png
        -> outputs/comparacao_sazonal/perfil_bruto.json

Por que existe, tendo F10
-------------------------
O F10 (`run_figuras_massa.fig10`) desenha a mesma ideia, mas parte de
`serie_anual.parquet`, que é **saída do ajuste espaço-temporal por nó**. Ou
seja: o rebaixamento que ele mostra já passou por um modelo que impõe
`h = β₀ + f(x,y) + β₁Δt + ½β₂Δt²`. Se o modelo estivesse errado, a figura
mostraria o erro do modelo com a mesma aparência de sinal.

Aqui a única operação sobre o dado é **mediana**. Nenhum parâmetro é ajustado,
nenhum termo temporal é imposto. Se as curvas se separam na ordem do tempo, é
porque os segmentos medidos estão mais baixos ano a ano.

O viés que esta figura tem de evitar
------------------------------------
A tentação é: filtrar uma faixa longitudinal, agrupar por caixa de latitude e
tirar a mediana da altura por ano. **Isso não mede rebaixamento.** Dentro de uma
caixa de latitude de 0,05° (~5,6 km em latitude, e a faixa tem ~50 km em
longitude) há centenas de metros de relevo. Se o conjunto de LUGARES amostrados
muda entre anos — e muda, porque a cobertura do ICESat-2 não é idêntica ano a
ano — a mediana muda sem que a superfície tenha se movido. O sinal seria
composição de amostra, não geofísica.

Duas defesas, ambas necessárias:

1. **`h_res = h_corr − REMA`, não `h_corr`.** O REMA é subtraído no PRÓPRIO
   ponto, com o DEM de 32 m. Como é estático, `d(h_res)/dt = d(h_corr)/dt`: a
   subtração não toca a taxa, mas remove o relevo local, que é o que
   contaminaria a média espacial.

2. **Anomalia por CÉLULA FIXA, com painel balanceado.** O domínio é dividido em
   células de 2 km; para cada (célula, ano) toma-se a mediana de `h_res`; e a
   anomalia é relativa ao PRIMEIRO ano DAQUELA célula. Só entram células
   observadas em TODOS os anos. Assim o que é comparado entre anos é sempre o
   mesmo conjunto de lugares — a composição de amostra deixa de poder gerar
   sinal. O preço é descartar células de cobertura incompleta, e o número
   descartado está no JSON.

Célula de 2 km: grande o bastante para conter dezenas de segmentos de 40 m de
cada passagem (a mediana por célula-ano é estável) e pequena o bastante para ser
subdivisão da caixa de latitude de 0,05°, que tem ~5,6 km. É também múltiplo
exato do tile de 50 km, o que importa: os arquivos de tile têm halo e um ponto
aparece em vários deles, então só o NÚCLEO de cada tile é lido. Com a grade
alinhada, nenhuma célula fica repartida entre tiles e a mediana por célula é
exata — não é mediana de medianas.

Limitações declaradas
---------------------
- **A faixa longitudinal é um corte fixo, não segue o tronco.** O tronco da
  Thwaites não é meridional, então uma faixa de longitude constante cruza o
  fluxo rápido em algumas latitudes e a margem lenta em outras. Um corredor
  guiado pelo máximo de velocidade do ITS_LIVE seria fisicamente melhor; não é
  o que está aqui. Duas sensibilidades são reportadas e **não devem ser
  somadas**: variar a LARGURA com o centro fixo mede o método (−15,3 a −17,1 m,
  ±6 %); variar o CENTRO muda de lugar, e a dispersão que isso produz é
  geografia — fora da janela do setor de saída da Thwaites o corte chega a
  entrar em Pine Island, onde o sinal é outro.
- **Viés entre feixes não é corrigido.** Medido em ≤ 0,056 m
  (`outputs/<estação>/tables/interbeam_bias.csv`), duas ordens de grandeza
  abaixo do rebaixamento discutido. A variante só com feixes fortes está no
  JSON como verificação.
- A curva é rebaixamento de SUPERFÍCIE. Não é perda de massa: falta separar ar
  no firn e o movimento do embasamento (GIA). Ver METODOS.md §4.

Uso:
    python pipelines/run_perfil_bruto.py
    python pipelines/run_perfil_bruto.py --recalcular    # ignora o cache
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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

INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e3e3e0"
# rampa temporal de hue única: ano é ordinal, e claro->escuro deixa a ordem
# legível mesmo sem consultar a legenda; arco-íris embaralharia a leitura
ANOS = LinearSegmentedColormap.from_list(
    "anos", ["#bcd7ea", "#7fb0d0", "#4a86b4", "#2a5f8c", "#123f63", "#06243c"])
OUT = ROOT / "outputs" / "comparacao_sazonal"

CELL_M = 2000.0          # lado da célula fixa (ver docstring)
MIN_PTS_CELULA = 10      # segmentos mínimos para a mediana de (célula, ano)
DLAT = 0.05              # caixa de latitude, em graus
MIN_CELULAS_CAIXA = 5    # células mínimas para a mediana da caixa de latitude
FORTES = (1, 3, 5)       # feixes fortes (verificação de viés entre feixes)

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": .7,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "grid.color": GRID, "grid.linewidth": .6,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
})


# --------------------------------------------------------------- agregação
def agregar(cfg, log, cell_m=CELL_M, recalcular=False):
    """
    Mediana de `h_res` por (célula de `cell_m`, ano), varrendo os tiles.

    Só o núcleo de cada tile entra (os arquivos têm halo — ver docstring do
    módulo). O resultado é pequeno (~10⁵ linhas) e fica em cache no `interim`
    da estação, porque a varredura lê ~11 GB e leva minutos.
    """
    cache = cfg.paths.interim / f"celulas_bruto_{int(cell_m)}m.parquet"
    if cache.exists() and not recalcular:
        log.info(f"cache: {cache.name}")
        return pd.read_parquet(cache)

    man = load_manifest(cfg)
    cols = ["x", "y", "lat", "lon", "t_year", "h_res", "beam"]
    partes, t0, n_lidos = [], time.time(), 0
    for k, e in enumerate(man, 1):
        d = pd.read_parquet(cfg.paths.tiles_dir / e["file"], columns=cols)
        # núcleo do tile: [x_min, x_max) × [y_min, y_max) — partição exata
        d = d[(d.x >= e["x_min"]) & (d.x < e["x_max"]) &
              (d.y >= e["y_min"]) & (d.y < e["y_max"])]
        d = d[np.isfinite(d.h_res)]
        if d.empty:
            continue
        n_lidos += len(d)
        d = d.assign(
            ix=np.floor(d.x.to_numpy() / cell_m).astype(np.int32),
            iy=np.floor(d.y.to_numpy() / cell_m).astype(np.int32),
            ano=np.floor(d.t_year.to_numpy()).astype(np.int16),
            forte=d.beam.isin(FORTES).to_numpy())
        g = d.groupby(["ix", "iy", "ano"], sort=False).agg(
            h_res=("h_res", "median"),
            lat=("lat", "median"), lon=("lon", "median"),
            n=("h_res", "size")).reset_index()
        # a variante só-feixes-fortes é agregada em separado — um `lambda` por
        # grupo dentro do `agg` acima seria ordens de grandeza mais lento
        gf = (d[d.forte].groupby(["ix", "iy", "ano"], sort=False)
              .agg(h_res_forte=("h_res", "median"),
                   n_forte=("h_res", "size")).reset_index())
        g = g.merge(gf, on=["ix", "iy", "ano"], how="left")
        partes.append(g)
        if k % 20 == 0 or k == len(man):
            log.info(f"  tile {k}/{len(man)} · {n_lidos/1e6:.1f} M pts núcleo · "
                     f"{time.time()-t0:.0f}s")

    cel = pd.concat(partes, ignore_index=True)
    cel = cel[cel.n >= MIN_PTS_CELULA].reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cel.to_parquet(cache, index=False)
    log.info(f"{len(cel):,} (célula, ano) de {n_lidos/1e6:.1f} M pontos de núcleo "
             f"-> {cache.name}")
    return cel


# ------------------------------------------------------------------ perfil
def painel_balanceado(cel, lon0, lon1, col="h_res"):
    """
    Anomalia por célula relativa ao primeiro ano, só em células observadas em
    TODOS os anos da faixa. Devolve (df, diagnóstico).
    """
    f = cel[(cel.lon > lon0) & (cel.lon < lon1) & np.isfinite(cel[col])]
    if f.empty:
        return f, {"n_celulas": 0}
    anos = sorted(f.ano.unique())
    ncel_bruto = f.set_index(["ix", "iy"]).index.nunique()

    n_anos = f.groupby(["ix", "iy"]).ano.nunique()
    manter = n_anos.index[n_anos == len(anos)]
    f = f.merge(manter.to_frame(index=False), on=["ix", "iy"], how="inner")
    if f.empty:
        return f, {"n_celulas": 0, "n_celulas_antes_balanceamento": int(ncel_bruto)}

    base = (f[f.ano == anos[0]].set_index(["ix", "iy"])[col]
            .groupby(level=[0, 1]).first())
    idx = pd.MultiIndex.from_arrays([f.ix.to_numpy(), f.iy.to_numpy()],
                                    names=["ix", "iy"])
    f = f.assign(anom=f[col].to_numpy() - base.reindex(idx).to_numpy())
    f = f.assign(_b=(np.round(f.lat.to_numpy() / DLAT) * DLAT).round(4))
    diag = {"n_celulas": int(len(manter)),
            "n_celulas_antes_balanceamento": int(ncel_bruto),
            "fracao_mantida": float(len(manter) / max(ncel_bruto, 1)),
            "anos": [int(a) for a in anos],
            "n_pontos_atl06": int(f.n.sum())}
    return f, diag


def perfil(f):
    """Mediana da anomalia por (ano, caixa de latitude)."""
    q = f.groupby(["ano", "_b"]).agg(anom=("anom", "median"),
                                    ncel=("anom", "size")).reset_index()
    return q[q.ncel >= MIN_CELULAS_CAIXA]


def sensibilidade_faixa(cel, centros, larguras):
    """
    Rebaixamento total (último ano, mediana das 3 caixas mais a jusante) para
    várias faixas. Mede o quanto a figura depende de uma escolha arbitrária.
    """
    linhas = []
    for c in centros:
        for w in larguras:
            f, d = painel_balanceado(cel, c - w / 2, c + w / 2)
            if d["n_celulas"] < 40:
                linhas.append({"centro_lon": c, "largura_lon": w,
                               "n_celulas": d["n_celulas"], "rebaixamento_m": None})
                continue
            q = perfil(f)
            ult = q[q.ano == q.ano.max()].sort_values("_b")
            # jusante = extremo NORTE da faixa (latitude menos negativa), que é
            # onde está a linha de aterramento; `tail` depois de ordenar por _b
            linhas.append({
                "centro_lon": c, "largura_lon": w,
                "n_celulas": d["n_celulas"],
                "n_caixas_lat": int(len(ult)),
                "rebaixamento_jusante_m": float(ult.anom.tail(3).median()),
                "lat_min": float(ult._b.min()), "lat_max": float(ult._b.max())})
    return linhas


# ------------------------------------------------------------------ figuras
def _desenha(a, q, titulo, legenda=False):
    anos = sorted(q.ano.unique())
    cores = {an: ANOS(i / max(len(anos) - 1, 1)) for i, an in enumerate(anos)}
    for an in anos:
        g = q[q.ano == an].sort_values("_b")
        a.plot(g._b, g.anom, lw=1.7, color=cores[an], label=str(an))
    a.axhline(0, color=MUTED, lw=.9, ls=":")
    a.set_ylabel("mudança de elevação (m)")
    a.set_title(titulo)
    a.grid(alpha=.6)
    a.set_axisbelow(True)
    a.invert_xaxis()
    if legenda:
        a.legend(fontsize=7.6, labelcolor=INK2, ncol=4, loc="lower left",
                 title="ano", title_fontsize=7.6, frameon=True,
                 facecolor="#fcfcfb", edgecolor=GRID, framealpha=.93)


def fig11(perfis, diags, faixa):
    # sharey: a comparação entre estações é o ponto da figura, e dois eixos
    # verticais diferentes fariam curvas iguais parecerem diferentes
    fig, ax = plt.subplots(2, 1, figsize=(11.0, 8.2), sharex=True, sharey=True)
    for k, (lab, est) in enumerate((("JJA", "inverno austral"),
                                    ("DJF", "verão austral"))):
        d = diags[lab]
        _desenha(ax[k], perfis[lab],
                 f"{'ab'[k]}. {lab} · {est} — {d['n_celulas']:,} células de 2 km "
                 f"observadas em todos os {len(d['anos'])} anos "
                 f"({d['n_pontos_atl06']/1e3:,.0f} mil segmentos ATL06)",
                 legenda=(k == 0))
    ax[1].set_xlabel("latitude (°)")
    fig.tight_layout(rect=[0, .012, 1, .922])
    fig.suptitle("Rebaixamento por latitude — direto dos segmentos ATL06",
                 fontsize=12.5, fontweight="semibold", y=.985, color=INK)
    fig.text(.5, .938,
             f"sem ajuste de nó e sem interpolação · anomalia de h−REMA por célula "
             f"fixa de {CELL_M/1e3:.0f} km relativa ao primeiro ano · mediana por "
             f"caixa de {DLAT:.2f}".replace(".", ",") +
             f"° de latitude · faixa {abs(faixa[1]):.1f}° a "
             f"{abs(faixa[0]):.1f}°W".replace(".", ","),
             ha="center", fontsize=8.6, color=INK2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "F11_perfil_bruto_por_latitude.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  F11_perfil_bruto_por_latitude.png")


def fig12(perfis, nos, faixa):
    """
    O mesmo rebaixamento acumulado pelas duas rotas: segmentos crus e produto
    de nós. Serve a uma pergunta só — o ajuste por nó cria o sinal ou o
    reproduz?
    """
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)
    for k, lab in enumerate(("JJA", "DJF")):
        a = ax[k]
        q = perfis[lab]
        u = q[q.ano == q.ano.max()].sort_values("_b")
        a.plot(u._b, u.anom, lw=2.1, color="#08306b",
               label=f"segmentos ATL06 (sem nó)")
        n = nos.get(lab)
        if n is not None and len(n):
            n = n.sort_values("_b")
            a.plot(n._b, n.anom, lw=1.8, ls="--", color="#d1552a",
                   label="produto de nós (F10)")
            # diferença mediana onde as duas existem
            j = u.set_index("_b").anom.reindex(n._b.round(4).to_numpy())
            dif = float(np.nanmedian(j.to_numpy() - n.anom.to_numpy()))
            # canto superior esquerdo: com o eixo x invertido, a jusante fica à
            # esquerda e a curva desce até lá — o topo esquerdo é o vazio
            a.text(.03, .93, f"diferença mediana {dif:+.2f} m".replace(".", ","),
                   transform=a.transAxes, fontsize=8.2, color=INK2, va="top")
        a.axhline(0, color=MUTED, lw=.9, ls=":")
        a.set_title(f"{'ab'[k]}. {lab} — rebaixamento acumulado até "
                    f"{int(q.ano.max())}")
        a.set_xlabel("latitude (°)")
        a.grid(alpha=.6); a.set_axisbelow(True); a.invert_xaxis()
        if k == 0:
            a.set_ylabel("mudança de elevação (m)")
            a.legend(fontsize=8.2, labelcolor=INK2, loc="lower right",
                     frameon=True, facecolor="#fcfcfb", edgecolor=GRID,
                     framealpha=.93)
    fig.tight_layout(rect=[0, .012, 1, .888])
    fig.suptitle("As duas rotas dão a mesma curva", fontsize=12.5,
                 fontweight="semibold", y=.985, color=INK)
    fig.text(.5, .906, "o ajuste espaço-temporal por nó reproduz o que os "
                       "segmentos mostram sozinhos — não o produz",
             ha="center", fontsize=8.6, color=INK2)
    fig.savefig(OUT / "F12_bruto_vs_nos.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  F12_bruto_vs_nos.png")


def perfil_de_nos(cfg, lon0, lon1):
    """Curva do último ano pelo produto de nós — a rota que o F10 usa."""
    p = cfg.paths.dhdt_dir / "serie_anual.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    nq = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
                         columns=["x", "y", "lat", "lon"])
    d = d.merge(nq, on=["x", "y"], how="inner")
    d = d[(d.lon > lon0) & (d.lon < lon1)]
    d = d[d.ano == d.ano.max()]
    if d.empty:
        return None
    d = d.assign(_b=(np.round(d.lat.to_numpy() / DLAT) * DLAT).round(4))
    q = d.groupby("_b").agg(anom=("anom", "median"), n=("anom", "size")).reset_index()
    return q[q.n >= 3]


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Rebaixamento por latitude direto dos segmentos ATL06.")
    ap.add_argument("--recalcular", action="store_true",
                    help="ignora o cache de células e varre os tiles de novo")
    ap.add_argument("--lon-centro", type=float, default=None,
                    help="centro da faixa; padrão: escolhido por cobertura")
    ap.add_argument("--lon-largura", type=float, default=2.0)
    args = ap.parse_args()

    cfgs = {"JJA": load_config(), "DJF": load_config("djf")}
    log = setup_logging(cfgs["JJA"].paths.logs, level=cfgs["JJA"].logging.level,
                        run_name="perfil_bruto")

    cel = {}
    for lab, cfg in cfgs.items():
        log.info(f"--- {lab} ---")
        cel[lab] = agregar(cfg, log, recalcular=args.recalcular)

    # --- escolha da faixa -------------------------------------------------
    # Duas decisões separadas, e é importante não confundi-las:
    #
    # (1) ONDE cortar é escolha GEOGRÁFICA, não estatística. A janela de
    #     candidatas é o setor de saída da Thwaites, −108,5° a −104,5°: a leste
    #     dela o corte cruza Pine Island e a oeste, Smith/Kohler/Pope — outras
    #     geleiras, com outra história. A varredura completa (−110° a −103°)
    #     fica no JSON como diagnóstico, e mostra rebaixamento a jusante de
    #     −6,8 a +1,5 m fora dessa janela: isso é geografia, não instabilidade
    #     de método, e seria desonesto apresentá-lo como barra de erro.
    #
    # (2) QUAL das faixas dessa janela, isso sim por cobertura: fica a que tem
    #     mais células observadas em todos os anos NA ESTAÇÃO MAIS POBRE — a
    #     que sustenta a curva com mais lugares, não a que dá o número maior.
    #     A sensibilidade que mede o método é a da LARGURA, com o centro fixo
    #     (mesmo lugar, abertura diferente), também no JSON.
    CENTRO_MIN, CENTRO_MAX = -108.5, -104.5
    centros = np.arange(-110.0, -102.9, 0.5)
    candidatos = [c for c in centros if CENTRO_MIN <= c <= CENTRO_MAX]
    if args.lon_centro is not None:
        centro, largura = args.lon_centro, args.lon_largura
    else:
        pontuacao = []
        for c in candidatos:
            n = min(painel_balanceado(cel[l], c - args.lon_largura / 2,
                                      c + args.lon_largura / 2)[1]["n_celulas"]
                    for l in ("JJA", "DJF"))
            pontuacao.append((n, c))
        n_best, centro = max(pontuacao)
        largura = args.lon_largura
        log.info(f"faixa escolhida em [{CENTRO_MIN}, {CENTRO_MAX}]: centro "
                 f"{centro:.2f}°, largura {largura:.1f}° ({n_best} células "
                 f"balanceadas na estação mais pobre)")
    lon0, lon1 = centro - largura / 2, centro + largura / 2

    perfis, diags, nos, extra = {}, {}, {}, {}
    print("figuras:")
    for lab, cfg in cfgs.items():
        f, d = painel_balanceado(cel[lab], lon0, lon1)
        if d["n_celulas"] < 40:
            raise RuntimeError(f"{lab}: só {d['n_celulas']} células balanceadas "
                               f"na faixa {lon0:.1f}..{lon1:.1f}")
        perfis[lab], diags[lab] = perfil(f), d
        nos[lab] = perfil_de_nos(cfg, lon0, lon1)

        # verificação de viés entre feixes: mesma conta só com os fortes
        ff, df = painel_balanceado(cel[lab], lon0, lon1, col="h_res_forte")
        qf = perfil(ff) if df["n_celulas"] >= 40 else None
        u = perfis[lab][perfis[lab].ano == perfis[lab].ano.max()].sort_values("_b")
        # ordenado por latitude crescente: as ÚLTIMAS caixas são as menos
        # negativas, isto é, o extremo norte — a jusante, junto à linha de
        # aterramento. As primeiras são o interior, a montante.
        extra[lab] = {
            **d,
            "rebaixamento_jusante_m": float(u.anom.tail(3).median()),
            "rebaixamento_montante_m": float(u.anom.head(3).median()),
            "so_feixes_fortes": None if qf is None else {
                "n_celulas": df["n_celulas"],
                "rebaixamento_jusante_m": float(
                    qf[qf.ano == qf.ano.max()].sort_values("_b").anom.tail(3).median())},
        }

    fig11(perfis, diags, (lon0, lon1))
    fig12(perfis, nos, (lon0, lon1))

    rel = {
        "metodo": ("mediana de h_res=h_corr-REMA por celula fixa de "
                   f"{int(CELL_M)} m e ano; anomalia relativa ao primeiro ano da "
                   "propria celula; painel balanceado (celula observada em todos "
                   f"os anos); mediana por caixa de {DLAT} deg de latitude"),
        "faixa_lon": [lon0, lon1],
        "celula_m": CELL_M, "dlat_deg": DLAT,
        "min_pts_celula": MIN_PTS_CELULA, "min_celulas_caixa": MIN_CELULAS_CAIXA,
        "janela_de_candidatas_lon": [CENTRO_MIN, CENTRO_MAX],
        "estacoes": extra,
        "nota_sensibilidade": (
            "sensibilidade_faixa varre CENTRO e LARGURA juntos. Variar o centro "
            "muda de LUGAR (a leste de -104,5 entra Pine Island; a oeste de "
            "-108,5, Smith/Kohler/Pope), logo a dispersao entre centros e "
            "geografia, NAO incerteza do metodo. A sensibilidade do metodo e a "
            "das LARGURAS com centro fixo: mesmo lugar, abertura diferente."),
        "sensibilidade_faixa": sensibilidade_faixa(
            cel["JJA"], centros, (1.0, 2.0, 3.0)),
    }
    (OUT / "perfil_bruto.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  perfil_bruto.json")
    for lab in ("JJA", "DJF"):
        e = extra[lab]
        print(f"  {lab}: {e['n_celulas']:,} células · "
              f"jusante {e['rebaixamento_jusante_m']:+.2f} m · "
              f"montante {e['rebaixamento_montante_m']:+.2f} m")


if __name__ == "__main__":
    main()
