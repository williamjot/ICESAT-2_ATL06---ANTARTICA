"""
pipelines/run_sealevel.py
=========================
Tendência da anomalia de altura da superfície do mar (SSHA) do ICESat-2 na
porção MARINHA da ROI, 2019-2025, a partir do ATL21 mensal gridado.

    data/atl21_ase_ssha.parquet
        -> outputs/<estacao>/tables/sealevel_ssha_trend.parquet
           outputs/<estacao>/tables/sealevel_report.json
           outputs/<estacao>/figures/sealevel_cobertura.png
           outputs/<estacao>/figures/sealevel_tendencia.png
           outputs/<estacao>/figures/sealevel_serie.png

ENQUADRAMENTO (decidido explicitamente, não implícito)
------------------------------------------------------
O que este produto mede: VARIABILIDADE DINÂMICA REGIONAL da superfície do
oceano coberto por gelo marinho no Amundsen Sea Embayment.

O que ele NÃO mede: "aumento do nível do mar". Quatro razões, medidas ou
documentadas, que devem constar da legenda de qualquer figura publicada:

1. SEM CALIBRAÇÃO ABSOLUTA DE DERIVA. A tendência global do nível do mar é
   medida por altimetria radar continuamente amarrada a marégrafos
   (Jason/Sentinel-6) e chega a ~0,3 mm/ano depois de DÉCADAS. O ATLAS não tem
   esse controle: um viés dependente de estado de mar, tipo de superfície ou
   first-photon bias que derive poucos mm/ano é indistinguível do sinal.

2. VIÉS AMOSTRAL NÃO ESTACIONÁRIO — o mais grave. Só há medida onde há *lead*,
   e a ocorrência de leads depende do estado do gelo marinho, que tem tendência
   própria e forte no Amundsen no período. A população amostrada muda ao longo
   da série; isso gera tendência aparente sem nenhuma mudança de nível do mar.
   O script mede esse risco (correlação entre n_refsurfs e o tempo) mas NÃO o
   corrige — não há como, com este dado.

3. JANELA CURTA EM REGIÃO DE ALTA VARIABILIDADE. Sete anos contra variabilidade
   interanual de dezenas de cm (Amundsen Sea Low, ENSO, SAM). O IC da tendência
   é dominado por essa variabilidade, não pelo erro de medida.

4. A FÍSICA CONTRARIA A PALAVRA "AUMENTO". Perto de uma geleira que perde
   massa, o nível do mar CAI (fingerprint gravitacional + rebound elástico):
   ordem de -1 a -3 mm/ano junto a Thwaites, contra ~+3-4 mm/ano globais. O
   sinal esperado localmente é próximo de zero — abaixo do limite de detecção.

Por isso o relatório reporta um LIMITE DE DETECÇÃO junto com a tendência: a
largura do IC de Sen é o resultado honesto, mesmo (principalmente) quando ela
engloba zero.

Uso:
    python pipelines/run_sealevel.py --name ssha_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.timeseries.trend import seasonal_mann_kendall_sen


def carregar(cfg, log) -> pd.DataFrame:
    path = cfg.paths.data_dir / cfg.sealevel.path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe — rode antes: python pipelines/fetch_atl21.py")
    df = pd.read_parquet(path)
    log.info(f"{len(df):,} linhas célula-mês, "
             f"{df.groupby(['cell_j','cell_i']).ngroups:,} células, "
             f"{df.groupby(['year','month']).ngroups} meses")
    return df


def filtrar(df: pd.DataFrame, cfg, log) -> tuple[pd.DataFrame, dict]:
    """Aplica os limiares de amostragem e registra o custo de cada um."""
    sl = cfg.sealevel
    n0 = len(df)
    df = df[df["n_refsurfs"] >= sl.min_refsurfs].copy()
    n1 = len(df)

    cont = df.groupby(["cell_j", "cell_i"]).agg(
        n_meses=("ssha", "size"), n_anos=("year", "nunique"))
    ok = cont[(cont["n_meses"] >= sl.min_months) & (cont["n_anos"] >= sl.min_years)]
    df = df.merge(ok.index.to_frame(index=False), on=["cell_j", "cell_i"])

    custo = {
        "linhas_iniciais": n0,
        "apos_min_refsurfs": n1,
        "custo_min_refsurfs_pct": round(100 * (n0 - n1) / max(n0, 1), 2),
        "celulas_iniciais": int(len(cont)),
        "celulas_com_serie_suficiente": int(len(ok)),
        "linhas_finais": len(df),
    }
    log.info(f"min_refsurfs>={sl.min_refsurfs}: custa "
             f"{custo['custo_min_refsurfs_pct']}% das linhas")
    log.info(f"células com >={sl.min_months} meses e >={sl.min_years} anos: "
             f"{len(ok):,} de {len(cont):,} "
             f"({100*len(ok)/max(len(cont),1):.1f}%)")
    return df, custo


def tendencia_por_celula(df: pd.DataFrame, cfg, log) -> pd.DataFrame:
    """
    Mann-Kendall SAZONAL + Sen por célula, com FDR entre células.

    Sazonal (não o MK simples) porque a SSHA tem ciclo anual forte e a
    amostragem por leads é ela própria sazonal: comparar fevereiro com agosto
    como se fossem amostras da mesma distribuição criaria tendência a partir do
    ciclo. O MK sazonal só compara o mesmo mês entre anos distintos.
    """
    from statsmodels.stats.multitest import multipletests

    alpha = cfg.trend.alpha
    linhas = []
    for (cj, ci), g in df.groupby(["cell_j", "cell_i"], sort=False):
        r = seasonal_mann_kendall_sen(g["year"].to_numpy(), g["month"].to_numpy(),
                                      g["ssha"].to_numpy(), alpha)
        if r["trend"] == "insuficiente":
            continue
        linhas.append({
            "cell_j": cj, "cell_i": ci,
            "x": float(g["x"].iloc[0]), "y": float(g["y"].iloc[0]),
            "lon": float(g["lon"].iloc[0]), "lat": float(g["lat"].iloc[0]),
            "n_meses": int(len(g)), "n_anos": int(g["year"].nunique()),
            "n_refsurfs_mediano": float(g["n_refsurfs"].median()),
            # mm/ano para leitura direta contra as escalas de nível do mar
            "sen_mm_ano": 1e3 * r["sens_slope"],
            "sen_lo_mm_ano": 1e3 * r["sens_lo"],
            "sen_hi_mm_ano": 1e3 * r["sens_hi"],
            "tau": r["tau"], "p_value": r["p_value"],
        })

    if not linhas:
        log.warning("Nenhuma célula com série suficiente para tendência.")
        return pd.DataFrame()

    out = pd.DataFrame(linhas)
    rej, p_fdr, _, _ = multipletests(out["p_value"].to_numpy(), alpha=alpha,
                                     method=cfg.trend.fdr_method.replace(
                                         "benjamini-hochberg", "fdr_bh"))
    out["p_fdr"] = p_fdr
    out["significativo"] = rej
    out["largura_ic_mm_ano"] = out["sen_hi_mm_ano"] - out["sen_lo_mm_ano"]
    log.info(f"{len(out):,} células testadas, {int(rej.sum()):,} significativas "
             f"(FDR α={alpha})")
    return out


def risco_vies_amostral(df: pd.DataFrame) -> dict:
    """
    Quantifica (não corrige) o viés amostral não estacionário.

    Se o nº de superfícies de referência por célula-mês tem tendência ao longo
    da série, a população medida está mudando. Isso não é ruído: é um caminho
    direto para tendência espúria de SSHA, e precisa ser reportado junto do
    resultado.
    """
    t = df["year"] + (df["month"] - 0.5) / 12.0
    por_mes = df.groupby(["year", "month"]).agg(
        n_celulas=("ssha", "size"), refsurfs=("n_refsurfs", "median"),
        ssha=("ssha", "median")).reset_index()
    tm = por_mes["year"] + (por_mes["month"] - 0.5) / 12.0

    def _sen(v):
        r = seasonal_mann_kendall_sen(por_mes["year"], por_mes["month"], v)
        return {"sen_por_ano": r["sens_slope"], "p": r["p_value"], "n": r["n"]}

    return {
        "explicacao": ("a medida só existe onde há lead; se a cobertura tem "
                       "tendência própria, a população amostrada muda ao longo "
                       "da série e isso é indistinguível de tendência de SSHA"),
        "corr_refsurfs_vs_tempo": float(np.corrcoef(t, df["n_refsurfs"])[0, 1]),
        "tendencia_n_celulas_por_mes": _sen(por_mes["n_celulas"].to_numpy()),
        "tendencia_refsurfs_medianos": _sen(por_mes["refsurfs"].to_numpy()),
        "cobertura_por_ano": por_mes.groupby("year")["n_celulas"].mean().round(1).to_dict(),
    }


def figuras(trend: pd.DataFrame, df: pd.DataFrame, cfg, args, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    figdir = cfg.paths.figures
    figdir.mkdir(parents=True, exist_ok=True)
    res = cfg.sealevel.grid_res_m
    rotulo = (cfg.roi.label if cfg.roi and cfg.roi.label else "ROI")
    rodape = ("ICESat-2 ATL21 v004 (SSHA mensal, 25 km) | 2019-2025 | "
              "variabilidade dinâmica regional — NÃO é medida de nível do mar "
              "global (sem calibração de deriva; amostragem condicionada a leads)")

    # ---- 1. cobertura ------------------------------------------------------
    cob = df.groupby(["cell_j", "cell_i"]).agg(
        x=("x", "first"), y=("y", "first"), n=("ssha", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sc = ax.scatter(cob["x"] / 1e3, cob["y"] / 1e3, c=cob["n"], s=res / 1e3 * 3.2,
                    marker="s", cmap="viridis", vmin=0, vmax=84)
    plt.colorbar(sc, ax=ax, label="meses com SSHA válida (de 84 possíveis)")
    ax.set_title(f"Cobertura da SSHA do ICESat-2 — {rotulo}\n"
                 f"{len(cob):,} células de 25 km com ao menos um mês válido")
    ax.set_xlabel("x EPSG:3031 (km)"); ax.set_ylabel("y EPSG:3031 (km)")
    ax.set_aspect("equal")
    fig.text(0.5, 0.01, rodape, ha="center", fontsize=6.5, wrap=True)
    fig.savefig(figdir / f"sealevel_cobertura_{args.name}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # ---- 2. tendência ------------------------------------------------------
    if not trend.empty:
        v = trend["sen_mm_ano"].to_numpy()
        lim = float(np.nanpercentile(np.abs(v), 98)) or 1.0
        fig, ax = plt.subplots(figsize=(9, 7.5))
        sc = ax.scatter(trend["x"] / 1e3, trend["y"] / 1e3, c=v,
                        s=res / 1e3 * 3.2, marker="s", cmap="RdBu_r",
                        norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
        plt.colorbar(sc, ax=ax, label="tendência de SSHA (mm/ano)")
        # Célula não significativa é marcada, não escondida: a ausência de
        # tendência detectável é resultado, e omiti-la sugeriria um campo
        # coerente onde há sobretudo ruído.
        ns = trend[~trend["significativo"]]
        ax.scatter(ns["x"] / 1e3, ns["y"] / 1e3, s=4, marker="x", c="k",
                   linewidths=0.5,
                   label=f"não significativa (FDR): {len(ns)}/{len(trend)}")
        ax.legend(loc="lower left", fontsize=7)
        ax.set_title(f"Tendência da SSHA 2019-2025 — {rotulo}\n"
                     f"Mann-Kendall sazonal + Sen, FDR α={cfg.trend.alpha}")
        ax.set_xlabel("x EPSG:3031 (km)"); ax.set_ylabel("y EPSG:3031 (km)")
        ax.set_aspect("equal")
        fig.text(0.5, 0.01, rodape, ha="center", fontsize=6.5, wrap=True)
        fig.savefig(figdir / f"sealevel_tendencia_{args.name}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- 3. série e cobertura no tempo -------------------------------------
    por_mes = df.groupby(["year", "month"]).agg(
        ssha=("ssha", "median"), n=("ssha", "size")).reset_index()
    t = por_mes["year"] + (por_mes["month"] - 0.5) / 12.0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                 height_ratios=[2, 1])
    a1.plot(t, 1e2 * por_mes["ssha"], "o-", ms=3, lw=0.9, color="#1f6f8b")
    a1.axhline(0, color="k", lw=0.6, ls=":")
    a1.set_ylabel("SSHA mediana da ROI (cm)")
    a1.set_title(f"SSHA mensal do ICESat-2 — {rotulo}")
    a2.bar(t, por_mes["n"], width=0.06, color="#999")
    a2.set_ylabel("células/mês")
    a2.set_xlabel("ano")
    # A cobertura é plotada JUNTO da série de propósito: se as duas variam em
    # fase, a "tendência" da SSHA pode ser a tendência da amostragem.
    fig.text(0.5, 0.005, rodape, ha="center", fontsize=6.5, wrap=True)
    fig.savefig(figdir / f"sealevel_serie_{args.name}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figuras -> {figdir}")


def main():
    ap = argparse.ArgumentParser(
        description="Tendência de SSHA (ATL21) na porção marinha da ROI.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"sealevel_{args.name}")

    df = carregar(cfg, log)
    df, custo = filtrar(df, cfg, log)
    if df.empty:
        raise SystemExit("Nenhuma célula sobrevive aos limiares de amostragem — "
                         "esse já é o resultado: a cobertura não sustenta mapa "
                         "de tendência nesta ROI.")

    trend = tendencia_por_celula(df, cfg, log)
    vies = risco_vies_amostral(df)

    tabdir = cfg.paths.tables
    tabdir.mkdir(parents=True, exist_ok=True)
    if not trend.empty:
        trend.to_parquet(tabdir / f"sealevel_ssha_trend_{args.name}.parquet",
                         index=False)

    med = float(np.nanmedian(trend["sen_mm_ano"])) if not trend.empty else float("nan")
    ic = float(np.nanmedian(trend["largura_ic_mm_ano"])) if not trend.empty else float("nan")
    rel = {
        "enquadramento": ("variabilidade dinâmica regional da superfície do "
                          "oceano coberto por gelo marinho; NÃO é medida de "
                          "aumento do nível do mar"),
        "fonte": f"{cfg.sealevel.short_name} v{cfg.sealevel.version}",
        "roi": cfg.roi.bounding_box if cfg.roi else cfg.area.bounding_box,
        "periodo": cfg.temporal.temporal_range,
        "filtros": custo,
        "celulas_testadas": int(len(trend)),
        "celulas_significativas": int(trend["significativo"].sum()) if not trend.empty else 0,
        "sen_mediano_mm_ano": med,
        "largura_ic_mediana_mm_ano": ic,
        "limite_de_deteccao_mm_ano": ic / 2.0,
        "risco_vies_amostral": vies,
        "ressalvas_obrigatorias": [
            "sem calibração absoluta de deriva do altímetro contra marégrafos",
            "amostragem condicionada à presença de leads (viés não estacionário)",
            "7 anos contra variabilidade interanual de dezenas de cm no ASE",
            "fingerprint gravitacional local é NEGATIVO perto de Thwaites",
        ],
    }
    (tabdir / f"sealevel_report_{args.name}.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    figuras(trend, df, cfg, args, log)

    log.info("=" * 70)
    log.info(f"Sen mediano: {med:+.2f} mm/ano | largura mediana do IC: "
             f"{ic:.2f} mm/ano => limite de detecção ~{ic/2:.1f} mm/ano")
    log.info(f"Relatório -> {tabdir / f'sealevel_report_{args.name}.json'}")


if __name__ == "__main__":
    main()
