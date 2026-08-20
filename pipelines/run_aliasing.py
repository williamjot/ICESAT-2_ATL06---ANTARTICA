"""
pipelines/run_aliasing.py
=========================
Teste de resposta harmônica: quanto do ciclo sazonal vaza para o dh/dt, dada a
amostragem só-JJA?

    data/tiles/*.parquet  ->  outputs/experiments/<nome>/
                                aliasing_nodes.parquet
                                aliasing_report.json

O que este teste responde
-------------------------
O teste quantifica se dh/dt restrito ao inverno constitui uma estimativa
conservadora da tendência anual, em vez de assumir essa interpretação.

Método: ATBD ATL14/ATL15 r005 §5 — injeta harmônico sintético de amplitude
unitária nas ÉPOCAS REAIS, roda o estimador de produção, e mede o dh/dt
recuperado. Como o sinal injetado não tem tendência, tudo o que aparece é
vazamento (aliasing).

A escala em metros vem da amplitude sazonal medida no GSFC-FDM (`h_a`), não de
valor de literatura.

Contrafactual: o mesmo cálculo com as observações redistribuídas sobre o ano
inteiro (ver `stretch_to_full_year`) isola quanto do vazamento vem da COBERTURA
DE FASE restrita, e não do número de observações.

Uso: python pipelines/run_aliasing.py --name aliasing_v1
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
from thwaites.grid.tiles import assign_xy, load_manifest
from thwaites.experiments.manifest import Manifest
from thwaites.timeseries.aliasing import (node_response, stretch_to_full_year,
                                          seasonal_amplitude,
                                          jja_vs_annual_trend)
from thwaites.io.memory import free_memory_gb

PERIODS = {"anual": 1.0, "semianual": 0.5}


def tile_responses(df, cfg, x_min, x_max, y_min, y_max):
    """Vazamento por nó no núcleo do tile, para cada período e amostragem."""
    from scipy.spatial import cKDTree

    d = cfg.dhdt
    df = assign_xy(df, cfg)
    ok = ~(df["x"].isna() | df["y"].isna() | df["t_year"].isna())
    x = df["x"].to_numpy()[ok]
    y = df["y"].to_numpy()[ok]
    t = df["t_year"].to_numpy()[ok].astype(float)
    s = (df["s_elv"].to_numpy()[ok].astype(float)
         if "s_elv" in df.columns else np.full(int(ok.sum()), 0.05))
    if len(t) < d.min_points:
        return pd.DataFrame()

    t_full = stretch_to_full_year(t)

    step = d.node_spacing_m
    gx = np.arange(x_min + step / 2, x_max, step)
    gy = np.arange(y_min + step / 2, y_max, step)
    if len(gx) == 0 or len(gy) == 0:
        return pd.DataFrame()
    GX, GY = np.meshgrid(gx, gy)

    tree = cKDTree(np.c_[x, y])
    rows = []
    for nx, ny in zip(GX.ravel(), GY.ravel()):
        idx = tree.query_ball_point([nx, ny], r=d.search_radius_m)
        if len(idx) < d.min_points:
            continue
        idx = np.asarray(idx)
        xi, yi, ti, si = x[idx], y[idx], t[idx], s[idx]
        rec = {"x": float(nx), "y": float(ny)}
        keep = True
        for pname, P in PERIODS.items():
            r_jja = node_response(xi, yi, ti, si, nx, ny, d, P)
            r_full = node_response(xi, yi, t_full[idx], si, nx, ny, d, P)
            if r_jja is None or r_full is None:
                keep = False
                break
            rec[f"R_{pname}_jja"] = r_jja["R"]
            rec[f"R_{pname}_fullyear"] = r_full["R"]
            rec["nobs"] = r_jja["nobs"]
        if keep:
            rows.append(rec)

    return pd.DataFrame(rows)


def fdm_tests(cfg, log, frac_lo, frac_hi):
    """
    Dois testes sobre o GSFC-FDM `h_a` (altura de superfície modelada):
      1. amplitude sazonal -> escala em metros para o vazamento harmônico;
      2. tendência JJA vs. ano inteiro -> teste DIRETO de "JJA é conservador".
    """
    from netCDF4 import Dataset

    p = cfg.paths.data_dir / cfg.firn.path
    if not p.exists():
        log.warning(f"{p} ausente — sem escala em metros nem teste pareado.")
        return None, None
    with Dataset(p) as f:
        if "h_a" not in f.variables:
            log.warning("GSFC-FDM sem 'h_a'.")
            return None, None
        t = np.asarray(f["time"][:], float)
        units = str(getattr(f["time"], "units", "")).lower()
        # ATENÇÃO: `np.asarray()` sobre o MaskedArray do netCDF4 DESCARTA a
        # máscara e devolve o _FillValue cru (-9999 neste arquivo, ~26% das
        # células). `np.ma.filled(..., nan)` preserva a mascaração.
        h_a = np.ma.filled(f["h_a"][:].astype(float), np.nan)
    if "year" not in units:
        raise ValueError(f"unidade de tempo do FDM não é ano: {units!r}")

    amp = seasonal_amplitude(h_a, t, period=1.0)
    v = amp[np.isfinite(amp)]
    amp_rep = None
    if v.size:
        log.info(f"amplitude sazonal do h_a (GSFC-FDM, {t.min():.2f}-{t.max():.2f}): "
                 f"mediana {np.median(v):.3f} m | p90 {np.percentile(v, 90):.3f} m "
                 f"({v.size} células)")
        amp_rep = {"mediana_m": float(np.median(v)),
                   "p10_m": float(np.percentile(v, 10)),
                   "p90_m": float(np.percentile(v, 90)),
                   "n_celulas": int(v.size),
                   "cobertura": f"{t.min():.3f}-{t.max():.3f}"}

    try:
        paired = jja_vs_annual_trend(h_a, t, frac_lo, frac_hi)
        log.info(f"JJA vs ano inteiro no h_a ({paired['anos'][0]:.0f}-"
                 f"{paired['anos'][1]:.0f}, {paired['n_epocas_jja']} vs "
                 f"{paired['n_epocas_anual']} épocas):")
        log.info(f"  slope anual  {paired['slope_anual_mediana']:+.4f} m/ano")
        log.info(f"  slope JJA    {paired['slope_jja_mediana']:+.4f} m/ano")
        log.info(f"  diferença    {paired['diff_mediana']:+.4f} m/ano "
                 f"(p10 {paired['diff_p10']:+.4f}, p90 {paired['diff_p90']:+.4f}) "
                 f"| {100*paired['frac_celulas_jja_conservador']:.1f}% das "
                 f"células com JJA conservador")
    except ValueError as e:
        log.warning(f"teste pareado JJA/anual não realizado: {e}")
        paired = None

    return amp_rep, paired


def main():
    ap = argparse.ArgumentParser(description="Vazamento sazonal para o dh/dt.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"aliasing_{args.name}")

    man = Manifest(cfg, args.name,
                   purpose="Vazamento sazonal para o dh/dt (ATBD ATL14/15 §5)",
                   overwrite=args.overwrite, seed=0)
    man.set("metodo", {
        "referencia": "ATBD ATL14/ATL15 r005, §5 (teste de resposta harmônica)",
        "sinal": "harmônico de amplitude unitária nas épocas REAIS",
        "estimador": ("o MESMO da produção (_build_A/_lstsq_iter de "
                      "thwaites.timeseries.dhdt), inclusive rejeição por MAD"),
        "R": ("sqrt(b_cos²+b_sin²) = vazamento de PIOR CASO sobre a fase, em "
              "(m/ano) por metro de amplitude do sinal; RMS sobre fase "
              "uniforme = R/sqrt(2)"),
        "contrafactual": ("mesmas observações remapeadas para o ano inteiro — "
                          "isola cobertura de fase, NÃO simula amostragem real "
                          "de ano completo"),
        "periodos": PERIODS,
    })

    entries = load_manifest(cfg)
    if args.max_tiles:
        entries = entries[:args.max_tiles]

    # Fora do diretório do experimento: `experiment_dir(overwrite=True)` faz
    # rmtree na árvore, e o checkpoint é cálculo caro, não saída. Ver a mesma
    # nota, mais longa, em run_acceleration.py.
    ckpt = cfg.paths.interim / f"aliasing_ckpt_{args.name}"
    ckpt.mkdir(parents=True, exist_ok=True)
    if args.restart:
        for p in ckpt.glob("*.parquet"):
            p.unlink()
    done = {p.stem for p in ckpt.glob("*.parquet")}
    todo = [e for e in entries if e["tile"] not in done]
    log.info(f"a processar: {len(todo)} de {len(entries)} tiles "
             f"({len(done)} em checkpoint)")

    import pyarrow.parquet as pq
    NEEDED = ["x", "y", "lon", "lat", "t_year", "s_elv"]

    # Janela de fase REALMENTE observada — o teste pareado tem de usar a janela
    # dos dados, não a definição de calendário de JJA. Medida numa AMOSTRA de
    # tiles (a janela é propriedade da missão, não do tile) e fora do laço
    # principal, para não depender de quantos tiles o checkpoint pulou.
    sample = entries[::max(len(entries) // 15, 1)]
    frac_lo, frac_hi = 1.0, 0.0
    for e in sample:
        tv = pd.read_parquet(cfg.paths.tiles_dir / e["file"],
                             columns=["t_year"], engine="pyarrow")["t_year"]
        tv = tv.to_numpy(float)
        tv = tv[np.isfinite(tv)]
        if tv.size:
            fr = tv - np.floor(tv)
            frac_lo = min(frac_lo, float(fr.min()))
            frac_hi = max(frac_hi, float(fr.max()))
    log.info(f"janela de fase observada: {frac_lo:.4f}-{frac_hi:.4f} do ano "
             f"(~dia {frac_lo*365.25:.0f} a {frac_hi*365.25:.0f}), "
             f"medida em {len(sample)} tiles")

    for i, e in enumerate(todo, 1):
        p = cfg.paths.tiles_dir / e["file"]
        avail = pq.ParquetFile(p).schema_arrow.names
        cols = [c for c in NEEDED if c in avail]
        tdf = pd.read_parquet(p, columns=cols, engine="pyarrow")
        nd = tile_responses(tdf, cfg, e["x_min"], e["x_max"],
                            e["y_min"], e["y_max"])
        del tdf
        if len(nd):
            nd["tile"] = e["tile"]
        nd.to_parquet(ckpt / f"{e['tile']}.parquet", index=False)
        log.info(f"  [{i}/{len(todo)}] {e['tile']}: {len(nd):,} nós "
                 f"| livre {free_memory_gb():.1f} GB")

    parts = [g for f in sorted(ckpt.glob("*.parquet"))
             if len(g := pd.read_parquet(f))]
    if not parts:
        log.warning("Nenhum nó avaliado.")
        return
    nodes = pd.concat(parts, ignore_index=True)
    out_p = man.path_for("aliasing_nodes.parquet")
    nodes.to_parquet(out_p, index=False)
    man.add_output(out_p)

    amp, paired = fdm_tests(cfg, log, frac_lo, frac_hi)

    def stats(col):
        v = nodes[col].to_numpy()
        v = v[np.isfinite(v)]
        return {"mediana": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                "p99": float(np.percentile(v, 99)), "max": float(v.max())}

    rep = {
        "n_nodes": int(len(nodes)),
        "janela_fase_observada": [frac_lo, frac_hi],
        "teste_1_aliasing_harmonico": {
            "pergunta": ("quanto de um ciclo sazonal ESTACIONÁRIO vaza para a "
                         "tendência, dada a amostragem real?"),
            "unidade_R": "(m/ano) por metro de amplitude do sinal periódico",
            "vazamento": {c: stats(c) for c in nodes.columns
                          if c.startswith("R_")},
            "amplitude_sazonal_fdm": amp,
        },
        "teste_2_jja_vs_anual": {
            "pergunta": ("a tendência estimada só em JJA difere da do ano "
                         "inteiro? — hipótese testada no h_a "
                         "do GSFC-FDM, que tem cobertura anual"),
            "sinal": ("diff = slope_jja − slope_anual; POSITIVO = JJA menos "
                      "negativo = subestima a perda = 'conservador'"),
            "escopo": ("vale para a componente firn+SMB da altura; a "
                       "componente dinâmica não é modelada pelo FDM"),
            "resultado": paired,
        },
    }

    if amp is not None:
        A = amp["mediana_m"]
        rep["vies_estimado_m_por_ano"] = {}
        for pname in PERIODS:
            Rj = float(np.median(nodes[f"R_{pname}_jja"]))
            Rf = float(np.median(nodes[f"R_{pname}_fullyear"]))
            rep["vies_estimado_m_por_ano"][pname] = {
                "jja_pior_caso": A * Rj,
                "jja_rms_fase": A * Rj / np.sqrt(2),
                "contrafactual_espalhado_pior_caso": A * Rf,
            }
        rep["nota_escala"] = (
            f"viés = R × amplitude sazonal mediana ({A:.3f} m, GSFC-FDM h_a). "
            "A amplitude do FDM cobre 2019-06/2022; os anos 2022-2025 do nosso "
            "período NÃO estão nela.")

    rep["limitacoes"] = [
        "NÃO ler 'contrafactual_espalhado' como 'amostragem de ano completo é "
        "pior'. O remapeamento preserva a estrutura de RAJADA das passagens e "
        "só a estica sobre o ano: cada ano passa a ter ~3 rajadas isoladas em "
        "fases diferentes, que não se cancelam. Amostragem anual densa e "
        "uniforme cancelaria o harmônico e vazaria MENOS. O número serve para "
        "mostrar que a janela estreita de JJA congela a fase — não para "
        "comparar qualidade de amostragem.",
        "a amplitude sazonal vem do GSFC-FDM (firn+SMB) e não inclui variação "
        "sazonal de origem dinâmica",
        "no teste 1 a amplitude é constante no tempo; amplitude que varia entre "
        "anos produz vazamento adicional — é o teste 2 que captura isso",
        "o teste 2 cobre 3 anos completos do FDM (2019-2021); o nosso período "
        "vai a 2025",
    ]

    rp = man.path_for("aliasing_report.json")
    rp.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    man.add_output(rp)
    man.write()

    log.info(f"{len(nodes):,} nós avaliados")
    for c in sorted(k for k in nodes.columns if k.startswith("R_")):
        st = stats(c)
        log.info(f"  {c}: mediana {st['mediana']:.4f} | p90 {st['p90']:.4f} "
                 f"(m/ano por m de amplitude)")
    if amp is not None:
        for pname, v in rep["vies_estimado_m_por_ano"].items():
            log.info(f"  viés {pname}: JJA {v['jja_pior_caso']:+.4f} m/ano "
                     f"(pior caso) | RMS sobre fase {v['jja_rms_fase']:+.4f}")
    log.info(f"Experimento em {man.dir}")


if __name__ == "__main__":
    main()
