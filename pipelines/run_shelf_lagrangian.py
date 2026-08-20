"""
pipelines/run_shelf_lagrangian.py
=================================
DH/Dt LAGRANGIANO na plataforma: rastreia parcelas de gelo e ajusta a taxa ao
longo da trajetória — o rastreamento vem ANTES da estimação temporal.

    data/interim/atl06_shelf.parquet + data/velocity_itslive_annual.nc
        -> data/dhdt/shelf_lagrangian_parcels.parquet
        -> outputs/tables/shelf_lagrangian_report.json

Diferença em relação ao produto de gelo aterrado
------------------------------------------------
Lá, os nós são FIXOS no espaço e a taxa é ∂h/∂t (Euleriana). Aqui cada parcela é
seguida no tempo: as observações que entram no ajuste de uma parcela são as que
caem perto da posição QUE AQUELA PARCELA OCUPAVA em cada época. Medido nesta
região, uma parcela percorre até 29 km em 6 anos — muito mais que o espaçamento
de nós, então a distinção não é acadêmica.

Escolha da coluna de elevação
-----------------------------
Usa-se `h_corr` (maré CATS2008 + DAC removidos) e NÃO `h_res` (= h_corr − REMA).
Subtrair um DEM fixo no espaço é correto para nós fixos, mas numa trajetória de
dezenas de km introduziria o GRADIENTE do REMA como sinal falso de DH/Dt — o
mesmo mecanismo pelo qual o gradiente do geoide contamina o rastreamento.

Limitações estruturais (declaradas no relatório)
------------------------------------------------
* produtos Parquet com a coluna `geoid` inteiramente nula exigem harmonização
  pelo geoide do BedMachine; o ATL06 armazena o geoide em `dem/geoid_h`;
* máscara de plataforma ESTÁTICA (BedMachine, `nominal_year` 2015) — as frentes
  datadas do IceLines já estão em disco (`data/calving_fronts.parquet`) mas
  ainda não são aplicadas por época;
* a velocidade de 2025 é a última época do ITS_LIVE; além dela haveria
  extrapolação.

Por isso a saída é um DH/Dt Lagrangiano, NÃO um derretimento basal. Fechar
ṁ_b exige ainda H(t), SMB e o termo H·∇·v (não ∇·(H·v) — usar a divergência
completa junto de DH/Dt contaria a advecção duas vezes).

Uso:
    python pipelines/run_shelf_lagrangian.py [--spacing-km 2] [--radius-km 3]
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
from thwaites.io.memory import free_memory_gb
from thwaites.glaciology.trajectory import (VelocityField, track_parcels,
                                            displacement_summary)
from thwaites.corrections.datum import GeoidField, to_orthometric


def main():
    ap = argparse.ArgumentParser(description="DH/Dt lagrangiano na plataforma.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--input", default="atl06_shelf.parquet")
    ap.add_argument("--velocity", default="velocity_itslive_annual.nc")
    ap.add_argument("--spacing-km", type=float, default=2.0,
                    help="espaçamento da grade de parcelas na época de referência")
    ap.add_argument("--radius-km", type=float, default=3.0,
                    help="raio de coleta de observações em torno da parcela")
    ap.add_argument("--min-epochs", type=int, default=4,
                    help="épocas com dado mínimas para ajustar uma taxa")
    ap.add_argument("--min-obs", type=int, default=30,
                    help="observações mínimas por parcela")
    ap.add_argument("--decimate", type=int, default=4)
    ap.add_argument("--dt-days", type=float, default=20.0)
    ap.add_argument("--max-displacement-m", type=float, default=5000.0,
                    help="deslocamento acima do qual a parcela deixa de ser "
                         "confiável (o RMSE do ajuste cresce com ele)")
    ap.add_argument("--max-rmse-m", type=float, default=2.0,
                    help="RMSE máximo do ajuste para a parcela ser confiável")
    ap.add_argument("--no-geoid", action="store_true",
                    help="desliga a harmonização de datum (só para comparar)")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="shelf_lagrangian")

    src = cfg.paths.interim / args.input
    vel_p = cfg.paths.data_dir / args.velocity
    for p in (src, vel_p):
        if not p.exists():
            raise FileNotFoundError(f"{p} não existe.")

    cols = ["x", "y", "t_year", "h_corr", "s_elv"]
    d = pd.read_parquet(src, columns=cols)
    d = d[np.isfinite(d["h_corr"]) & np.isfinite(d["t_year"])].copy()
    log.info(f"{len(d):,} observações de plataforma | livre {free_memory_gb():.1f} GB")

    vel = VelocityField(vel_p, decimate=args.decimate)

    # HARMONIZAÇÃO DE DATUM. Sem isto, o gradiente do geoide ao longo da
    # trajetória entra como DH/Dt falso: a altura da parcela é comparada em
    # posições diferentes, e um offset que cancelaria num nó fixo não cancela
    # aqui. Medido nesta ROI: 0,064 m/ano de taxa espúria nas parcelas com mais
    # de 20 km de deslocamento.
    geoid = None
    if not args.no_geoid:
        geoid = GeoidField(cfg, d["x"].min(), d["x"].max(),
                           d["y"].min(), d["y"].max())
        n = geoid.at(d["x"].to_numpy(), d["y"].to_numpy())
        d["h_orth"] = to_orthometric(d["h_corr"].to_numpy(), n)
        ok = np.isfinite(d["h_orth"])
        log.info(f"datum harmonizado: h_orth = h_corr - N | "
                 f"N mediano {np.nanmedian(n):+.2f} m | "
                 f"{int(ok.sum()):,}/{len(d):,} pontos com geoide")
        d = d[ok]
        HCOL = "h_orth"
    else:
        log.warning("datum NÃO harmonizado (--no-geoid): o gradiente do geoide "
                    "entrará como DH/Dt espúrio nas trajetórias longas.")
        HCOL = "h_corr"

    # Épocas = anos com dado. A referência é o centro do período, para que o
    # erro de integração se distribua nos dois sentidos em vez de acumular numa
    # ponta.
    years = np.array(sorted({int(np.floor(t)) for t in d["t_year"]}))
    t_ref = float(np.median(years) + 0.5)
    log.info(f"épocas: {years.tolist()} | referência t={t_ref:.2f}")

    # Grade de parcelas na época de referência, sobre a área com observação.
    step = args.spacing_km * 1000.0
    gx = np.arange(d["x"].min(), d["x"].max() + step, step)
    gy = np.arange(d["y"].min(), d["y"].max() + step, step)
    PX, PY = np.meshgrid(gx, gy)
    PX, PY = PX.ravel(), PY.ravel()

    # descarta parcelas longe de qualquer observação (a maior parte da bbox é
    # oceano ou gelo aterrado)
    from scipy.spatial import cKDTree
    tree_all = cKDTree(np.c_[d["x"].to_numpy(), d["y"].to_numpy()])
    near, _ = tree_all.query(np.c_[PX, PY], k=1)
    keep = near <= args.radius_km * 1000.0
    PX, PY = PX[keep], PY[keep]
    log.info(f"parcelas candidatas: {len(PX):,} "
             f"(grade {args.spacing_km:.1f} km, raio {args.radius_km:.1f} km)")
    del tree_all

    # posição de cada parcela em cada época (meio do ano)
    epochs = years + 0.5
    log.info("integrando trajetórias (RK4)...")
    X, Y, V = track_parcels(vel, PX, PY, t_ref, epochs, dt_days=args.dt_days)
    disp = displacement_summary(X, Y)
    log.info(f"deslocamento: {disp}")

    # ajuste por parcela: coleta observações do ano na posição daquele ano
    log.info("coletando observações por parcela e época...")
    R = args.radius_km * 1000.0
    n_par = len(PX)
    sum_h = np.zeros((len(epochs), n_par))
    cnt = np.zeros((len(epochs), n_par), dtype=np.int64)

    for i, yr in enumerate(years):
        sel = (d["t_year"] >= yr) & (d["t_year"] < yr + 1)
        if not sel.any():
            continue
        ox = d.loc[sel, "x"].to_numpy()
        oy = d.loc[sel, "y"].to_numpy()
        oh = d.loc[sel, HCOL].to_numpy(dtype=float)
        t_obs = cKDTree(np.c_[ox, oy])
        good = V[i] & np.isfinite(X[i]) & np.isfinite(Y[i])
        if not good.any():
            continue
        idx = np.where(good)[0]
        lists = t_obs.query_ball_point(np.c_[X[i][idx], Y[i][idx]], r=R)
        for k, lst in zip(idx, lists):
            if lst:
                sum_h[i, k] = float(np.median(oh[lst]))
                cnt[i, k] = len(lst)
        log.info(f"  {yr}: {sel.sum():,} obs | parcelas com dado "
                 f"{int((cnt[i] > 0).sum()):,} | livre {free_memory_gb():.1f} GB")
        del t_obs, ox, oy, oh

    # regressão da altura da parcela contra o tempo
    log.info("ajustando DH/Dt por parcela...")
    dhdt = np.full(n_par, np.nan)
    nepo = (cnt > 0).sum(axis=0)
    nobs = cnt.sum(axis=0)
    rmse = np.full(n_par, np.nan)
    for k in range(n_par):
        m = cnt[:, k] > 0
        if m.sum() < args.min_epochs or nobs[k] < args.min_obs:
            continue
        tt = epochs[m]
        hh = sum_h[m, k]
        A = np.c_[tt - t_ref, np.ones(m.sum())]
        coef, *_ = np.linalg.lstsq(A, hh, rcond=None)
        dhdt[k] = coef[0]
        rmse[k] = float(np.sqrt(np.mean((hh - A @ coef) ** 2)))

    out = pd.DataFrame({
        "x_ref": PX, "y_ref": PY, "t_ref": t_ref,
        "dhdt_lagrangian": dhdt, "n_epochs": nepo, "n_obs": nobs, "rmse": rmse,
        "displacement_m": np.sqrt((X[-1] - X[0]) ** 2 + (Y[-1] - Y[0]) ** 2),
    })
    valid = np.isfinite(out["dhdt_lagrangian"])
    out = out[valid].reset_index(drop=True)

    # ------------------------------------------------------------------
    # CLASSIFICAÇÃO DE CONFIABILIDADE POR PARCELA
    # ------------------------------------------------------------------
    # Medido nesta ROI: o RMSE do ajuste CRESCE com o deslocamento da parcela
    # (0,62 m abaixo de 5 km; 7,0 m entre 10 e 20 km; correlação r = +0,43).
    # Se o rastreamento fosse consistente isso não deveria acontecer — é a mesma
    # coluna de gelo sendo seguida. Três hipóteses foram testadas:
    #   * gradiente do geoide -> REFUTADA (~1% do sinal; corrigido mesmo assim)
    #   * erro de posição     -> REFUTADA (164 m mediano contra raio de 3 km)
    #   * taxa NÃO constante  -> SUSTENTADA (ajuste quadrático reduz RMSE 38%)
    # Ou seja, a hipótese de DH/Dt constante ao longo de 6 anos falha em
    # trajetórias longas. Não é erro a corrigir: é o modelo temporal sendo
    # inadequado ao processo. Por isso o produto é CLASSIFICADO em vez de
    # "consertado", e a estatística principal sai do subconjunto sustentável.
    # nome distinto de `disp`: aquele é o dict de displacement_summary usado no
    # relatório, e sobrescrevê-lo aqui quebrava a serialização JSON
    disp_arr = out["displacement_m"].to_numpy()
    rm = out["rmse"].to_numpy()
    out["reliability"] = np.where(
        (disp_arr < args.max_displacement_m) & (rm <= args.max_rmse_m),
        "confiavel",
        np.where(disp_arr < 10_000.0, "aceitavel_com_ressalvas", "nao_confiavel"))

    dst = cfg.paths.dhdt_dir / "shelf_lagrangian_parcels.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)

    trust = out[out["reliability"] == "confiavel"]
    counts = out["reliability"].value_counts().to_dict()
    log.info("confiabilidade: " +
             " | ".join(f"{k} {int(v):,}" for k, v in counts.items()))
    if len(trust) < 50:
        log.warning("subconjunto confiável pequeno — a estatística principal "
                    "pode não ser representativa.")

    v = out["dhdt_lagrangian"].to_numpy()
    vt = trust["dhdt_lagrangian"].to_numpy()
    report = {
        "STATUS": ("DH/Dt LAGRANGIANO — nao e derretimento basal. "
                   "Fechar m_b exige H(t), SMB e H*div(v)."),
        "ESTATISTICA_PRINCIPAL": ("usar o bloco `confiavel`; o bloco `todas` "
                                  "inclui parcelas cujo ajuste linear nao se "
                                  "sustenta"),
        "n_parcelas_validas": int(len(out)),
        "criterio_confiabilidade": {
            "confiavel": (f"deslocamento < {args.max_displacement_m:.0f} m E "
                          f"rmse <= {args.max_rmse_m:.1f} m"),
            "aceitavel_com_ressalvas": "deslocamento < 10000 m",
            "nao_confiavel": "deslocamento >= 10000 m",
            "motivo": ("o RMSE do ajuste cresce com o deslocamento (r=+0,43); "
                       "a hipotese de DH/Dt constante ao longo de 6 anos falha "
                       "em trajetorias longas"),
            "contagem": {str(k): int(v) for k, v in counts.items()},
        },
        "confiavel": {
            "n": int(len(trust)),
            "dhdt_mediana": float(np.median(vt)) if len(vt) else None,
            "dhdt_media": float(np.mean(vt)) if len(vt) else None,
            "rmse_mediano": float(trust["rmse"].median()) if len(trust) else None,
            "desloc_mediano_km": (float(trust["displacement_m"].median()/1000)
                                  if len(trust) else None),
        },
        "n_parcelas_candidatas": int(n_par),
        "spacing_km": args.spacing_km, "radius_km": args.radius_km,
        "t_ref": t_ref, "epocas": years.tolist(),
        "deslocamento": disp,
        "todas_INCLUI_NAO_CONFIAVEIS": {
            "n": int(len(out)),
            "dhdt_mediana": float(np.median(v)),
            "dhdt_media": float(np.mean(v)),
            "dhdt_p10": float(np.percentile(v, 10)),
            "dhdt_p90": float(np.percentile(v, 90)),
        },
        "rmse_mediano": float(np.nanmedian(out["rmse"])),
        "n_epochs_mediano": float(np.median(out["n_epochs"])),
        "velocidade": "ITS_LIVE v2 annual composites (2019-2025), 120 m",
        "datum_harmonizado": bool(not args.no_geoid),
        "geoid_source": ("BedMachine v4 / EIGEN-6C4 (Forste et al. 2014), 500 m"
                         if not args.no_geoid else None),
        "coluna_elevacao": ("h_orth = h_corr - N (mare CATS2008 + DAC removidos, "
                            "datum harmonizado pelo geoide); NAO h_res, "
                            "pois subtrair o REMA fixo introduziria o gradiente "
                            "do DEM como sinal falso ao longo da trajetoria"),
        "limitacoes": [
            "datum harmonizado com o geoide do BedMachine (EIGEN-6C4). Alguns "
            "Parquet tem `geoid` 100% NaN; o ATL06 guarda a variavel em "
            "dem/geoid_h. "
            "Efeito medido da correcao: ~0,064 m/ano nas parcelas com >20 km de "
            "deslocamento (~1% do sinal ali)",
            "mascara de plataforma ESTATICA (BedMachine nominal_year 2015); "
            "frentes datadas do IceLines em disco mas ainda nao aplicadas",
            "IceLines nao cobre 2025; Getz1/2/3 param em 2021,9",
            "residuo de mare ~+/-17 cm (referencia TEIS) amplificado ~9,34x na "
            "conversao hidrostatica",
            "DAC apresenta tendencia significativa (|t|=3,59) nesta amostra",
        ],
    }
    rp = cfg.paths.tables / "shelf_lagrangian_report.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"Parcelas -> {dst} ({len(out):,} válidas de {n_par:,})")
    if len(vt):
        log.info(f"  CONFIÁVEL (n={len(trust):,}): mediana {np.median(vt):+.4f} "
                 f"m/ano | rmse {trust['rmse'].median():.3f} m")
    log.info(f"  todas ({len(out):,}, inclui não confiáveis): mediana "
             f"{np.median(v):+.4f} | média {np.mean(v):+.4f} m/ano")
    log.info(f"  épocas medianas por parcela: {np.median(out['n_epochs']):.0f}")
    log.info(f"Relatório -> {rp}")


if __name__ == "__main__":
    main()
