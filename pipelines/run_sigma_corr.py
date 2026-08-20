"""
pipelines/run_sigma_corr.py
===========================
A2 — termo de erro de geolocalização × declividade (ATBD ATL14/ATL15 §3.4.4).

    data/qc_flags/*.parquet  (calibração)
    data/tiles/*.parquet     (estrutura de passagens)
    REMA                     (declividade)
        -> outputs/experiments/<nome>/
             sigma_corr_nodes.parquet
             sigma_corr_report.json

O que muda
----------
Quando a incerteza de cada nó usa somente a dispersão residual do ajuste, a
incerteza de massa usava UM comprimento de correlação para tudo. Este pipeline
acrescenta a componente que o ATBD chama de dominante e, com ela, permite
propagar as duas componentes com os seus comprimentos de correlação próprios
(ver `two_component_mass_sigma`).

Aproximação declarada
---------------------
A alavancagem temporal Σ(t_k − t̄)² é calculada POR TILE, não por nó. Um tile
tem ~50 km e as passagens do ICESat-2 o cruzam inteiro, então os nós de um
mesmo tile compartilham essencialmente o mesmo conjunto de épocas. A alternativa
— refazer a busca por raio em cada nó só para listar épocas — custaria uma
passagem completa sobre 200 milhões de pontos para alterar o terceiro decimal.

Uso: python pipelines/run_sigma_corr.py --name sigma_corr_v1
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
from thwaites.grid.tiles import load_manifest
from thwaites.experiments.manifest import Manifest
from thwaites.corrections.slope import resolve_rema_path
from thwaites.uncertainty.geolocation import (calibrate_from_qc,
                                              overpass_leverage,
                                              sigma_dhdt_from_geolocation,
                                              slope_magnitude_grid)
from thwaites.uncertainty.mass_balance import two_component_mass_sigma


def node_slopes(rema_path, nodes_x, nodes_y, radius_m, log):
    """
    Declividade representativa em cada nó: mediana de |∇REMA| num quadrado de
    lado 2·`radius_m` centrado no nó.

    Mediana e não média porque o REMA tem artefatos pontuais (costuras de
    faixa, buracos preenchidos) que produzem gradientes enormes em poucos
    pixels; a média os deixaria dominar a declividade do nó inteiro.
    """
    x0, x1 = nodes_x.min() - radius_m, nodes_x.max() + radius_m
    y0, y1 = nodes_y.min() - radius_m, nodes_y.max() + radius_m
    xs, ys, mag = slope_magnitude_grid(rema_path, x0, x1, y0, y1)
    if mag.size == 0:
        return np.full(nodes_x.size, np.nan)

    dx = xs[1] - xs[0] if xs.size > 1 else 32.0
    dy = abs(ys[1] - ys[0]) if ys.size > 1 else 32.0
    rx = max(int(round(radius_m / dx)), 1)
    ry = max(int(round(radius_m / dy)), 1)

    ix = np.clip(np.searchsorted(xs, nodes_x), 0, xs.size - 1)
    # ys é decrescente (raster do topo para baixo)
    iy = np.clip(np.searchsorted(-ys, -nodes_y), 0, ys.size - 1)

    out = np.full(nodes_x.size, np.nan)
    for k in range(nodes_x.size):
        a = mag[max(iy[k] - ry, 0):iy[k] + ry + 1,
                max(ix[k] - rx, 0):ix[k] + rx + 1]
        a = a[np.isfinite(a)]
        if a.size:
            out[k] = float(np.median(a))
    return out


def main():
    ap = argparse.ArgumentParser(description="A2 — erro de geolocalização×declividade.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--nodes", default="dhdt_nodes_qc.parquet")
    ap.add_argument("--slope-radius-m", type=float, default=2000.0)
    ap.add_argument("--n-granules", type=int, default=40)
    ap.add_argument("--l-corr-m", type=float, default=None,
                    help="comprimento de correlação da componente de "
                         "geolocalização; padrão = mass_balance.correlation_length_m")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"sigma_corr_{args.name}")

    # ---- 1. calibra o modelo de erro contra o próprio ATL06 ----------------
    model = calibrate_from_qc(cfg.paths.qc_flags,
                              n_granules=args.n_granules)

    # ---- 2. alavancagem temporal por tile ---------------------------------
    entries = load_manifest(cfg)
    lev_by_tile = {}
    for e in entries:
        t = pd.read_parquet(cfg.paths.tiles_dir / e["file"],
                            columns=["t_year"], engine="pyarrow")["t_year"]
        n_pass, lev = overpass_leverage(t.to_numpy(float))
        lev_by_tile[e["tile"]] = {"n_passagens": n_pass, "leverage": lev,
                                  "x_min": e["x_min"], "x_max": e["x_max"],
                                  "y_min": e["y_min"], "y_max": e["y_max"]}
    npass = np.array([v["n_passagens"] for v in lev_by_tile.values()])
    levs = np.array([v["leverage"] for v in lev_by_tile.values()])
    log.info(f"passagens por tile: mediana {np.median(npass):.0f} "
             f"(min {npass.min()}, max {npass.max()}) | "
             f"alavancagem Σ(t−t̄)² mediana {np.median(levs):.1f} ano²")

    # ---- 3. nós: declividade e σ_geo --------------------------------------
    nodes_path = cfg.paths.dhdt_dir / args.nodes
    if not nodes_path.exists():
        nodes_path = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
    nodes = pd.read_parquet(nodes_path, engine="pyarrow")
    log.info(f"{len(nodes):,} nós de {nodes_path.name}")

    rema = resolve_rema_path(cfg)
    nx = nodes["x"].to_numpy(float)
    ny = nodes["y"].to_numpy(float)

    # a leitura do REMA é feita por blocos de tile para não materializar o
    # mosaico inteiro (22.729 x 21.173 px)
    slope = np.full(len(nodes), np.nan)
    for tile, info in lev_by_tile.items():
        sel = ((nx >= info["x_min"]) & (nx < info["x_max"])
               & (ny >= info["y_min"]) & (ny < info["y_max"]))
        if not sel.any():
            continue
        slope[sel] = node_slopes(rema, nx[sel], ny[sel],
                                 args.slope_radius_m, log)
    nodes["slope_mag"] = slope

    # alavancagem do tile de cada nó
    lev = np.full(len(nodes), np.nan)
    for tile, info in lev_by_tile.items():
        sel = ((nx >= info["x_min"]) & (nx < info["x_max"])
               & (ny >= info["y_min"]) & (ny < info["y_max"]))
        lev[sel] = info["leverage"]
    nodes["leverage_yr2"] = lev

    nodes["sigma_geo_m"] = model.sigma_geo(nodes["slope_mag"].to_numpy())
    sg = nodes["sigma_geo_m"].to_numpy()
    nodes["sigma_dhdt_geo"] = np.where(
        np.isfinite(lev) & (lev > 0), sg / np.sqrt(np.maximum(lev, 1e-12)), np.nan)

    ok = np.isfinite(nodes["sigma_dhdt_geo"])
    sd_geo = nodes.loc[ok, "sigma_dhdt_geo"].to_numpy()
    sd_fit = nodes.loc[ok, "dhdt_err"].to_numpy()
    nodes["dhdt_err_total"] = np.hypot(nodes["dhdt_err"], nodes["sigma_dhdt_geo"])

    log.info(f"declividade nos nós: mediana {np.nanmedian(slope):.5f} "
             f"(p90 {np.nanpercentile(slope, 90):.5f})")
    log.info(f"σ_geo (altura): mediana {np.nanmedian(sg):.4f} m "
             f"(p90 {np.nanpercentile(sg, 90):.4f})")
    log.info(f"σ_dhdt geolocação: mediana {np.median(sd_geo):.5f} m/ano | "
             f"σ_dhdt do ajuste: mediana {np.median(sd_fit):.5f} m/ano | "
             f"razão mediana {np.median(sd_geo/np.maximum(sd_fit,1e-12)):.2f}")

    # ---- 4. efeito na incerteza de massa ----------------------------------
    mb = cfg.mass_balance
    L_corr = args.l_corr_m if args.l_corr_m else mb.correlation_length_m
    cell = cfg.interpolation.grid_res_m ** 2
    # área do produto aterrado consolidado (mesma base do -110 Gt/ano)
    area_m2 = 200_600.0 * 1e6

    s_white = float(np.sqrt(np.mean(sd_fit ** 2)))
    s_corr = float(np.sqrt(np.mean(sd_geo ** 2)))

    antes = two_component_mass_sigma(s_white, None, 0.0, L_corr,
                                     area_m2, cell, mb.ice_density)
    depois = two_component_mass_sigma(s_white, None, s_corr, L_corr,
                                      area_m2, cell, mb.ice_density)
    unico_L = two_component_mass_sigma(
        float(np.sqrt(s_white ** 2 + s_corr ** 2)), L_corr, 0.0, L_corr,
        area_m2, cell, mb.ice_density)

    rep = {
        "modelo_de_erro": model.as_dict(),
        "referencia_atbd": ("ATBD ATL14/ATL15 r005 §3.4.4 — erro de "
                            "geolocalização sobre superfície inclinada como "
                            "fonte dominante de erro correlacionado, "
                            "'consistent over ... spatial scales of tens of km'"),
        "passagens": {
            "mediana_por_tile": float(np.median(npass)),
            "min": int(npass.min()), "max": int(npass.max()),
            "alavancagem_mediana_ano2": float(np.median(levs)),
            "nota": ("é o nº de PASSAGENS, não de segmentos, que define a "
                     "amostra efetiva deste termo — a mediana de 163.178 "
                     "observações por nó não se traduz em 163.178 amostras "
                     "independentes do erro de apontamento"),
        },
        "nos": {
            "n": int(ok.sum()),
            "slope_mediana": float(np.nanmedian(slope)),
            "sigma_geo_altura_mediana_m": float(np.nanmedian(sg)),
            "sigma_dhdt_geo_mediana": float(np.median(sd_geo)),
            "sigma_dhdt_ajuste_mediana": float(np.median(sd_fit)),
            "razao_mediana": float(np.median(sd_geo / np.maximum(sd_fit, 1e-12))),
        },
        "incerteza_de_massa_Gt_ano": {
            "so_ajuste_branca": antes["sigma_dMdt_Gt_yr"],
            "duas_componentes": depois["sigma_dMdt_Gt_yr"],
            "detalhe_duas_componentes": depois,
            "componente_unica_com_L_correlacionado": unico_L["sigma_dMdt_Gt_yr"],
            "nota": ("'componente_unica' é o tratamento ANTIGO: aplica o L "
                     "correlacionado a TODA a variância, inclusive à parte "
                     "branca. É por isso que ele inflava a barra de erro e a "
                     "deixava tão sensível à escolha de L."),
            "ALERTA": (
                "NÃO adotar 'duas_componentes' como nova barra de erro. O valor "
                "só vale se a dispersão residual do ajuste for BRANCA na escala "
                "da célula, e essa premissa não está verificada — há evidência "
                "DIRETA contra ela: (a) o variograma do próprio campo tem "
                "alcance de 34-154 km, o que é estrutura espacial, não ruído "
                "branco; (b) o run_acceleration rejeitou 99,4% dos nós por "
                "resíduo autocorrelado (ac1 = 0,52-0,70), o que é estrutura "
                "temporal. Adotar 0,33 Gt/ano subestimaria a incerteza por um "
                "fator ~8. O número está aqui para mostrar o TAMANHO do efeito "
                "da decomposição, não para substituir a barra de erro."),
        },
        "conclusao": (
            "A2 NÃO resolve A1, e o motivo é informativo. O ATBD aponta a "
            "geolocalização×declividade como fonte DOMINANTE de erro "
            "correlacionado, mas na nossa geometria ela responde por apenas "
            "14% do σ do nó: com ~94 passagens por tile e 6,2 anos de "
            "alavancagem (Σ(t−t̄)² = 355 ano²), o erro de apontamento por "
            "passagem é fortemente promediado. A discrepância com o ATBD é "
            "esperada — eles descrevem alturas por ponto de referência do "
            "ATL11, geometria diferente da nossa média por nó de 15 km. "
            "Consequência: a estrutura espacial de 34-154 km que domina a "
            "nossa barra de erro NÃO é de geolocalização. Sobra origem "
            "geofísica — variabilidade de firn/SMB ou dinâmica — e é aí que a "
            "investigação de A1 deve continuar, não em mais termos "
            "instrumentais."),
        "limitacoes": [
            "alavancagem temporal calculada por TILE, não por nó (ver docstring)",
            "σ_horiz é calibrado em 40 grânulos amostrados, não nos 1.106",
            "a declividade vem do REMA (época ~2015) e não do ATL06; em zonas "
            "de mudança rápida de superfície as duas divergem",
            "o L da componente correlacionada continua sendo escolhido, não "
            "medido — o ganho aqui é que ele agora se aplica só à parte do "
            "erro que o ATBD atribui a essa escala, não a toda a variância",
        ],
    }

    man = Manifest(cfg, args.name,
                   purpose="A2 — erro de geolocalização×declividade (ATBD §3.4.4)",
                   overwrite=args.overwrite, seed=0)
    man.add_input(nodes_path, columns=["x", "y", "dhdt", "dhdt_err"])
    out_nodes = man.path_for("sigma_corr_nodes.parquet")
    nodes.to_parquet(out_nodes, index=False, engine="pyarrow")
    man.add_output(out_nodes)
    out_json = man.path_for("sigma_corr_report.json")
    out_json.write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    man.add_output(out_json)
    man.set("geo_error_model", model.as_dict())
    man.write()

    log.info("=" * 64)
    log.info(f"  σ_massa só ajuste (branca)      : {antes['sigma_dMdt_Gt_yr']:6.2f} Gt/ano")
    log.info(f"  σ_massa duas componentes        : {depois['sigma_dMdt_Gt_yr']:6.2f} Gt/ano"
             f"  (branca {depois['contrib_branca_Gt_yr']:.2f} + "
             f"geoloc {depois['contrib_correlacionada_Gt_yr']:.2f})")
    log.info(f"  σ_massa tratamento antigo (L p/ tudo): {unico_L['sigma_dMdt_Gt_yr']:6.2f} Gt/ano")
    log.info("=" * 64)
    log.info(f"Experimento em {man.dir}")


if __name__ == "__main__":
    main()
