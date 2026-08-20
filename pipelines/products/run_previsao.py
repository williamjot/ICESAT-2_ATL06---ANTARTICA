"""
pipelines/run_previsao.py
=========================
Até onde dá para extrapolar dh/dt — e quanto vale a extrapolação.

    data/<estação>/tiles/*.parquet
        -> outputs/<estação>/tables/previsao_skill.json
        -> data/<estação>/dhdt/previsao_nodes.parquet

O problema com "prever"
-----------------------
Extrapolar uma reta é trivial e quase sempre errado. A pergunta que importa não é
qual valor a reta dá em 2035, é: **a reta ajustada nos primeiros anos acertou os
últimos?** Isso é verificável com o registro que já existe, e é o único jeito de
declarar um horizonte sem inventá-lo.

O teste (hindcast)
------------------
Ajusta-se o dh/dt usando SÓ os anos até `corte`, e prevê-se a elevação de cada
ano posterior. O erro de previsão é comparado com duas referências:

  * PERSISTÊNCIA — supor que nada muda (dh/dt = 0). Se a extrapolação não bate
    isso, ela não tem competência nenhuma.
  * a própria dispersão do dado no ano previsto.

O horizonte viável é o maior avanço em que a extrapolação ainda vence a
persistência por margem folgada.

Por que o horizonte NÃO é o que a barra de erro do dh/dt sugere
--------------------------------------------------------------
A incerteza formal cresce linearmente com o avanço e mantém o erro RELATIVO
constante — por ela, extrapolar cem anos pareceria tão bom quanto extrapolar um.
O que de fato limita é a taxa não ser constante: as janelas móveis deste projeto
variam de −0,41 a −0,65 m/ano, e o `run_acceleration` rejeita 99,6% dos nós por
resíduo autocorrelado (ac1 entre 0,52 e 0,70), que é a assinatura de
variabilidade interanual não capturada por reta nenhuma.

Uso: python pipelines/run_previsao.py --profile djf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
from thwaites.io.memory import free_memory_gb


def hindcast_tile(df, cfg, x_min, x_max, y_min, y_max, corte, min_pts=30):
    """
    Por nó: ajusta com t < corte e prevê CADA PONTO observado depois.

    A comparação é PONTO A PONTO, com o modelo avaliado na posição e na época
    de cada observação. Comparar o intercepto no centro do nó com a mediana de
    um disco de 15 km confundiria a estrutura espacial de `h_res` com erro de
    previsão.

    A matriz de projeto é construída UMA vez para treino e teste juntos. Isso
    não é detalhe: `_build_A` normaliza as colunas espaciais pelo desvio da
    amostra que recebe, então montá-la em separado daria escalas diferentes e o
    coeficiente ajustado no treino não valeria no teste.

    A referência de PERSISTÊNCIA é o mesmo modelo sem a coluna temporal —
    mesmo tratamento espacial, mesma robustez, sem tendência. Assim a
    comparação isola exatamente o que se quer testar.
    """
    from scipy.spatial import cKDTree
    from thwaites.grid.tiles import assign_xy
    from thwaites.timeseries.dhdt import _build_A, _lstsq_iter

    d = cfg.dhdt
    df = assign_xy(df, cfg)
    col = "h_res" if "h_res" in df.columns else "h_corr"
    ok = ~(df["x"].isna() | df["y"].isna() | df[col].isna() | df["t_year"].isna())
    x = df["x"].to_numpy()[ok]
    y = df["y"].to_numpy()[ok]
    h = df[col].to_numpy()[ok].astype(float)
    t = df["t_year"].to_numpy()[ok].astype(float)
    s = (df["s_elv"].to_numpy()[ok].astype(float)
         if "s_elv" in df.columns else np.full(int(ok.sum()), .05))
    if len(h) < min_pts:
        return pd.DataFrame()

    passo = d.node_spacing_m
    gx = np.arange(x_min + passo / 2, x_max, passo)
    gy = np.arange(y_min + passo / 2, y_max, passo)
    if not len(gx) or not len(gy):
        return pd.DataFrame()
    GX, GY = np.meshgrid(gx, gy)
    arv = cKDTree(np.c_[x, y])

    linhas = []
    for nx, ny in zip(GX.ravel(), GY.ravel()):
        idx = np.asarray(arv.query_ball_point([nx, ny], r=d.search_radius_m))
        if idx.size < min_pts:
            continue
        ti, hi, si = t[idx], h[idx], s[idx]
        dx, dy = x[idx] - nx, y[idx] - ny
        tr = ti < corte
        te = ~tr
        if tr.sum() < min_pts or te.sum() < 8:
            continue
        if (ti[tr].max() - ti[tr].min()) < d.dt_min_years:
            continue

        A = _build_A(dx, dy, ti - d.t_ref, d.poly_order, 1)     # escala comum
        w = 1.0 / (np.where(si <= 0, .05, si) ** 2 + 1e-12)

        ct, _, mt, _ = _lstsq_iter(A[tr], hi[tr], w[tr], d.max_iter, d.n_sigma,
                                   d.resid_limit)
        if ct is None or mt.sum() < min_pts:
            continue
        if abs(float(ct[1])) > d.rate_limit:
            continue
        # persistência: mesmo modelo, coluna temporal removida
        Ap = np.delete(A, 1, axis=1)
        cp, _, mp, _ = _lstsq_iter(Ap[tr], hi[tr], w[tr], d.max_iter, d.n_sigma,
                                   d.resid_limit)
        if cp is None:
            continue

        prev = A[te] @ ct
        pers = Ap[te] @ cp
        ano = np.floor(ti[te]).astype(int)
        for a in np.unique(ano):
            m = ano == a
            if m.sum() < 8:
                continue
            linhas.append((float(nx), float(ny), int(a),
                           float(np.median(ti[te][m]) - (corte - .5)),
                           float(np.sqrt(np.mean((hi[te][m] - prev[m]) ** 2))),
                           float(np.sqrt(np.mean((hi[te][m] - pers[m]) ** 2))),
                           float(np.mean(hi[te][m] - prev[m])),
                           float(ct[1]), int(tr.sum())))
    return pd.DataFrame(linhas, columns=["x", "y", "ano", "avanco", "rms_extrap",
                                         "rms_persist", "vies", "taxa", "n_treino"])


def main():
    ap = argparse.ArgumentParser(description="Competência da extrapolação de dh/dt.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--corte", type=float, default=2023.0,
                    help="ajusta com t < corte, prevê o resto")
    ap.add_argument("--max-tiles", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    est = cfg.season.name
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"previsao_{est}")

    ent = load_manifest(cfg)
    if args.max_tiles:
        ent = ent[:args.max_tiles]
    log.info(f"[{est}] hindcast: treino em t < {args.corte:.0f}, "
             f"{len(ent)} tiles")

    COLS = ["x", "y", "lon", "lat", "t_year", "h_res", "h_corr", "s_elv"]
    import pyarrow.parquet as pq
    partes = []
    for i, e in enumerate(ent, 1):
        p = cfg.paths.tiles_dir / e["file"]
        disp = pq.ParquetFile(p).schema_arrow.names
        df = pd.read_parquet(p, columns=[c for c in COLS if c in disp],
                             engine="pyarrow")
        r = hindcast_tile(df, cfg, e["x_min"], e["x_max"], e["y_min"], e["y_max"],
                          args.corte)
        del df
        if len(r):
            partes.append(r)
        if i % 20 == 0 or i == len(ent):
            log.info(f"  [{i}/{len(ent)}] livre {free_memory_gb():.1f} GB")

    if not partes:
        log.warning("sem resultado")
        return
    H = pd.concat(partes, ignore_index=True)

    # recorte ao conjunto validado de nós
    qc = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
                         columns=["x", "y"])
    chave = set(zip(np.rint(qc.x).astype(np.int64), np.rint(qc.y).astype(np.int64)))
    manter = np.fromiter(((a, b) in chave for a, b in
                          zip(np.rint(H.x).astype(np.int64),
                              np.rint(H.y).astype(np.int64))), bool, len(H))
    H = H[manter]
    H.to_parquet(cfg.paths.dhdt_dir / "previsao_nodes.parquet", index=False)

    rel = {"estacao": est, "corte": args.corte,
           "n_nos": int(len(H.drop_duplicates(subset=["x", "y"]))),
           "por_avanco": {}}
    nn = len(H.drop_duplicates(subset=["x", "y"]))
    log.info(f"{nn:,} nós | {len(H):,} avaliações")
    log.info(f"{'avanço':>7} {'n':>7} {'RMS extrap':>11} {'RMS persist':>12} "
             f"{'ganho':>7}")
    for ano, g in H.groupby("ano"):
        av = float(g.avanco.median())
        re_ = float(np.sqrt(np.mean(g.rms_extrap ** 2)))
        rp = float(np.sqrt(np.mean(g.rms_persist ** 2)))
        ganho = 1 - re_ / rp if rp > 0 else np.nan
        rel["por_avanco"][f"{av:.1f}"] = {
            "ano": int(ano), "n": int(len(g)),
            "rms_extrapolacao_m": re_, "rms_persistencia_m": rp,
            "ganho_sobre_persistencia": ganho,
            "vies_m": float(g.vies.mean()),
        }
        log.info(f"{av:>7.1f} {len(g):>7,} {re_:>11.3f} {rp:>12.3f} "
                 f"{100*ganho:>6.1f}%")

    rel["leitura"] = (
        "ganho = 1 − RMS(extrapolação)/RMS(persistência). Positivo significa que "
        "estender a reta erra menos do que supor que nada muda. Ganho caindo com o "
        "avanço marca a perda de competência; ganho negativo significa que a "
        "extrapolação é PIOR que não prever nada.")
    rp = cfg.paths.tables / "previsao_skill.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"-> {rp}")


if __name__ == "__main__":
    main()
