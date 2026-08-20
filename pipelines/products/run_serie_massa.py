"""
pipelines/run_serie_massa.py
============================
Série temporal de massa acumulada e mapa em metros de água equivalente — os
dois painéis do produto GRACE, mas construídos com ICESat-2 e restritos à ROI.

    data/<estação>/tiles/*.parquet
        -> data/<estação>/dhdt/serie_anual.parquet
        -> outputs/<estação>/tables/serie_massa.json

Escopo — a diferença que importa
--------------------------------
O produto do GRACE cobre a Antártica inteira desde 2002 e mede a massa
DIRETAMENTE, pela deformação do campo gravitacional. Aqui a cobertura é o
Amundsen Sea Embayment (200.600 km², ~1,4% do continente), o período é
2019–2025, e a massa não é medida: é INFERIDA da variação de altura da
superfície multiplicada por uma densidade suposta.

As duas coisas não são intercambiáveis. O que se pode fazer é comparar o total
regional com o continental e mostrar quanto desta bacia responde pelo todo.

Como a série é construída
-------------------------
Para cada nó e cada ano, a mediana de `h_res` (altura menos o REMA). A anomalia
de cada nó é tomada em relação ao PRIMEIRO ano em que ele tem dado — assim a
curva começa em zero e desce, que é a leitura direta de "quanto baixou desde o
início". A massa é a média das anomalias ponderada por área, vezes densidade.

Por que a mediana por nó e não a média dos pontos: a densidade de observações
varia muito entre nós, e a média simples deixaria os nós mais visitados
dominarem a curva regional.

Uso: python pipelines/run_serie_massa.py --profile djf
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
from thwaites.grid.tiles import load_manifest, assign_xy
from thwaites.io.memory import free_memory_gb

RHO_ICE = 917.0
RHO_W = 1000.0


def por_no_e_ano(df, cfg, x_min, x_max, y_min, y_max, min_pts=30):
    """Mediana de h_res por nó e por ano, dentro do núcleo do tile."""
    from scipy.spatial import cKDTree

    d = cfg.dhdt
    df = assign_xy(df, cfg)
    col = "h_res" if "h_res" in df.columns else "h_corr"
    ok = ~(df["x"].isna() | df["y"].isna() | df[col].isna() | df["t_year"].isna())
    x = df["x"].to_numpy()[ok]
    y = df["y"].to_numpy()[ok]
    h = df[col].to_numpy()[ok].astype(float)
    t = df["t_year"].to_numpy()[ok].astype(float)
    if len(h) < min_pts:
        return pd.DataFrame()
    ano = np.floor(t).astype(int)

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
        hi, ai = h[idx], ano[idx]
        for a in np.unique(ai):
            m = ai == a
            if m.sum() < 12:
                continue
            linhas.append((float(nx), float(ny), int(a), float(np.median(hi[m])),
                           int(m.sum())))
    return pd.DataFrame(linhas, columns=["x", "y", "ano", "h", "n"])


def main():
    ap = argparse.ArgumentParser(description="Série de massa acumulada.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--max-tiles", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    est = cfg.season.name
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"serie_massa_{est}")

    ent = load_manifest(cfg)
    if args.max_tiles:
        ent = ent[:args.max_tiles]
    log.info(f"[{est}] elevação por nó e ano, {len(ent)} tiles")

    import pyarrow.parquet as pq
    COLS = ["x", "y", "lon", "lat", "t_year", "h_res", "h_corr"]
    partes = []
    for i, e in enumerate(ent, 1):
        p = cfg.paths.tiles_dir / e["file"]
        disp = pq.ParquetFile(p).schema_arrow.names
        df = pd.read_parquet(p, columns=[c for c in COLS if c in disp],
                             engine="pyarrow")
        r = por_no_e_ano(df, cfg, e["x_min"], e["x_max"], e["y_min"], e["y_max"])
        del df
        if len(r):
            partes.append(r)
        if i % 25 == 0 or i == len(ent):
            log.info(f"  [{i}/{len(ent)}] livre {free_memory_gb():.1f} GB")

    S = pd.concat(partes, ignore_index=True)
    qc = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
                         columns=["x", "y"])
    chave = set(zip(np.rint(qc.x).astype(np.int64), np.rint(qc.y).astype(np.int64)))
    S = S[np.fromiter(((a, b) in chave for a, b in
                       zip(np.rint(S.x).astype(np.int64),
                           np.rint(S.y).astype(np.int64))), bool, len(S))]

    # anomalia de cada nó em relação ao PRIMEIRO ano em que ele aparece
    S = S.sort_values(["x", "y", "ano"])
    base = S.groupby(["x", "y"]).h.transform("first")
    S["anom"] = S.h - base
    S.to_parquet(cfg.paths.dhdt_dir / "serie_anual.parquet", index=False)

    # Só os nós presentes em TODOS os anos entram na curva regional. Sem isso a
    # curva mistura mudança de elevação com mudança de QUEM foi medido, e um ano
    # com cobertura diferente aparece como salto de massa que não existe.
    cont = S.groupby(["x", "y"]).ano.nunique()
    completos = set(cont[cont == S.ano.nunique()].index)
    C = S[[(a, b) in completos for a, b in zip(S.x, S.y)]]
    log.info(f"{len(completos):,} nós com todos os {S.ano.nunique()} anos "
             f"(de {S.groupby(['x','y']).ngroups:,})")

    area = cfg.dhdt.node_spacing_m ** 2
    serie = []
    for a, g in C.groupby("ano"):
        # MASSA usa a MÉDIA, não a mediana.
        #
        # O campo é fortemente assimétrico (skew −2,05: p10 = −8,8 m, p90 = +0,6 m):
        # poucos nós no tronco rápido carregam a maior parte da perda. A mediana
        # descreve o nó TÍPICO e vale para caracterizar o campo; a massa é uma
        # SOMA sobre a área, e soma pede média. Usar a mediana dava −41,4 Gt/ano
        # contra os −110 do produto de balanço — um fator 2 que não é dado, é
        # escolha errada de estatística.
        dh_med = float(g.anom.mean())
        massa = dh_med * area * len(completos) * RHO_ICE / 1e12
        serie.append({"ano": int(a), "dh_medio_m": dh_med,
                      "dh_mediano_m": float(g.anom.median()),
                      "mwe": dh_med * RHO_ICE / RHO_W,
                      "massa_acumulada_Gt": massa, "n_nos": int(len(g))})
    rel = {"estacao": est, "n_nos_completos": len(completos),
           "area_km2": len(completos) * area / 1e6,
           "densidade_gelo": RHO_ICE, "serie": serie,
           "nota": ("massa INFERIDA de altura × densidade suposta, não medida "
                    "como no GRACE; anomalia relativa ao primeiro ano de cada nó")}
    rp = cfg.paths.tables / "serie_massa.json"
    rp.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"{'ano':>6} {'Δh (m)':>9} {'m w.e.':>9} {'massa (Gt)':>12}")
    for s in serie:
        log.info(f"{s['ano']:>6} {s['dh_medio_m']:>9.3f} {s['mwe']:>9.3f} "
                 f"{s['massa_acumulada_Gt']:>12.1f}")
    log.info(f"-> {rp}")


if __name__ == "__main__":
    main()
