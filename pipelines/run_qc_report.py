"""
pipelines/run_qc_report.py
==========================
Controle de qualidade consolidado do recorte espacial e do dh/dt.

Faz três coisas, nesta ordem:

1. FILTRA OS NÓS por posição. Filtrar os pontos não basta: os nós ficam numa
   grade regular e o raio de busca (15 km) alcança pontos de gelo aterrado a
   partir de uma posição sobre o mar. Medido antes da correção: 771 nós (7,9%)
   estavam sobre OCEANO e 1.128 sobre plataforma — 19,4% fora do alvo.
   O critério exige que o próprio nó esteja em gelo aterrado, respeitados os
   buffers, e que uma fração mínima do seu raio de busca também esteja.

2. CLASSIFICA cada nó em confiável / aceitável com ressalvas / não confiável,
   por critérios pré-declarados (thwaites/qc/reliability.py).

3. PRODUZ os mapas e as tabelas de QC, incluindo a comparação antes/depois.

Saídas:
    data/dhdt/dhdt_nodes_qc.parquet          (nós filtrados + colunas de QC)
    outputs/tables/qc_report.json
    outputs/tables/reliability_criteria.json
    outputs/figures/qc_*.png

Uso: python pipelines/run_qc_report.py [--profile P] [--no-filter-nodes]
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
from thwaites.qc.grounded_mask import (sample_fields_at, BM_NAMES,
                                       BM_GROUNDED_ICE)
from thwaites.qc.reliability import (classify_nodes, reliability_report,
                                     CRITERIA)
from thwaites.viz.qc_maps import (fig_mask_map, fig_points_kept_removed,
                                  fig_nobs_map, fig_temporal_distribution,
                                  fig_reliability_map)


def load_dist(cfg):
    z = np.load(cfg.paths.interim / "bedmachine_distfields.npz")
    return (z["sx"], z["sy"], z["mask"],
            {k: z[k] for k in ("dist_to_nongrounded", "dist_to_floating",
                               "dist_to_ocean")})


def grounded_fraction(nodes, sx, sy, keep_field, radius_m, n_ring=16):
    """
    Fração do raio de busca de cada nó que cai em área analisável.

    Amostra um anel de `n_ring` direções no raio — barato e suficiente para
    distinguir um nó interior (fração 1) de um nó de margem (fração ~0,5), que
    é o efeito de borda que interessa medir.
    """
    ang = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    dx = sx[1] - sx[0]
    dy = sy[1] - sy[0]
    acc = np.zeros(len(nodes), dtype=float)
    for a in ang:
        px = nodes["x"].to_numpy() + radius_m * np.cos(a)
        py = nodes["y"].to_numpy() + radius_m * np.sin(a)
        j = np.clip(np.rint((px - sx[0]) / dx).astype(np.int64), 0, len(sx) - 1)
        i = np.clip(np.rint((py - sy[0]) / dy).astype(np.int64), 0, len(sy) - 1)
        acc += keep_field[i, j].astype(float)
    return acc / n_ring


def main():
    ap = argparse.ArgumentParser(description="QC do recorte espacial e do dh/dt.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--no-filter-nodes", action="store_true")
    ap.add_argument("--sample", type=int, default=250_000,
                    help="pontos amostrados para os mapas de dispersão")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="qc_report")
    figs = cfg.paths.figures
    tables = cfg.paths.tables
    tables.mkdir(parents=True, exist_ok=True)

    sx, sy, bm, F = load_dist(cfg)
    px_m = abs(sx[1] - sx[0])
    gl_buf = cfg.grounded.buffer_grounding_line_m
    co_buf = cfg.grounded.buffer_coast_m

    # área analisável na grade do BedMachine (mesma regra dos pontos)
    keep_field = ((bm == BM_GROUNDED_ICE) &
                  (F["dist_to_nongrounded"] >= co_buf) &
                  (F["dist_to_floating"] >= gl_buf))
    log.info(f"área analisável: {100*keep_field.mean():.1f}% da janela "
             f"({keep_field.sum()*(px_m/1000)**2:,.0f} km²)")

    # ---------------------------------------------------------------- nós
    nodes = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes.parquet")
    n_before = len(nodes)
    s = sample_fields_at(nodes["x"], nodes["y"], sx, sy, F)
    dxp = sx[1] - sx[0]
    dyp = sy[1] - sy[0]
    j = np.clip(np.rint((nodes["x"].to_numpy() - sx[0]) / dxp).astype(np.int64), 0, len(sx) - 1)
    i = np.clip(np.rint((nodes["y"].to_numpy() - sy[0]) / dyp).astype(np.int64), 0, len(sy) - 1)
    nodes["mask_class"] = bm[i, j]
    nodes["dist_gl"] = s["dist_to_floating"]
    nodes["dist_coast"] = s["dist_to_nongrounded"]
    nodes["grounded_frac"] = grounded_fraction(nodes, sx, sy, keep_field,
                                               cfg.dhdt.search_radius_m)

    class_before = {BM_NAMES.get(int(k), str(k)): int(v) for k, v in
                    zip(*np.unique(nodes["mask_class"], return_counts=True))}

    node_ok = ((nodes["mask_class"] == BM_GROUNDED_ICE) &
               (nodes["dist_coast"] >= co_buf) &
               (nodes["dist_gl"] >= gl_buf) &
               (nodes["grounded_frac"] >= cfg.grounded.min_grounded_fraction))
    if args.no_filter_nodes or not cfg.grounded.filter_nodes:
        log.warning("filtro de nós DESLIGADO — nós fora do gelo aterrado seguem "
                    "na amostra.")
        kept = nodes.copy()
    else:
        kept = nodes[node_ok].copy()
    log.info(f"nós: {n_before:,} -> {len(kept):,} "
             f"({100*len(kept)/max(n_before,1):.1f}% mantidos)")

    kept = classify_nodes(kept)
    rep_rel = reliability_report(kept)
    out_nodes = cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet"
    kept.to_parquet(out_nodes, index=False)
    log.info(f"nós QC -> {out_nodes}")

    # ------------------------------------------------- comparação antes/depois
    prev_p = ROOT / "data" / "pre_grounded" / "dhdt_nodes.parquet"
    prev = pd.read_parquet(prev_p) if prev_p.exists() else None

    def summ(d, name):
        v = d["dhdt"].to_numpy()
        v = v[np.isfinite(v)]
        return {"conjunto": name, "n_nos": int(len(d)),
                "dhdt_mediana": float(np.median(v)),
                "dhdt_media": float(np.mean(v)),
                "afinando_pct": float(100 * np.mean(v < 0)),
                "sigma_mediano": float(d["dhdt_err"].median()),
                "rmse_mediano": float(d["rmse"].median()),
                "nobs_mediano": float(d["nobs"].median())}

    comparison = [summ(kept, "novo (máscara de gelo aterrado)")]
    if prev is not None:
        comparison.insert(0, summ(prev, "antigo (máscara larga)"))

    report = {
        "mascara": {
            "fonte": "BedMachine Antarctica v4, variável 'mask', 500 m, EPSG:3031",
            "classe_alvo": f"{BM_GROUNDED_ICE} (grounded_ice)",
            "buffer_grounding_line_m": gl_buf,
            "buffer_coast_m": co_buf,
            "min_grounded_fraction": cfg.grounded.min_grounded_fraction,
            "search_radius_m": cfg.dhdt.search_radius_m,
            "area_analisavel_km2": float(keep_field.sum() * (px_m / 1000) ** 2),
            "area_analisavel_pct_janela": float(100 * keep_field.mean()),
        },
        "nos": {
            "antes": n_before, "depois": int(len(kept)),
            "removidos": int(n_before - len(kept)),
            "classes_antes": class_before,
            "motivo_remocao": {
                "fora_de_grounded_ice":
                    int((nodes["mask_class"] != BM_GROUNDED_ICE).sum()),
                "dentro_buffer_costa":
                    int(((nodes["mask_class"] == BM_GROUNDED_ICE) &
                         (nodes["dist_coast"] < co_buf)).sum()),
                "dentro_buffer_grounding_line":
                    int(((nodes["mask_class"] == BM_GROUNDED_ICE) &
                         (nodes["dist_coast"] >= co_buf) &
                         (nodes["dist_gl"] < gl_buf)).sum()),
                "fracao_aterrada_insuficiente":
                    int(((nodes["mask_class"] == BM_GROUNDED_ICE) &
                         (nodes["dist_coast"] >= co_buf) &
                         (nodes["dist_gl"] >= gl_buf) &
                         (nodes["grounded_frac"] <
                          cfg.grounded.min_grounded_fraction)).sum()),
            },
        },
        "confiabilidade": rep_rel,
        "comparacao_antes_depois": comparison,
    }
    (tables / "qc_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (tables / "reliability_criteria.json").write_text(
        json.dumps(CRITERIA, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"relatório -> {tables/'qc_report.json'}")

    # ------------------------------------------------------------- figuras
    out = fig_mask_map(sx, sy, bm, cfg, figs / "qc_mask_map.png",
                       keep_field=keep_field)
    log.info(f"figura -> {out}")

    out = fig_nobs_map(kept, cfg, figs / "qc_nobs_map.png")
    log.info(f"figura -> {out}")

    out = fig_reliability_map(kept, cfg, figs / "qc_reliability_map.png")
    log.info(f"figura -> {out}")

    # pontos mantidos vs removidos (amostra aleatória, fração declarada)
    import pyarrow.parquet as pq
    rng = np.random.default_rng(0)
    gp = cfg.paths.interim / "atl06_grounded.parquet"
    fp = cfg.paths.interim / "atl06_filtered.parquet"
    if gp.exists() and fp.exists():
        kept_pts, t_after = _sample_points(gp, args.sample, rng)
        all_pts, t_before = _sample_points(fp, args.sample * 2, rng,
                                           with_class=True)
        rem_xy, rem_reason = _removed_sample(all_pts, sx, sy, F, co_buf, gl_buf)
        out = fig_points_kept_removed(kept_pts, rem_xy, rem_reason, cfg,
                                      figs / "qc_points_kept_removed.png")
        log.info(f"figura -> {out}")
        out = fig_temporal_distribution(t_after, cfg,
                                        figs / "qc_temporal_distribution.png",
                                        t_before=t_before)
        log.info(f"figura -> {out}")

    log.info("QC concluído.")
    log.info("Próximo: run_interpolation.py --nodes dhdt_nodes_qc.parquet "
             "e depois run_figures.py")


def _sample_points(path, n, rng, with_class=False):
    """Amostra aleatória de pontos (x, y, t_year[, mask_class]) sem carregar tudo."""
    import pyarrow.parquet as pq
    cols = ["x", "y", "t_year"] + (["mask_class"] if with_class else [])
    f = pq.ParquetFile(path)
    total = f.metadata.num_rows
    frac = min(1.0, n / max(total, 1))
    xs, ys, ts, cs = [], [], [], []
    for b in f.iter_batches(batch_size=2_000_000, columns=cols):
        d = b.to_pydict()
        k = len(d["x"])
        sel = rng.random(k) < frac
        xs.append(np.asarray(d["x"])[sel])
        ys.append(np.asarray(d["y"])[sel])
        ts.append(np.asarray(d["t_year"])[sel])
        if with_class:
            cs.append(np.asarray(d["mask_class"])[sel])
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    T = np.concatenate(ts)
    if with_class:
        return (np.c_[X, Y, np.concatenate(cs)], T)
    return np.c_[X, Y], T


def _removed_sample(all_pts, sx, sy, F, co_buf, gl_buf):
    """Classifica a amostra pelo motivo de remoção (mesma precedência do pipeline)."""
    x, y, mc = all_pts[:, 0], all_pts[:, 1], all_pts[:, 2]
    s = sample_fields_at(x, y, sx, sy, F)
    is_gr = mc == BM_GROUNDED_ICE
    ok_coast = s["dist_to_nongrounded"] >= co_buf
    ok_gl = s["dist_to_floating"] >= gl_buf
    reason = np.full(len(x), "", dtype=object)
    reason[~is_gr] = "não-aterrado"
    reason[is_gr & ~ok_coast] = "buffer costa"
    reason[is_gr & ok_coast & ~ok_gl] = "buffer linha de aterramento"
    sel = reason != ""
    return np.c_[x[sel], y[sel]], reason[sel]


if __name__ == "__main__":
    main()
