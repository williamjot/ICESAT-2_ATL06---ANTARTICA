"""
thwaites.experiments.sensitivity
================================
Prioridade 2 (§3 do PLANO): análise de sensibilidade dos filtros de qualidade.

PERGUNTA: quanto as escolhas de filtro alteram cobertura, dispersão, resíduos e
dh/dt? Nenhum parâmetro deve ser mantido só porque já foi usado antes.

DESENHO
-------
- varia UM parâmetro por vez a partir de uma configuração-base (§3.3);
- parte do Parquet `atl06_slopecorr.parquet` (ANTES do filttrack), porque variar
  parâmetros do filtro sobre dados já filtrados seria filtragem dupla;
- roda em N sub-regiões representativas em vez do domínio inteiro. Rodar todas
  as configurações no domínio completo levaria ~14 h; com 6 sub-regiões de
  50 km cai para ~1 h. **Isso é uma limitação a declarar**: a sensibilidade é
  avaliada em amostra, não no domínio todo;
- as sub-regiões são escolhidas para cobrir regimes distintos (aterrado/
  flutuante, denso/esparso, alto/baixo |dh/dt|), não aleatoriamente;
- os critérios de aceite são PRÉ-DEFINIDOS na config e vão ao manifesto antes
  da comparação (§3.5).

COMPARAÇÃO (§3.4): não se limita a médias globais — usa diferenças PAREADAS por
nó, com intervalo bootstrap, e estratifica por classe de máscara e densidade.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger


# --------------------------------------------------------------- utilidades
def apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """
    Cópia da config com sobrescritas em notação de ponto.
    Ex.: {"dhdt.min_points": 20, "filttrack.window": 31}
    """
    new = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        obj = new
        parts = dotted.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise AttributeError(f"parâmetro inexistente na config: {dotted}")
        setattr(obj, parts[-1], value)
    return new


def default_param_grid() -> list[dict]:
    """
    Configurações do §3.3 — um parâmetro por vez em torno da base.

    Cada item: {"name": ..., "overrides": {...}, "stage": "filter"|"fit"}
    `stage` indica o passo mais a montante afetado (define o que recalcular).
    """
    g: list[dict] = [{"name": "baseline", "overrides": {}, "stage": "filter"}]

    def add(name, dotted, value, stage):
        g.append({"name": name, "overrides": {dotted: value}, "stage": stage})

    # janela da mediana ao longo da trilha
    add("filttrack_window_11", "filttrack.window", 11, "filter")
    add("filttrack_window_41", "filttrack.window", 41, "filter")
    # multiplicador da janela de MAD
    add("filttrack_madfac_3", "filttrack.mad_window_factor", 3, "filter")
    add("filttrack_madfac_9", "filttrack.mad_window_factor", 9, "filter")
    # limiar robusto (nº de desvios) no ajuste
    add("fit_nsigma_2", "dhdt.n_sigma", 2.0, "fit")
    add("fit_nsigma_4", "dhdt.n_sigma", 4.0, "fit")
    # limite absoluto de resíduo
    add("fit_residlim_3", "dhdt.resid_limit", 3.0, "fit")
    add("fit_residlim_10", "dhdt.resid_limit", 10.0, "fit")
    # nº mínimo de observações por nó
    add("min_points_15", "dhdt.min_points", 15, "fit")
    add("min_points_60", "dhdt.min_points", 60, "fit")
    # raio de busca
    add("radius_10km", "dhdt.search_radius_m", 10_000.0, "fit")
    add("radius_22km", "dhdt.search_radius_m", 22_000.0, "fit")
    # filtro espaço-temporal ligado
    add("filtst_on", "filtst.enabled", True, "filter")
    return g


# ------------------------------------------------------------- sub-regiões
def load_region_points(src, regions: list[dict], columns: list[str],
                       halo_m: float, batch: int = 2_000_000) -> pd.DataFrame:
    """
    Carrega SOMENTE os pontos dentro das sub-regiões (+halo), em streaming.

    Sem isso, a sensibilidade carregaria os ~20 M de pontos e cada filtro faria
    uma cópia — o padrão que já esgotou a RAM neste projeto duas vezes. Aqui a
    memória fica proporcional ao subconjunto, não ao domínio.
    """
    import pyarrow.parquet as pq

    logger = get_logger()
    pf = pq.ParquetFile(src)
    boxes = [(r["x_min"] - halo_m, r["x_max"] + halo_m,
              r["y_min"] - halo_m, r["y_max"] + halo_m) for r in regions]
    keep = []
    n_seen = 0
    for b in pf.iter_batches(batch_size=batch, columns=columns):
        df = b.to_pandas()
        n_seen += len(df)
        x = df["x"].to_numpy(); y = df["y"].to_numpy()
        m = np.zeros(len(df), dtype=bool)
        for (x0, x1, y0, y1) in boxes:
            m |= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        if m.any():
            keep.append(df.loc[m])
    out = (pd.concat(keep, ignore_index=True) if keep
           else pd.DataFrame(columns=columns))
    logger.info(f"sensibilidade: {len(out):,} pontos carregados das sub-regiões "
                f"(de {n_seen:,} no arquivo — {100*len(out)/max(n_seen,1):.1f}%)")
    return out


def select_regions_streaming(src, cfg: Config, batch: int = 2_000_000) -> list[dict]:
    """
    Escolhe as sub-regiões lendo só x, y e mask_class em streaming (leve),
    sem materializar a tabela inteira.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(src)
    names = pf.schema_arrow.names
    cols = [c for c in ("x", "y", "mask_class") if c in names]
    parts = [b.to_pandas() for b in pf.iter_batches(batch_size=batch, columns=cols)]
    light = pd.concat(parts, ignore_index=True)
    return select_regions(light, cfg)


def select_regions(points: pd.DataFrame, cfg: Config) -> list[dict]:
    """
    Escolhe sub-regiões representativas cobrindo regimes distintos.

    Estratifica por classe de máscara (aterrado/flutuante) e densidade, para não
    concluir sobre a geleira inteira a partir de um único regime.
    """
    logger = get_logger()
    s = cfg.sensitivity
    size = s.region_km * 1000.0
    x, y = points["x"].to_numpy(), points["y"].to_numpy()

    ix = np.floor(x / size).astype(np.int64)
    iy = np.floor(y / size).astype(np.int64)
    key = pd.Series(list(zip(ix.tolist(), iy.tolist())))
    grp = pd.DataFrame({"key": key,
                        "mask": points["mask_class"].to_numpy()
                        if "mask_class" in points.columns else 2})
    agg = grp.groupby("key").agg(n=("mask", "size"),
                                 frac_float=("mask", lambda v: float(np.mean(v == 3))))
    agg = agg[agg["n"] >= 20_000]          # densidade mínima p/ ajuste
    if agg.empty:
        raise ValueError("nenhuma sub-região com pontos suficientes.")

    # metade com mais gelo flutuante, metade aterrado; dentro de cada, as mais densas
    floaty = agg[agg["frac_float"] > 0.2].nlargest(s.n_regions // 2 or 1, "n")
    ground = agg[agg["frac_float"] <= 0.2].nlargest(s.n_regions - len(floaty), "n")
    chosen = pd.concat([floaty, ground])

    regions = []
    for (cx, cy), row in chosen.iterrows():
        regions.append({
            "id": f"r{cx}_{cy}",
            "x_min": cx * size, "x_max": (cx + 1) * size,
            "y_min": cy * size, "y_max": (cy + 1) * size,
            "n_points": int(row["n"]), "frac_floating": float(row["frac_float"]),
        })
    logger.info(f"sensibilidade: {len(regions)} sub-regiões de {s.region_km:.0f} km "
                f"({sum(r['n_points'] for r in regions):,} pontos)")
    return regions


# ------------------------------------------------------ execução de 1 config
def run_single_config(points: pd.DataFrame, cfg_run: Config, regions: list[dict],
                      halo_m: float | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Aplica filtros (filttrack, filtst) e calcula dh/dt nas sub-regiões.

    Retorna (nós, estatísticas de retenção por etapa).
    """
    from thwaites.qc.filttrack import filter_along_track
    from thwaites.qc.filtst import filter_space_time
    from thwaites.timeseries.dhdt import compute_tile_dhdt

    n0 = len(points)
    df = points
    if cfg_run.filttrack.enabled:
        df = filter_along_track(df, cfg_run)
    n_after_track = len(df)
    if cfg_run.filtst.enabled:
        df = filter_space_time(df, cfg_run)
    n_after_st = len(df)

    halo = halo_m if halo_m is not None else cfg_run.dhdt.search_radius_m
    nodes = []
    for r in regions:
        sel = ((df["x"] >= r["x_min"] - halo) & (df["x"] < r["x_max"] + halo) &
               (df["y"] >= r["y_min"] - halo) & (df["y"] < r["y_max"] + halo))
        sub = df.loc[sel]
        if len(sub) < cfg_run.dhdt.min_points:
            continue
        nd = compute_tile_dhdt(sub, cfg_run, r["x_min"], r["x_max"],
                               r["y_min"], r["y_max"])
        if len(nd):
            nd = nd.copy()
            nd["region"] = r["id"]
            nodes.append(nd)

    nodes_df = (pd.concat(nodes, ignore_index=True) if nodes
                else pd.DataFrame(columns=["x", "y", "dhdt", "rmse", "nobs", "region"]))
    stats = {
        "n_points_in": n0,
        "n_after_filttrack": n_after_track,
        "n_after_filtst": n_after_st,
        "frac_retained": n_after_st / n0 if n0 else np.nan,
        "rejected_filttrack": n0 - n_after_track,
        "rejected_filtst": n_after_track - n_after_st,
        "n_nodes": int(len(nodes_df)),
        "s_elv_median": float(np.nanmedian(df["s_elv"])) if "s_elv" in df else None,
    }
    if len(nodes_df):
        stats.update({
            "dhdt_median": float(nodes_df["dhdt"].median()),
            "dhdt_mad": float(1.4826 * np.median(np.abs(
                nodes_df["dhdt"] - nodes_df["dhdt"].median()))),
            "rmse_median": float(nodes_df["rmse"].median()),
            "nobs_median": float(nodes_df["nobs"].median()),
        })
    return nodes_df, stats


# --------------------------------------------------------------- comparação
def _bootstrap_median_ci(v: np.ndarray, iters: int, seed: int, alpha=0.05):
    """IC bootstrap da mediana (percentil)."""
    v = v[np.isfinite(v)]
    if v.size < 3:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(iters, v.size), replace=True), axis=1)
    return (float(np.median(v)),
            float(np.percentile(meds, 100 * alpha / 2)),
            float(np.percentile(meds, 100 * (1 - alpha / 2))))


def compare_to_baseline(nodes_base: pd.DataFrame, nodes_test: pd.DataFrame,
                        cfg: Config) -> dict:
    """
    Diferenças PAREADAS por nó (mesmo x,y), com IC bootstrap (§3.4).

    Compara só nós presentes nas duas configurações — evita confundir mudança
    de valor com mudança de cobertura (esta é reportada à parte).
    """
    s = cfg.sensitivity
    if nodes_base.empty or nodes_test.empty:
        return {"n_paired": 0}

    key = ["x", "y"]
    m = nodes_base[key + ["dhdt", "rmse"]].merge(
        nodes_test[key + ["dhdt", "rmse"]], on=key, suffixes=("_base", "_test"))
    if m.empty:
        return {"n_paired": 0}

    d = (m["dhdt_test"] - m["dhdt_base"]).to_numpy()
    med, lo, hi = _bootstrap_median_ci(d, s.bootstrap_iters, s.seed)
    dr = (m["rmse_test"] - m["rmse_base"]).to_numpy()

    only_base = len(nodes_base) - len(m)
    only_test = len(nodes_test) - len(m)
    return {
        "n_paired": int(len(m)),
        "nodes_only_baseline": int(only_base),
        "nodes_only_test": int(only_test),
        "coverage_change": float((len(nodes_test) - len(nodes_base)) /
                                 max(len(nodes_base), 1)),
        "dhdt_diff_median": med,
        "dhdt_diff_ci95": [lo, hi],
        "dhdt_diff_mad": float(1.4826 * np.median(np.abs(d - np.median(d)))),
        "dhdt_diff_p95_abs": float(np.nanpercentile(np.abs(d), 95)),
        "rmse_diff_median": float(np.nanmedian(dr)),
        # o IC cruza zero? -> diferença não distinguível de zero
        "diff_significant": bool(not (lo <= 0.0 <= hi)),
    }


def evaluate_acceptance(comparison: dict, base_stats: dict, test_stats: dict,
                        cfg: Config) -> dict:
    """
    Aplica os critérios de aceite PRÉ-DEFINIDOS (§3.5).

    Uma configuração só "passa" se não deslocar materialmente o dh/dt, não
    degradar o resíduo além do limite e (se for para substituir a base) trouxer
    ganho material de cobertura.
    """
    s = cfg.sensitivity
    shift = abs(comparison.get("dhdt_diff_median", np.nan))
    cov = comparison.get("coverage_change", 0.0)
    r_base = base_stats.get("rmse_median", np.nan)
    r_test = test_stats.get("rmse_median", np.nan)
    r_inc = ((r_test - r_base) / r_base) if (r_base and np.isfinite(r_base)) else np.nan

    checks = {
        "dhdt_shift_ok": bool(np.isfinite(shift) and shift <= s.max_median_dhdt_shift),
        "residual_ok": bool(not np.isfinite(r_inc) or r_inc <= s.max_residual_increase),
        "coverage_material_gain": bool(cov >= s.min_coverage_gain),
    }
    checks["passes"] = bool(checks["dhdt_shift_ok"] and checks["residual_ok"])
    checks["would_replace_baseline"] = bool(checks["passes"] and
                                            checks["coverage_material_gain"])
    checks["measured"] = {"dhdt_shift": shift, "coverage_change": cov,
                          "residual_increase_frac": r_inc}
    checks["thresholds"] = {
        "max_median_dhdt_shift": s.max_median_dhdt_shift,
        "max_residual_increase": s.max_residual_increase,
        "min_coverage_gain": s.min_coverage_gain,
    }
    return checks
