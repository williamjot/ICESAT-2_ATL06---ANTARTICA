"""
thwaites.validation.agreement
=============================
Prioridade 5 (§6): concordância ESPACIAL entre o ajuste local (fitsec) e os
crossovers.

A pergunta (§6.1): o campo de dh/dt do ajuste local concorda espacialmente com
os crossovers, **ou a concordância das medianas globais esconde divergências
regionais?**

Isso importa porque uma concordância global boa (ex.: −0,477 vs −0,504 m/ano)
é compatível com dois campos que discordam bastante em setores opostos, com os
erros se cancelando na mediana.

RESSALVA OBRIGATÓRIA (§6.5): crossovers **NÃO são validação totalmente
independente** — usam o MESMO produto altimétrico (ATL06), a mesma máscara e as
mesmas correções. São um estimador *metodologicamente* alternativo (geometria
de cruzamento em vez de ajuste de superfície), o que é valioso, mas erros
comuns ao produto de origem não são detectados por eles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger

INDEPENDENCE_CAVEAT = (
    "Crossovers NÃO constituem validação independente: usam o mesmo produto "
    "altimétrico (ATL06), a mesma máscara e as mesmas correções geofísicas. "
    "São um estimador metodologicamente alternativo, não uma fonte externa.")


def match_xovers_to_nodes(xovers: pd.DataFrame, nodes: pd.DataFrame,
                          max_dist_m: float = 5000.0) -> pd.DataFrame:
    """
    Associa cada crossover ao nó de dh/dt mais próximo (§6.4, passo 2).

    Crossovers a mais de `max_dist_m` de qualquer nó são descartados — comparar
    com um nó distante mediria diferença espacial, não discordância de método.
    """
    from scipy.spatial import cKDTree

    v = xovers[np.isfinite(xovers["dhdt"])].copy()
    if v.empty or nodes.empty:
        return pd.DataFrame()
    tree = cKDTree(np.c_[nodes["x"].to_numpy(), nodes["y"].to_numpy()])
    d, idx = tree.query(np.c_[v["x"].to_numpy(), v["y"].to_numpy()], k=1)
    keep = d <= max_dist_m
    v = v.loc[keep].copy()
    idx = idx[keep]
    v["dist_to_node_m"] = d[keep]
    for col in ("dhdt", "dhdt_err", "nobs", "rmse"):
        if col in nodes.columns:
            v[f"local_{col}"] = nodes[col].to_numpy()[idx]
    v["node_x"] = nodes["x"].to_numpy()[idx]
    v["node_y"] = nodes["y"].to_numpy()[idx]
    return v


def paired_differences(matched: pd.DataFrame) -> pd.DataFrame:
    """
    Diferença pareada e sua NORMALIZAÇÃO pela incerteza combinada (§6.4, 3–4).

    A diferença normalizada z = Δ/σ_comb é o que permite dizer se a discordância
    é compatível com as incertezas declaradas — |z| tipicamente ≫1 significa que
    ao menos um dos dois estimadores tem incerteza subestimada.
    """
    m = matched.copy()
    m["diff"] = m["dhdt"] - m["local_dhdt"]
    e1 = m.get("dhdt_err")
    e2 = m.get("local_dhdt_err")
    if e1 is not None and e2 is not None:
        comb = np.sqrt(np.nan_to_num(e1.to_numpy(), nan=np.nan) ** 2 +
                       np.nan_to_num(e2.to_numpy(), nan=np.nan) ** 2)
        m["sigma_combined"] = comb
        with np.errstate(invalid="ignore", divide="ignore"):
            m["z"] = np.where(comb > 0, m["diff"].to_numpy() / comb, np.nan)
    else:
        m["sigma_combined"] = np.nan
        m["z"] = np.nan
    return m


def robust_regression(x, y) -> dict:
    """
    Regressão robusta (Theil-Sen) entre os dois estimadores (§6.4).

    Inclinação ≈1 e intercepto ≈0 indicam consistência; inclinação ≠1 indica
    diferença sistemática de escala entre os métodos.
    """
    from scipy.stats import theilslopes, pearsonr, spearmanr

    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return {"n": int(ok.sum())}
    slope, intercept, lo, hi = theilslopes(y[ok], x[ok], alpha=0.95)
    return {
        "n": int(ok.sum()),
        "slope": float(slope), "slope_ci": [float(lo), float(hi)],
        "intercept": float(intercept),
        "pearson_r": float(pearsonr(x[ok], y[ok])[0]),
        "spearman_r": float(spearmanr(x[ok], y[ok])[0]),
        "slope_consistent_with_1": bool(lo <= 1.0 <= hi),
    }


def spatial_structure_of_differences(m: pd.DataFrame, cfg: Config | None = None,
                                     n_lags: int = 12,
                                     max_lag_m: float = 100_000.0) -> dict:
    """
    Autocorrelação espacial das DIFERENÇAS (§6.4).

    Se as diferenças fossem ruído, o variograma seria plano (efeito pepita puro).
    Estrutura espacial significa discordância SISTEMÁTICA por região — que é
    exatamente o que a mediana global esconde.
    """
    from thwaites.interp.variogram import empirical_variogram

    d = m["diff"].to_numpy()
    ok = np.isfinite(d)
    if ok.sum() < 50:
        return {"status": "pontos insuficientes"}
    lags, gamma, counts = empirical_variogram(
        m["x"].to_numpy()[ok], m["y"].to_numpy()[ok], d[ok],
        n_lags=n_lags, max_lag=max_lag_m, seed=0)
    if len(gamma) < 3:
        return {"status": "variograma insuficiente"}
    nugget = float(gamma[0])
    sill = float(np.nanmax(gamma))
    ratio = float(nugget / sill) if sill > 0 else np.nan
    return {
        "status": "ok",
        "lags_m": [float(v) for v in lags],
        "gamma": [float(v) for v in gamma],
        "nugget": nugget, "sill": sill,
        "nugget_to_sill": ratio,
        # razão baixa => forte estrutura espacial => discordância regional
        "spatially_structured": bool(np.isfinite(ratio) and ratio < 0.7),
    }


def find_hotspots(m: pd.DataFrame, cell_km: float = 25.0,
                  min_count: int = 10, z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Setores com discordância persistente (§6.4/§6.5).

    Agrega por célula e sinaliza células cuja diferença MEDIANA é grande em
    relação à incerteza — persistência, não um ponto isolado.
    """
    if m.empty:
        return pd.DataFrame()
    size = cell_km * 1000.0
    ix = np.floor(m["x"].to_numpy() / size).astype(int)
    iy = np.floor(m["y"].to_numpy() / size).astype(int)
    g = m.assign(_ix=ix, _iy=iy).groupby(["_ix", "_iy"])
    agg = g.agg(n=("diff", "size"),
                median_diff=("diff", "median"),
                mad_diff=("diff", lambda s: 1.4826 * np.median(np.abs(s - s.median()))),
                median_z=("z", "median"),
                x=("x", "mean"), y=("y", "mean")).reset_index(drop=True)
    agg = agg[agg["n"] >= min_count]
    agg["hotspot"] = agg["median_z"].abs() >= z_threshold
    return agg.sort_values("median_z", key=np.abs, ascending=False)


def assess_agreement(xovers: pd.DataFrame, nodes: pd.DataFrame,
                     cfg: Config | None = None, max_dist_m: float = 5000.0):
    """
    Análise completa de concordância (§6.4), global E espacial.

    Retorna a tupla (resumo, pareados, hotspots) — os dois últimos servem para
    mapear a discordância, que é o produto que o §6.6 pede.
    """
    logger = get_logger()
    matched = match_xovers_to_nodes(xovers, nodes, max_dist_m)
    if matched.empty:
        return ({"status": "nenhum crossover pareado",
                 "caveat": INDEPENDENCE_CAVEAT}, pd.DataFrame(), pd.DataFrame())
    m = paired_differences(matched)
    d = m["diff"].to_numpy()
    fin = np.isfinite(d)

    res = {
        "status": "ok",
        "caveat": INDEPENDENCE_CAVEAT,
        "n_matched": int(fin.sum()),
        "max_dist_m": max_dist_m,
        # --- global -------------------------------------------------------
        "median_xover": float(np.nanmedian(m["dhdt"])),
        "median_local": float(np.nanmedian(m["local_dhdt"])),
        "median_diff": float(np.nanmedian(d[fin])),
        "mad_diff": float(1.4826 * np.median(np.abs(d[fin] - np.median(d[fin])))),
        "rmse_diff": float(np.sqrt(np.nanmean(d[fin] ** 2))),
        # --- normalização pela incerteza combinada -------------------------
        "median_abs_z": float(np.nanmedian(np.abs(m["z"]))) if m["z"].notna().any() else np.nan,
        "frac_within_2sigma": (float(np.nanmean(np.abs(m["z"]) <= 2.0))
                               if m["z"].notna().any() else np.nan),
        # --- relação entre estimadores -------------------------------------
        "regression": robust_regression(m["local_dhdt"], m["dhdt"]),
        # --- estrutura espacial (o ponto do §6.1) --------------------------
        "spatial": spatial_structure_of_differences(m, cfg),
    }
    hs = find_hotspots(m)
    res["n_cells"] = int(len(hs))
    res["n_hotspots"] = int(hs["hotspot"].sum()) if not hs.empty else 0

    # dependência com o intervalo temporal do crossover (§6.4)
    if "dt" in m.columns:
        q = pd.qcut(m["dt"].abs(), 4, labels=False, duplicates="drop")
        res["diff_by_dt_quartile"] = {
            int(k): float(np.nanmedian(d[(q == k).to_numpy() & fin]))
            for k in np.unique(q.dropna())}

    logger.info(
        f"concordância: n={res['n_matched']:,} | Δmediana "
        f"{res['median_diff']:+.4f} m/ano | MAD {res['mad_diff']:.4f} | "
        f"|z| mediano {res['median_abs_z']:.2f} | "
        f"hotspots {res['n_hotspots']}/{res['n_cells']}")
    if res["spatial"].get("spatially_structured"):
        logger.warning(
            "As diferenças têm ESTRUTURA ESPACIAL — a concordância das medianas "
            "globais esconde discordância regional (§6.1). Investigar antes do "
            "mapa final.")
    return res, m, hs


def subsampling_sensitivity(points: pd.DataFrame, cfg: Config,
                            subsamples=(5, 10, 20), seeds=(0, 1, 2)) -> pd.DataFrame:
    """
    Sensibilidade dos crossovers à subamostragem da geometria e à semente
    (§6.3/§6.5): o resultado precisa ser estável.
    """
    from thwaites.qc.xover import find_crossovers
    import copy

    rows = []
    for sub in subsamples:
        for seed in seeds:
            c = copy.deepcopy(cfg)
            c.xover.subsample = sub
            c.xover.seed = seed
            xo = find_crossovers(points, c)
            v = xo["dhdt"].dropna()
            rows.append({"subsample": sub, "seed": seed, "n_xovers": int(len(xo)),
                         "n_valid": int(len(v)),
                         "median_dhdt": float(v.median()) if len(v) else np.nan,
                         "mad_dhdt": (float(1.4826 * np.median(np.abs(v - v.median())))
                                      if len(v) else np.nan)})
    return pd.DataFrame(rows)
