"""
thwaites.timeseries.trend
=========================
Teste de tendência FORMAL por nó sobre a série temporal de elevação:

  - Mann-Kendall (não-paramétrico, robusto a outliers e à assimetria já
    observada; não exige normalidade) -> existe tendência monotônica? (tau, p)
  - Sen's slope (Theil-Sen) -> magnitude da tendência (m/ano) com IC, robusta
    e calculada sobre os ANOS reais (lida com anos faltantes).
  - Correção para testes múltiplos (FDR, Benjamini-Hochberg) entre os milhares
    de nós — sem ela, ~5% dos nós dariam "significativo" por acaso.

Implementação com SciPy (kendalltau, theilslopes) + statsmodels (FDR); nenhuma
dependência nova. As variantes SAZONAIS / com autocorrelação (Hamed-Rao,
seasonal MK) do ciclo anual (Fase 7) exigirão `pymannkendall` — sinalizado
explicitamente aqui, não silenciosamente aproximado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes, norm

from thwaites.config import Config
from thwaites.logging import get_logger

TREND_COLUMNS = ["node_x", "node_y", "lon", "lat", "n_epochs",
                 "sens_slope", "sens_lo", "sens_hi", "intercept",
                 "tau", "p_value", "p_fdr", "significant", "trend"]


def mann_kendall_sen(years, values, alpha: float = 0.05) -> dict:
    """
    Mann-Kendall + Sen's slope numa série (years, values).

    Retorna dict com tau, p_value, sens_slope (m/ano) e IC, intercept, trend.
    Requer >= 3 pontos; caso contrário retorna NaN.
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(years)
    years, values = years[order], values[order]
    n = len(values)
    if n < 3:
        return {"tau": np.nan, "p_value": np.nan, "sens_slope": np.nan,
                "sens_lo": np.nan, "sens_hi": np.nan, "intercept": np.nan,
                "trend": "insuficiente", "n": n}

    tau, p = kendalltau(years, values)
    slope, intercept, lo, hi = theilslopes(values, years, alpha=1 - alpha)

    if np.isnan(p):
        trend = "insuficiente"
    elif p < alpha and slope < 0:
        trend = "decrescente"
    elif p < alpha and slope > 0:
        trend = "crescente"
    else:
        trend = "sem_tendencia"

    return {"tau": float(tau), "p_value": float(p), "sens_slope": float(slope),
            "sens_lo": float(lo), "sens_hi": float(hi), "intercept": float(intercept),
            "trend": trend, "n": n}


def seasonal_mann_kendall_sen(years, months, values, alpha: float = 0.05) -> dict:
    """Mann-Kendall e Sen sazonal com estratos mensais e lacunas permitidas.

    Comparações são feitas apenas entre o mesmo mês de anos distintos. Isso
    remove o ciclo anual sem usar pares janeiro-versus-julho como se fossem
    amostras da mesma distribuição. O IC do declive usa postos das inclinações
    intra-mês e a variância sazonal total de S.
    """
    years = np.asarray(years, dtype=float)
    months = np.asarray(months, dtype=int)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(years) & np.isfinite(values) & (months >= 1) & (months <= 12)
    years, months, values = years[ok], months[ok], values[ok]
    if len(values) < 4:
        return {"tau": np.nan, "p_value": np.nan, "sens_slope": np.nan,
                "sens_lo": np.nan, "sens_hi": np.nan, "intercept": np.nan,
                "trend": "insuficiente", "n": len(values)}

    s_total = var_total = 0.0
    pair_count = n_used = 0
    slopes: list[float] = []
    for mon in range(1, 13):
        sel = months == mon
        yy, vv = years[sel], values[sel]
        order = np.argsort(yy)
        yy, vv = yy[order], vv[order]
        n = len(vv)
        if n < 2:
            continue
        n_used += n
        diff = vv[None, :] - vv[:, None]
        iu = np.triu_indices(n, k=1)
        d = diff[iu]
        s_total += float(np.sign(d).sum())
        pair_count += len(d)
        year_diff = (yy[None, :] - yy[:, None])[iu]
        slopes.extend((d / year_diff).tolist())
        _, ties = np.unique(vv, return_counts=True)
        var_total += (n * (n - 1) * (2 * n + 5)
                      - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0

    if pair_count == 0 or var_total <= 0 or not slopes:
        return {"tau": np.nan, "p_value": np.nan, "sens_slope": np.nan,
                "sens_lo": np.nan, "sens_hi": np.nan, "intercept": np.nan,
                "trend": "insuficiente", "n": n_used}

    z = ((s_total - 1.0) / np.sqrt(var_total) if s_total > 0 else
         (s_total + 1.0) / np.sqrt(var_total) if s_total < 0 else 0.0)
    p = float(2.0 * norm.sf(abs(z)))
    slopes_arr = np.sort(np.asarray(slopes, dtype=float))
    slope = float(np.median(slopes_arr))
    c_alpha = norm.ppf(1.0 - alpha / 2.0) * np.sqrt(var_total)
    lo_idx = max(0, int(np.floor((len(slopes_arr) - c_alpha) / 2.0)))
    hi_idx = min(len(slopes_arr) - 1,
                 int(np.ceil((len(slopes_arr) + c_alpha) / 2.0)))
    intercept = float(np.median(values - slope * years))
    trend = ("decrescente" if p < alpha and slope < 0 else
             "crescente" if p < alpha and slope > 0 else "sem_tendencia")
    return {"tau": float(s_total / pair_count), "p_value": p,
            "sens_slope": slope, "sens_lo": float(slopes_arr[lo_idx]),
            "sens_hi": float(slopes_arr[hi_idx]), "intercept": intercept,
            "trend": trend, "n": int(n_used)}


def compute_trends(series_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Aplica Mann-Kendall + Sen por nó e corrige os p-valores por FDR entre nós.
    `series_df` no formato longo de thwaites.timeseries.build.
    """
    from statsmodels.stats.multitest import multipletests

    logger = get_logger()
    if cfg.trend.mk_variant not in ("original", "auto", "seasonal"):
        raise ValueError(f"mk_variant inválido: {cfg.trend.mk_variant!r}")
    seasonal = cfg.trend.mk_variant == "seasonal"
    if seasonal and "month" not in series_df.columns:
        raise ValueError("MK sazonal exige coluna 'month'; reconstrua node_series com perfil anual.")

    alpha = cfg.trend.alpha
    min_epochs = cfg.timeseries.min_epochs

    rows = []
    for (nx, ny), g in series_df.groupby(["node_x", "node_y"], sort=False):
        if g["year"].nunique() < min_epochs:
            continue
        r = (seasonal_mann_kendall_sen(g["year"].to_numpy(), g["month"].to_numpy(),
                                       g["h_node"].to_numpy(), alpha)
             if seasonal else
             mann_kendall_sen(g["year"].to_numpy(), g["h_node"].to_numpy(), alpha))
        if r["trend"] == "insuficiente":
            continue
        rows.append({
            "node_x": nx, "node_y": ny,
            "lon": float(g["lon"].iloc[0]), "lat": float(g["lat"].iloc[0]),
            "n_epochs": r["n"], "sens_slope": r["sens_slope"],
            "sens_lo": r["sens_lo"], "sens_hi": r["sens_hi"],
            "intercept": r["intercept"], "tau": r["tau"], "p_value": r["p_value"],
        })

    if not rows:
        logger.warning("Nenhum nó com épocas suficientes para tendência.")
        return pd.DataFrame({c: [] for c in TREND_COLUMNS})

    out = pd.DataFrame(rows)
    # FDR (Benjamini-Hochberg) entre todos os nós testados.
    reject, p_fdr, _, _ = multipletests(out["p_value"].to_numpy(),
                                        alpha=alpha, method="fdr_bh")
    out["p_fdr"] = p_fdr
    out["significant"] = reject
    out["trend"] = np.where(
        ~reject, "sem_tendencia",
        np.where(out["sens_slope"] < 0, "decrescente", "crescente"))
    n_sig = int(reject.sum())
    logger.info(f"Tendência: {len(out):,} nós testados, {n_sig:,} significativos "
                f"(FDR α={alpha})")
    return out[TREND_COLUMNS]
