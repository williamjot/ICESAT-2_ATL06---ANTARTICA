"""
thwaites.validation.evaluate
============================
Avaliação sem vazamento (§5.4): ajusta nos dados de TREINO e prevê as
OBSERVAÇÕES retidas.

O alvo da previsão é a elevação de cada observação retida, não um nó já
calculado. Isso é o que torna a validação honesta: o modelo precisa produzir
uma previsão em (x, y, t) sem nunca ter visto aquela observação.

Como se prevê a elevação num ponto/instante arbitrário:

    h(x, y, t) ≈ p0(x, y) + dh/dt(x, y) · (t − t_ref)

onde `p0` (elevação em t_ref) e `dh/dt` vêm dos nós ajustados SOMENTE com
observações de treino e são interpolados até a posição da observação de teste.
A incerteza da previsão combina as variâncias dos dois campos interpolados —
o que permite avaliar a CALIBRAÇÃO (§5.4), e não só o RMSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger
from thwaites.validation.folds import Fold, verify_no_leakage


def _height_column(df: pd.DataFrame) -> str:
    for c in ("h_res", "h_corr", "h_elv"):
        if c in df.columns:
            return c
    raise ValueError("nenhuma coluna de elevação (h_res/h_corr/h_elv)")


def fit_nodes_from_observations(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Ajusta os nós de dh/dt na extensão dos dados fornecidos (só treino)."""
    from thwaites.timeseries.dhdt import compute_tile_dhdt

    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    step = cfg.dhdt.node_spacing_m
    x0 = np.floor(x.min() / step) * step
    x1 = np.ceil(x.max() / step) * step
    y0 = np.floor(y.min() / step) * step
    y1 = np.ceil(y.max() / step) * step
    return compute_tile_dhdt(df, cfg, x0, x1, y0, y1)


def predict_at_observations(nodes: pd.DataFrame, test: pd.DataFrame, cfg: Config,
                            method: str, vparams: dict | None = None):
    """
    Prevê a elevação nas observações de teste a partir dos nós de treino.

    Retorna (h_pred, var_pred, n_neighbors_used).
    """
    from thwaites.interp.methods import PREDICTORS

    if nodes.empty:
        n = len(test)
        return np.full(n, np.nan), np.full(n, np.nan), 0

    nx = nodes["x"].to_numpy(); ny = nodes["y"].to_numpy()
    tx = test["x"].to_numpy(); ty = test["y"].to_numpy()
    sig = (nodes["dhdt_err"].to_numpy() if "dhdt_err" in nodes.columns
           else np.ones(len(nodes)))
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nanmedian(sig[sig > 0])
                   if np.any(sig > 0) else 1.0)

    pred_fn = PREDICTORS[method]
    rate, rate_var = pred_fn(nx, ny, nodes["dhdt"].to_numpy(), sig, tx, ty, cfg, vparams)
    p0, p0_var = pred_fn(nx, ny, nodes["p0"].to_numpy(), sig, tx, ty, cfg, vparams)

    dt = test["t_year"].to_numpy() - cfg.dhdt.t_ref
    h_pred = p0 + rate * dt
    # variância da previsão: soma das contribuições (p0 e taxa são estimados
    # do mesmo ajuste, então isto é uma aproximação — declarada como tal)
    var_pred = np.asarray(p0_var) + (dt ** 2) * np.asarray(rate_var)
    return h_pred, var_pred, len(nodes)


def fold_metrics(resid: np.ndarray, var_pred: np.ndarray,
                 strat: dict | None = None) -> dict:
    """
    Métricas do §5.4, incluindo CALIBRAÇÃO da incerteza.

    RMSE sozinho não basta para escolher interpolador (§5.4): um método pode ter
    RMSE baixo e incerteza mal calibrada, o que estraga qualquer propagação de
    erro a jusante.
    """
    r = np.asarray(resid, float)
    ok = np.isfinite(r)
    r = r[ok]
    if r.size == 0:
        return {"n": 0}
    v = np.asarray(var_pred, float)[ok]

    out = {
        "n": int(r.size),
        "rmse": float(np.sqrt(np.mean(r ** 2))),
        "mae": float(np.mean(np.abs(r))),
        "bias": float(np.mean(r)),
        "median_error": float(np.median(r)),
        "mad": float(1.4826 * np.median(np.abs(r - np.median(r)))),
    }
    good = np.isfinite(v) & (v > 0)
    if good.sum() > 10:
        z = r[good] / np.sqrt(v[good])
        out.update({
            "coverage_68": float(np.mean(np.abs(z) <= 1.0)),
            "coverage_95": float(np.mean(np.abs(z) <= 1.96)),
            "z_std": float(np.std(z)),          # ideal ≈ 1
        })
    else:
        out.update({"coverage_68": np.nan, "coverage_95": np.nan, "z_std": np.nan})

    if strat:
        for name, values in strat.items():
            vals = np.asarray(values)[ok]
            fin = np.isfinite(vals) if vals.dtype.kind == "f" else np.ones(len(vals), bool)
            if fin.sum() < 10:
                continue
            # erro por quartil da variável de estratificação
            try:
                q = pd.qcut(pd.Series(vals[fin]), 4, labels=False, duplicates="drop")
                for k in np.unique(q.dropna()):
                    m = (q == k).to_numpy()
                    out[f"rmse_{name}_q{int(k)+1}"] = float(np.sqrt(np.mean(r[fin][m] ** 2)))
            except Exception:
                pass
    return out


def evaluate_fold(points: pd.DataFrame, fold: Fold, cfg: Config, method: str,
                  vparams: dict | None = None) -> dict:
    """
    Avalia UM fold: ajusta nos dados de treino, prevê as observações de teste.
    """
    verify_no_leakage(fold)                    # §5.5 — verificação absoluta
    hcol = _height_column(points)
    train = points.loc[fold.train]
    test = points.loc[fold.test]
    if len(train) < cfg.dhdt.min_points * 3 or len(test) == 0:
        return {"strategy": fold.strategy, "fold": fold.index, "method": method,
                "n": 0, "status": "dados insuficientes"}

    nodes = fit_nodes_from_observations(train, cfg)
    if nodes.empty:
        return {"strategy": fold.strategy, "fold": fold.index, "method": method,
                "n": 0, "status": "nenhum nó ajustado"}

    h_pred, var_pred, n_nodes = predict_at_observations(nodes, test, cfg, method, vparams)
    resid = test[hcol].to_numpy(float) - h_pred

    # estratificações do §5.4
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[train["x"].to_numpy(), train["y"].to_numpy()])
    dist_train, _ = tree.query(np.c_[test["x"].to_numpy(), test["y"].to_numpy()], k=1)
    strat = {"dist_to_train": dist_train, "year": np.floor(test["t_year"].to_numpy())}
    if "s_elv" in test.columns:
        strat["s_elv"] = test["s_elv"].to_numpy()

    m = fold_metrics(resid, var_pred, strat)
    m.update({"strategy": fold.strategy, "fold": fold.index, "method": method,
              "status": "ok", "n_train": fold.n_train, "n_test": fold.n_test,
              "n_nodes": int(n_nodes),
              "median_dist_to_train_m": float(np.median(dist_train)),
              **{k: v for k, v in fold.info.items() if np.isscalar(v)}})
    return m


def run_validation(points: pd.DataFrame, cfg: Config, folds: list[Fold],
                   methods: list[str], vparams: dict | None = None) -> pd.DataFrame:
    """
    Roda todos os folds × métodos. Retorna tabela longa (uma linha por
    fold×método), preservando a DISPERSÃO entre folds (§5.5) em vez de
    esconder tudo numa média agregada.
    """
    logger = get_logger()
    rows = []
    for fold in folds:
        for method in methods:
            r = evaluate_fold(points, fold, cfg, method, vparams)
            rows.append(r)
            if r.get("status") == "ok":
                logger.info(f"  [{fold.strategy} f{fold.index}] {method}: "
                            f"RMSE {r['rmse']:.3f} | viés {r['bias']:+.3f} | "
                            f"cob68 {r.get('coverage_68', float('nan')):.2f}")
            else:
                logger.warning(f"  [{fold.strategy} f{fold.index}] {method}: "
                               f"{r.get('status')}")
    return pd.DataFrame(rows)


def summarize_by_method(table: pd.DataFrame) -> pd.DataFrame:
    """
    Resumo por (estratégia, método) COM dispersão entre folds (§5.5).

    A dispersão é parte do resultado: um método com RMSE médio bom mas instável
    entre folds não é preferível a um estável.
    """
    ok = table[table.get("status", "ok") == "ok"]
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby(["strategy", "method"])
    out = g.agg(
        n_folds=("rmse", "size"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        bias_mean=("bias", "mean"), bias_std=("bias", "std"),
        cov68_mean=("coverage_68", "mean"),
        cov95_mean=("coverage_95", "mean"),
        zstd_mean=("z_std", "mean"),
    ).reset_index()
    # desvios dos alvos ideais de calibração
    out["cov68_dev"] = (out["cov68_mean"] - 0.6827).abs()
    out["zstd_dev"] = (out["zstd_mean"] - 1.0).abs()
    return out
