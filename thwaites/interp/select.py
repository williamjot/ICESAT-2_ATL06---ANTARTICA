"""
thwaites.interp.select
======================
Seleção OBJETIVA do interpolador por validação cruzada com BLOCOS ESPACIAIS.

Por que blocos, e não CV aleatória: pontos vizinhos são espacialmente
correlacionados; retirar pontos ao acaso deixa vizinhos quase colados entre
treino e teste, vazando informação e superestimando a qualidade. Retirar
BLOCOS inteiros mede a real capacidade de extrapolar para regiões sem dado.

Métricas por candidato: RMSE, MAE, viés e CALIBRAÇÃO da incerteza (fração de
resíduos padronizados |z|<1, ideal ≈0,68; e desvio de z, ideal ≈1). A
incerteza bem calibrada importa tanto quanto o valor para a propagação
dh/dt → Gt/ano (Fase 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.interp.methods import PREDICTORS
from thwaites.interp.variogram import fit_variogram
from thwaites.logging import get_logger

# métodos que precisam do variograma: krigagem/OI o usam diretamente; os
# kernels o usam para derivar sua escala (σ, raio) a partir dos dados.
_NEEDS_VARIOGRAM = {"ordinary_kriging", "oi_markov", "gaussian_kernel", "median_kernel"}


def spatial_block_folds(x, y, block_m, n_folds, seed=0) -> np.ndarray:
    """Atribui cada ponto a um fold, agrupando por blocos espaciais inteiros."""
    x = np.asarray(x); y = np.asarray(y)
    bi = np.floor(x / block_m).astype(int)
    bj = np.floor(y / block_m).astype(int)
    keys = bi * 100003 + bj                      # id de bloco
    uniq = np.unique(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    block_fold = {b: (i % n_folds) for i, b in enumerate(uniq)}
    return np.array([block_fold[k] for k in keys], dtype=int)


def _metrics(actual, pred, var):
    res = pred - actual
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    bias = float(np.mean(res))
    v = np.asarray(var, float)
    ok = v > 0
    if ok.sum() > 1:
        z = res[ok] / np.sqrt(v[ok])
        cal_frac = float(np.mean(np.abs(z) < 1.0))
        cal_zstd = float(np.std(z))
    else:
        cal_frac = cal_zstd = np.nan
    return rmse, mae, bias, cal_frac, cal_zstd


def cross_validate(x, y, v, sig, cfg: Config) -> tuple[pd.DataFrame, str]:
    """
    Roda a CV por blocos para todos os candidatos. Retorna (tabela, vencedor).
    Vencedor = menor RMSE (empate técnico desempata por calibração mais próxima de 0,68).
    """
    logger = get_logger()
    x = np.asarray(x, float); y = np.asarray(y, float)
    v = np.asarray(v, float); sig = np.asarray(sig, float)
    ic = cfg.interpolation
    fold = spatial_block_folds(x, y, ic.cv.block_km * 1000.0, ic.cv.n_folds, ic.cv.seed)

    rows = []
    for name in ic.candidates:
        if name not in PREDICTORS:
            logger.warning(f"candidato desconhecido ignorado: {name}")
            continue
        preds, actuals, varis = [], [], []
        for f in range(ic.cv.n_folds):
            test = fold == f
            train = ~test
            if test.sum() == 0 or train.sum() < max(ic.neighbors, 10):
                continue
            vparams = None
            if name in _NEEDS_VARIOGRAM:
                try:
                    vparams = fit_variogram(
                        x[train], y[train], v[train],
                        n_lags=ic.variogram.n_lags, max_lag=ic.variogram.max_lag_m,
                        seed=ic.cv.seed)
                except Exception as e:
                    logger.warning(f"{name}: variograma falhou no fold {f} ({e}) — pulado.")
                    continue
            p, va = PREDICTORS[name](x[train], y[train], v[train], sig[train],
                                     x[test], y[test], cfg, vparams)
            preds.append(p); actuals.append(v[test]); varis.append(va)

        if not preds:
            logger.warning(f"{name}: sem folds válidos.")
            continue
        rmse, mae, bias, cal_frac, cal_zstd = _metrics(
            np.concatenate(actuals), np.concatenate(preds), np.concatenate(varis))
        rows.append({"method": name, "rmse": rmse, "mae": mae, "bias": bias,
                     "cal_frac_1sigma": cal_frac, "cal_zstd": cal_zstd,
                     "n_pred": int(sum(len(p) for p in preds))})

    if not rows:
        raise RuntimeError("Nenhum candidato produziu predições na CV.")

    table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    # desempate: se RMSE do 2º está a <2% do 1º, prefere calibração melhor
    winner = table.iloc[0]["method"]
    if len(table) > 1 and table.iloc[1]["rmse"] <= table.iloc[0]["rmse"] * 1.02:
        top = table.head(2).copy()
        top["cal_dev"] = (top["cal_frac_1sigma"] - 0.6827).abs()
        winner = top.sort_values("cal_dev").iloc[0]["method"]
    logger.info(f"Seleção de interpolador — vencedor: {winner}\n{table.to_string(index=False)}")
    return table, winner


def select_interpolator(nodes_df: pd.DataFrame, cfg: Config,
                        value_col="dhdt", sigma_col="dhdt_err") -> dict:
    """
    Executa a seleção sobre os nós de dh/dt e ajusta o variograma global.
    Retorna dict {metrics, winner, variogram}.
    """
    x = nodes_df["x"].to_numpy(); y = nodes_df["y"].to_numpy()
    v = nodes_df[value_col].to_numpy()
    if sigma_col in nodes_df.columns:
        sig = nodes_df[sigma_col].to_numpy()
        med = np.nanmedian(sig[sig > 0]) if np.any(sig > 0) else 1.0
        sig = np.where(np.isnan(sig) | (sig <= 0), med, sig)
    else:
        sig = np.ones_like(v)

    table, winner = cross_validate(x, y, v, sig, cfg)
    vgram = fit_variogram(x, y, v, n_lags=cfg.interpolation.variogram.n_lags,
                          max_lag=cfg.interpolation.variogram.max_lag_m,
                          seed=cfg.interpolation.cv.seed)
    return {"metrics": table, "winner": winner, "variogram": vgram}


def interpolate_to_grid(nodes_df: pd.DataFrame, cfg: Config, method: str,
                        vparams: dict | None = None,
                        value_col="dhdt", sigma_col="dhdt_err") -> pd.DataFrame:
    """
    Grade regular (EPSG:3031) do campo interpolado pelo `method` escolhido.
    Retorna DataFrame de células (x, y, lon, lat, pred, var).
    """
    from thwaites.grid.reproject import to_lonlat

    x = nodes_df["x"].to_numpy(); y = nodes_df["y"].to_numpy()
    v = nodes_df[value_col].to_numpy()
    sig = (nodes_df[sigma_col].to_numpy() if sigma_col in nodes_df.columns
           else np.ones_like(v))

    if method in _NEEDS_VARIOGRAM and vparams is None:
        vparams = fit_variogram(x, y, v, n_lags=cfg.interpolation.variogram.n_lags,
                                max_lag=cfg.interpolation.variogram.max_lag_m,
                                seed=cfg.interpolation.cv.seed)

    res = cfg.interpolation.grid_res_m
    gx = np.arange(x.min(), x.max() + res, res)
    gy = np.arange(y.min(), y.max() + res, res)
    GX, GY = np.meshgrid(gx, gy)
    tx, ty = GX.ravel(), GY.ravel()

    pred, var = PREDICTORS[method](x, y, v, sig, tx, ty, cfg, vparams)

    # --- propagação da INCERTEZA DE ENTRADA (nós) --------------------------
    # Alguns interpoladores (IDW, kernels) derivam a variância apenas do
    # espalhamento dos valores vizinhos e IGNORAM o erro dos nós. Por isso, a
    # incerteza dos nós é propagada pela interpolação do próprio campo de σ com
    # o mesmo método.
    #
    # HIPÓTESE DECLARADA: os erros dos nós vizinhos são tratados como
    # FORTEMENTE CORRELACIONADOS — o que dá σ_entrada = Σwᵢσᵢ (média ponderada)
    # em vez de √(Σwᵢ²σᵢ²). É o limite conservador, e é o realista aqui: nós a
    # 5 km com raio de busca de 15–22 km compartilham a maior parte das
    # observações, então seus erros não são independentes.
    sigma_in, _ = PREDICTORS[method](x, y, sig, sig, tx, ty, cfg, vparams)
    sigma_in = np.abs(np.asarray(sigma_in, dtype=float))

    lon, lat = to_lonlat(tx, ty, cfg)
    return pd.DataFrame({"x": tx, "y": ty, "lon": lon, "lat": lat,
                         "pred": pred,
                         "var_interp": var,          # erro de predição espacial
                         "sigma_input": sigma_in,    # erro herdado dos nós
                         "var": np.asarray(var, dtype=float) + sigma_in ** 2})
