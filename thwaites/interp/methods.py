"""
thwaites.interp.methods
=======================
Três interpoladores locais (baseados em k vizinhos, via KDTree — escaláveis),
cada um devolvendo predição E variância de estimativa (para calibrar a
incerteza na seleção por CV):

  - ordinary_kriging_predict : krigagem ordinária com variograma ajustado
  - oi_markov_predict        : Interpolação Ótima com covariância de Markov
  - idw_predict              : inverse distance weighting (linha de base)

Assinatura unificada em PREDICTORS para a validação cruzada (select.py).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _knn(tree, tx, ty, k, n_train):
    k = int(min(k, n_train))
    d, idx = tree.query(np.c_[tx, ty], k=k)
    if k == 1:
        d = d[:, None]; idx = idx[:, None]
    return d, idx, k


def idw_predict(px, py, pv, tx, ty, power=2.0, k=32):
    px, py, pv = map(np.asarray, (px, py, pv))
    tx, ty = np.atleast_1d(tx), np.atleast_1d(ty)
    tree = cKDTree(np.c_[px, py])
    d, idx, k = _knn(tree, tx, ty, k, len(px))
    pred = np.empty(len(tx)); var = np.empty(len(tx))
    for i in range(len(tx)):
        di, vi = d[i], pv[idx[i]]
        if di[0] < 1e-9:                      # coincide com um ponto: exato
            pred[i] = vi[0]; var[i] = 0.0; continue
        w = 1.0 / di ** power
        w /= w.sum()
        pred[i] = float(np.dot(w, vi))
        var[i] = float(np.dot(w, (vi - pred[i]) ** 2))   # dispersão ponderada (heurística)
    return pred, var


def oi_markov_predict(px, py, pv, psig, tx, ty, corr_len, sill, k=32):
    px, py, pv, psig = map(np.asarray, (px, py, pv, psig))
    tx, ty = np.atleast_1d(tx), np.atleast_1d(ty)
    tree = cKDTree(np.c_[px, py])
    d, idx, k = _knn(tree, tx, ty, k, len(px))

    def cov(r):
        r = np.asarray(r, float)
        return sill * (1.0 + r / corr_len) * np.exp(-r / corr_len)

    pred = np.empty(len(tx)); var = np.empty(len(tx))
    for i in range(len(tx)):
        j = idx[i]
        xl, yl, vl = px[j], py[j], pv[j]
        sl = np.where(psig[j] > 0, psig[j], 0.05)
        dx = xl[:, None] - xl[None, :]; dy = yl[:, None] - yl[None, :]
        C = cov(np.sqrt(dx * dx + dy * dy)) + np.diag(sl ** 2)
        c0 = cov(d[i])
        try:
            w = np.linalg.solve(C, c0)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(C) @ c0
        m0 = float(np.mean(vl))
        pred[i] = float(np.dot(w, vl) + (1.0 - w.sum()) * m0)
        var[i] = float(max(sill - np.dot(w, c0), 0.0))
    return pred, var


def ordinary_kriging_predict(px, py, pv, tx, ty, gamma, k=32):
    px, py, pv = map(np.asarray, (px, py, pv))
    tx, ty = np.atleast_1d(tx), np.atleast_1d(ty)
    tree = cKDTree(np.c_[px, py])
    d, idx, k = _knn(tree, tx, ty, k, len(px))

    pred = np.empty(len(tx)); var = np.empty(len(tx))
    for i in range(len(tx)):
        j = idx[i]
        xl, yl, vl = px[j], py[j], pv[j]
        dx = xl[:, None] - xl[None, :]; dy = yl[:, None] - yl[None, :]
        G = gamma(np.sqrt(dx * dx + dy * dy))
        # sistema de krigagem ordinária com multiplicador de Lagrange
        A = np.ones((k + 1, k + 1)); A[:k, :k] = G; A[k, k] = 0.0
        b = np.ones(k + 1); b[:k] = gamma(d[i])
        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(A) @ b
        w, mu = sol[:k], sol[k]
        pred[i] = float(np.dot(w, vl))
        var[i] = float(max(np.dot(w, gamma(d[i])) + mu, 0.0))
    return pred, var


# --------- assinatura unificada para a validação cruzada (select.py) --------
def gaussian_kernel_predict(px, py, pv, tx, ty, sigma, k=32):
    """
    Kernel gaussiano (equivalente ao `interpgaus.py` do captoolkit).

    Peso w_i = exp(−d_i²/(2σ²)). Suaviza mais que o IDW e não tem a
    singularidade dele em d→0. `sigma` é a escala de suavização (m).
    """
    px, py, pv = map(np.asarray, (px, py, pv))
    tx, ty = np.atleast_1d(tx), np.atleast_1d(ty)
    tree = cKDTree(np.c_[px, py])
    d, idx, k = _knn(tree, tx, ty, k, len(px))
    pred = np.empty(len(tx)); var = np.empty(len(tx))
    two_s2 = 2.0 * sigma * sigma
    for i in range(len(tx)):
        di, vi = d[i], pv[idx[i]]
        w = np.exp(-(di * di) / two_s2)
        sw = w.sum()
        if not np.isfinite(sw) or sw <= 0:
            pred[i] = np.nan; var[i] = np.nan; continue
        w = w / sw
        pred[i] = float(np.dot(w, vi))
        # variância ponderada corrigida pelo nº efetivo de amostras (Kish)
        n_eff = 1.0 / float(np.sum(w * w))
        var[i] = float(np.dot(w, (vi - pred[i]) ** 2) / max(n_eff, 1.0))
    return pred, var


def median_kernel_predict(px, py, pv, tx, ty, radius, k=32):
    """
    Kernel de mediana (equivalente ao `interpmed.py` do captoolkit).

    Mediana dos vizinhos dentro de `radius` — o mais robusto a outliers dos
    candidatos, ao custo de suavizar extremos reais. Incerteza via MAD/√n.
    """
    px, py, pv = map(np.asarray, (px, py, pv))
    tx, ty = np.atleast_1d(tx), np.atleast_1d(ty)
    tree = cKDTree(np.c_[px, py])
    d, idx, k = _knn(tree, tx, ty, k, len(px))
    pred = np.empty(len(tx)); var = np.empty(len(tx))
    for i in range(len(tx)):
        m = d[i] <= radius
        vi = pv[idx[i]][m] if m.any() else pv[idx[i]][:1]
        if vi.size == 0:
            pred[i] = np.nan; var[i] = np.nan; continue
        med = float(np.median(vi))
        pred[i] = med
        mad = 1.4826 * float(np.median(np.abs(vi - med)))
        # erro padrão da mediana ≈ 1.253·σ/√n
        var[i] = float((1.253 * mad) ** 2 / max(vi.size, 1))
    return pred, var


def _p_idw(trx, try_, trv, trs, tx, ty, cfg, vparams):
    return idw_predict(trx, try_, trv, tx, ty,
                       power=cfg.interpolation.idw_power, k=cfg.interpolation.neighbors)


def _p_oi(trx, try_, trv, trs, tx, ty, cfg, vparams):
    return oi_markov_predict(trx, try_, trv, trs, tx, ty,
                             corr_len=vparams["range_m"], sill=vparams["sill"],
                             k=cfg.interpolation.neighbors)


def _p_ok(trx, try_, trv, trs, tx, ty, cfg, vparams):
    from thwaites.interp.variogram import make_gamma
    return ordinary_kriging_predict(trx, try_, trv, tx, ty,
                                    gamma=make_gamma(vparams), k=cfg.interpolation.neighbors)


def _p_gauss(trx, try_, trv, trs, tx, ty, cfg, vparams):
    # escala de suavização: config, senão ~1/3 do alcance do variograma
    sigma = cfg.interpolation.gauss_sigma_m
    if sigma is None:
        sigma = (vparams["range_m"] / 3.0) if vparams else 10_000.0
    return gaussian_kernel_predict(trx, try_, trv, tx, ty, sigma=sigma,
                                   k=cfg.interpolation.neighbors)


def _p_median(trx, try_, trv, trs, tx, ty, cfg, vparams):
    radius = cfg.interpolation.median_radius_m
    if radius is None:
        radius = (vparams["range_m"] / 2.0) if vparams else 15_000.0
    return median_kernel_predict(trx, try_, trv, tx, ty, radius=radius,
                                 k=cfg.interpolation.neighbors)


PREDICTORS = {
    "idw": _p_idw,
    "oi_markov": _p_oi,
    "ordinary_kriging": _p_ok,
    "gaussian_kernel": _p_gauss,
    "median_kernel": _p_median,
}
