"""
thwaites.timeseries.acceleration
================================
Prioridade 3 (§4), recorte de ACELERAÇÃO.

A pergunta científica: **há evidência estatística suficiente para afirmar
aceleração do adelgaçamento?**

§4.4 é explícito: "A aceleração não deverá ser incluída automaticamente em
todos os nós. Um período de aproximadamente sete anos é curto para separar
aceleração de variabilidade interanual."

Este módulo NÃO estima aceleração em todo lugar — ele decide, nó a nó, se os
dados sustentam o termo quadrático, exigindo concordância de VÁRIOS critérios
independentes (§4.4):

  1. **AICc**: o modelo com aceleração vence o linear parcimonioso?
  2. **Validação fora da amostra**: retendo um ANO inteiro por vez, o modelo com
     aceleração prevê melhor? (é o teste mais duro — AICc sozinho premia ajuste)
  3. **Intervalo de confiança** da aceleração não cruza zero;
  4. **Estabilidade bootstrap**: o sinal do coeficiente se mantém em reamostragem;
  5. **Sensibilidade a remover um ano**: nenhum ano isolado sustenta sozinho o
     resultado (leave-one-year-out sobre o próprio coeficiente);
  6. **Autocorrelação residual** baixa (resíduo estruturado invalida o erro formal).

Só se TODOS os critérios exigidos passarem a aceleração é reportada; caso
contrário o nó fica com `accel = NaN` e o motivo registrado. É deliberadamente
conservador: com ~7 invernos, o risco de reportar variabilidade interanual como
aceleração é alto.

CONVENÇÃO (§4.3): o termo é ½·β2·(t−t0)², logo **β2 = d²h/dt² em m/ano²**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thwaites.logging import get_logger
from thwaites.timeseries.model import (
    fit_model, leave_one_year_out, residual_autocorrelation, check_identifiability,
)

LINEAR_TERMS = ("constant", "spatial", "linear")
ACCEL_TERMS = ("constant", "spatial", "linear", "acceleration")

ACCEL_COLUMNS = ["x", "y", "lon", "lat", "dhdt", "dhdt_err", "accel", "accel_err",
                 "accel_supported", "reason", "delta_aicc", "delta_oos",
                 "accel_ci95_lo", "accel_ci95_hi",
                 "boot_sign_frac", "loyo_max_shift", "resid_ac1", "n_obs", "n_years"]


@dataclass
class AccelCriteria:
    """Limiares dos critérios (§4.4). Pré-definidos, não ajustados a posteriori."""
    min_years: int = 5           # anos distintos mínimos p/ sequer tentar
    alpha: float = 0.05          # nível do IC da aceleração
    delta_aicc: float = 2.0      # ganho de AICc exigido (2 = evidência fraca; usar ≥2)
    require_oos_gain: bool = True    # exigir melhora fora da amostra
    boot_iters: int = 200
    min_boot_sign_frac: float = 0.95  # fração de reamostras com o mesmo sinal
    max_loyo_shift_frac: float = 0.5  # nenhum ano pode mudar |accel| mais que isso
    max_resid_ac1: float = 0.5        # autocorrelação lag-1 máxima tolerada


def _bootstrap_accel(h, t, dx, dy, t_ref, weights, iters, seed):
    """
    Distribuição bootstrap da aceleração — reamostrando BLOCOS DE ANO.

    POR QUE ISSO É O TESTE PRINCIPAL (e não o erro formal):
    o erro formal do mínimos quadrados e o AICc tratam as ~1000 observações
    como independentes. Mas a aceleração é um padrão ENTRE ANOS: a amostra
    efetiva é o número de anos (~7), não o de observações. Com ~150 pontos por
    ano, a média anual é praticamente fixa, então o erro formal subestima a
    incerteza da curvatura em cerca de uma ordem de magnitude — e um ruído
    puro passa como "aceleração significativa a 2,3σ".

    Reamostrar ANOS mede a incerteza que de fato importa. Retorna o vetor de
    acelerações reamostradas.
    """
    rng = np.random.default_rng(seed)
    yr = np.floor(t)
    years = np.unique(yr)
    if years.size < 4:
        return np.array([])

    # ------------------------------------------------------------------
    # IMPLEMENTAÇÃO POR EQUAÇÕES NORMAIS PRÉ-ACUMULADAS POR ANO
    # ------------------------------------------------------------------
    # A reamostragem é de ANOS INTEIROS: pré-computa por ano os blocos das
    # equações normais S_y = Aᵀ W A e b_y = Aᵀ W h (4x4 e 4x1). Uma
    # reamostra é a SOMA dos blocos dos anos sorteados + solve 4x4.
    #
    # A matriz de projeto é montada UMA vez com todos os pontos. Isso é válido
    # apesar de `build_design_matrix` normalizar as colunas espaciais pelo
    # desvio da amostra: reescalar uma coluna não altera o espaço-coluna, logo
    # o coeficiente de aceleração — que vive na coluna 0,5·dt² — é invariante.
    # A equivalência é verificada em tests/test_acceleration_bootstrap.py.
    from thwaites.timeseries.model import build_design_matrix

    A, names = build_design_matrix(t, dx, dy, ACCEL_TERMS, t_ref)
    if "accel" not in names:
        return np.array([])
    ia = names.index("accel")

    if weights is None:
        w = np.ones(len(h), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / (np.mean(w) + 1e-30)
    sw = np.sqrt(w)
    Aw = A * sw[:, None]
    hw = np.asarray(h, dtype=float) * sw

    S_y, b_y, n_y = {}, {}, {}
    for y in years:
        m = yr == y
        Ay = Aw[m]
        S_y[y] = Ay.T @ Ay
        b_y[y] = Ay.T @ hw[m]
        n_y[y] = int(m.sum())

    k = A.shape[1]
    vals = []
    for _ in range(iters):
        pick = rng.choice(years, size=years.size, replace=True)
        if np.unique(pick).size < 3:          # reamostra degenerada
            continue
        if sum(n_y[y] for y in pick) < k + 3:  # graus de liberdade insuficientes
            continue
        S = np.zeros((k, k))
        b = np.zeros(k)
        for y in pick:
            S += S_y[y]
            b += b_y[y]
        try:
            c = np.linalg.solve(S, b)
        except np.linalg.LinAlgError:          # bloco singular -> descartada
            continue
        a = c[ia]
        if np.isfinite(a):
            vals.append(a)
    return np.asarray(vals, dtype=float)


def _leave_one_year_shift(h, t, dx, dy, t_ref, weights, accel_full):
    """
    Maior mudança RELATIVA no coeficiente ao remover um ano inteiro (§4.4).

    Se remover um único ano muda drasticamente a aceleração, o resultado
    depende daquele ano — não é tendência de longo prazo.
    """
    years = np.unique(np.floor(t))
    if years.size < 3 or not np.isfinite(accel_full) or abs(accel_full) < 1e-12:
        return np.nan
    shifts = []
    for y in years:
        keep = np.floor(t) != y
        if keep.sum() < 20:
            continue
        f = fit_model(h[keep], t[keep], dx[keep], dy[keep], ACCEL_TERMS, t_ref,
                      weights=None if weights is None else weights[keep])
        if f is None:
            continue
        a = f.value("accel")
        if np.isfinite(a):
            shifts.append(abs(a - accel_full) / abs(accel_full))
    return float(np.max(shifts)) if shifts else np.nan


def assess_acceleration(h, t, dx, dy, t_ref, weights=None,
                        criteria: AccelCriteria | None = None,
                        seed: int = 0) -> dict:
    """
    Decide se há suporte estatístico para aceleração num nó.

    Retorna dict com o veredito (`accel_supported`), o valor (ou NaN) e todos
    os diagnósticos, para que a decisão seja auditável.
    """
    c = criteria or AccelCriteria()
    h = np.asarray(h, float); t = np.asarray(t, float)
    dx = np.asarray(dx, float); dy = np.asarray(dy, float)
    w = None if weights is None else np.asarray(weights, float)

    out = {k: np.nan for k in
           ("dhdt", "dhdt_err", "accel", "accel_err", "delta_aicc", "delta_oos",
            "boot_sign_frac", "loyo_max_shift", "resid_ac1",
            "accel_ci95_lo", "accel_ci95_hi")}
    out["accel_supported"] = False
    out["n_obs"] = int(h.size)
    out["n_years"] = int(np.unique(np.floor(t)).size)

    # --- pré-requisito: período suficiente ---------------------------------
    ident = check_identifiability(t, ACCEL_TERMS, min_years_for_accel=c.min_years)
    if "acceleration" not in ident["allowed_terms"]:
        out["reason"] = ident["reasons"].get("acceleration", "não identificável")
        lin = fit_model(h, t, dx, dy, LINEAR_TERMS, t_ref, weights=w)
        if lin is not None:
            out["dhdt"] = lin.value("dhdt"); out["dhdt_err"] = lin.stderr("dhdt")
        return out

    # --- ajusta os dois modelos --------------------------------------------
    lin = fit_model(h, t, dx, dy, LINEAR_TERMS, t_ref, weights=w)
    acc = fit_model(h, t, dx, dy, ACCEL_TERMS, t_ref, weights=w)
    if lin is None or acc is None:
        out["reason"] = "ajuste falhou (mal condicionado ou dados insuficientes)"
        if lin is not None:
            out["dhdt"] = lin.value("dhdt"); out["dhdt_err"] = lin.stderr("dhdt")
        return out

    # taxa reportada vem do modelo LINEAR (parcimonioso) por padrão
    out["dhdt"] = lin.value("dhdt")
    out["dhdt_err"] = lin.stderr("dhdt")
    a_val, a_err = acc.value("accel"), acc.stderr("accel")

    # --- critério 1: AICc ---------------------------------------------------
    out["delta_aicc"] = float(lin.aicc - acc.aicc)      # >0 => aceleração melhor
    aicc_ok = out["delta_aicc"] >= c.delta_aicc

    # --- critério 6: autocorrelação residual --------------------------------
    out["resid_ac1"] = residual_autocorrelation(acc.resid, t[-acc.resid.size:])
    ac_ok = (not np.isfinite(out["resid_ac1"])) or abs(out["resid_ac1"]) <= c.max_resid_ac1

    # atalho: se já falhou nos baratos, não gasta bootstrap/OOS
    if not (aicc_ok and ac_ok):
        motivos = []
        if not aicc_ok:
            motivos.append(f"ΔAICc {out['delta_aicc']:.1f} < {c.delta_aicc}")
        if not ac_ok:
            motivos.append(f"resíduo autocorrelacionado (ac1={out['resid_ac1']:.2f})")
        out["reason"] = "; ".join(motivos)
        return out

    # --- critério 3+4: IC BOOTSTRAP POR ANO (teste principal) ---------------
    # substitui o IC do erro formal: aquele trata ~1000 observações como
    # independentes e superestima a significância de um padrão temporal cuja
    # amostra efetiva é o nº de anos (~7).
    boot = _bootstrap_accel(h, t, dx, dy, t_ref, w, c.boot_iters, seed)
    if boot.size >= 20:
        lo = float(np.percentile(boot, 100 * c.alpha / 2))
        hi = float(np.percentile(boot, 100 * (1 - c.alpha / 2)))
        out["accel_ci95_lo"], out["accel_ci95_hi"] = lo, hi
        ci_ok = not (lo <= 0.0 <= hi)
        ref = np.sign(np.median(boot))
        out["boot_sign_frac"] = float(np.mean(np.sign(boot) == ref))
    else:
        ci_ok = False
        out["accel_ci95_lo"] = out["accel_ci95_hi"] = np.nan
    boot_ok = np.isfinite(out["boot_sign_frac"]) and out["boot_sign_frac"] >= c.min_boot_sign_frac

    if not (ci_ok and boot_ok):
        motivos = []
        if not ci_ok:
            motivos.append("IC bootstrap (por ano) da aceleração cruza zero")
        if not boot_ok:
            motivos.append(f"sinal instável no bootstrap ({out['boot_sign_frac']:.0%})")
        out["reason"] = "; ".join(motivos)
        return out

    # --- critério 2: fora da amostra (retendo um ano) -----------------------
    oos_lin = leave_one_year_out(h, t, dx, dy, LINEAR_TERMS, t_ref)
    oos_acc = leave_one_year_out(h, t, dx, dy, ACCEL_TERMS, t_ref)
    out["delta_oos"] = float(oos_lin - oos_acc) if np.isfinite(oos_lin) and np.isfinite(oos_acc) else np.nan
    oos_ok = (not c.require_oos_gain) or (np.isfinite(out["delta_oos"]) and out["delta_oos"] > 0)

    # --- critério 5: sensibilidade a remover um ano -------------------------
    out["loyo_max_shift"] = _leave_one_year_shift(h, t, dx, dy, t_ref, w, a_val)
    loyo_ok = (not np.isfinite(out["loyo_max_shift"])) or \
              (out["loyo_max_shift"] <= c.max_loyo_shift_frac)

    supported = bool(aicc_ok and ci_ok and ac_ok and oos_ok and boot_ok and loyo_ok)
    out["accel_supported"] = supported
    if supported:
        out["accel"] = a_val
        # erro reportado = desvio do bootstrap POR ANO (não o formal, que
        # subestima a incerteza de um padrão temporal)
        out["accel_err"] = float(np.std(boot)) if boot.size >= 20 else a_err
        # com aceleração sustentada, a taxa reportada é a do modelo completo
        out["dhdt"] = acc.value("dhdt"); out["dhdt_err"] = acc.stderr("dhdt")
        out["reason"] = "todos os critérios atendidos"
    else:
        motivos = []
        if not oos_ok:
            motivos.append("sem ganho fora da amostra")
        if not boot_ok:
            motivos.append(f"instável no bootstrap (sinal em {frac:.0%})")
        if not loyo_ok:
            motivos.append(f"depende de um único ano (Δ {out['loyo_max_shift']:.0%})")
        out["reason"] = "; ".join(motivos)
    return out


# ------------------------------------------------------------- por região
def acceleration_field(points: pd.DataFrame, cfg, x_min, x_max, y_min, y_max,
                       criteria: AccelCriteria | None = None) -> pd.DataFrame:
    """
    Avalia aceleração nos nós de uma região, usando a mesma grade e raio de
    busca do dh/dt (comparabilidade direta com o produto principal).
    """
    from scipy.spatial import cKDTree
    from thwaites.grid.reproject import to_lonlat
    from thwaites.grid.tiles import assign_xy

    logger = get_logger()
    d = cfg.dhdt
    points = assign_xy(points, cfg)
    hcol = next((c for c in ("h_res", "h_corr", "h_elv") if c in points.columns), None)
    if hcol is None:
        raise ValueError("nenhuma coluna de elevação disponível")

    ok = ~(points["x"].isna() | points["y"].isna() |
           points[hcol].isna() | points["t_year"].isna())
    x = points["x"].to_numpy()[ok]; y = points["y"].to_numpy()[ok]
    h = points[hcol].to_numpy()[ok].astype(float)
    t = points["t_year"].to_numpy()[ok]
    s = (points["s_elv"].to_numpy()[ok].astype(float)
         if "s_elv" in points.columns else np.full(ok.sum(), 0.05))
    w = 1.0 / (np.where(s > 0, s, np.median(s[s > 0]) if (s > 0).any() else 0.05) ** 2 + 1e-12)

    step = d.node_spacing_m
    gx = np.arange(x_min + step / 2, x_max, step)
    gy = np.arange(y_min + step / 2, y_max, step)
    if gx.size == 0 or gy.size == 0 or h.size < d.min_points:
        return pd.DataFrame({c: [] for c in ACCEL_COLUMNS})
    GX, GY = np.meshgrid(gx, gy)
    tree = cKDTree(np.c_[x, y])

    rows = []
    for nx, ny in zip(GX.ravel(), GY.ravel()):
        idx = np.asarray(tree.query_ball_point([nx, ny], r=d.search_radius_m), dtype=int)
        if idx.size < d.min_points:
            continue
        res = assess_acceleration(h[idx], t[idx], x[idx] - nx, y[idx] - ny,
                                  d.t_ref, weights=w[idx], criteria=criteria)
        res["x"], res["y"] = float(nx), float(ny)
        rows.append(res)

    if not rows:
        return pd.DataFrame({c: [] for c in ACCEL_COLUMNS})
    out = pd.DataFrame(rows)
    lon, lat = to_lonlat(out["x"].to_numpy(), out["y"].to_numpy(), cfg)
    out["lon"], out["lat"] = lon, lat
    n_sup = int(out["accel_supported"].sum())
    logger.info(f"aceleração: {n_sup:,}/{len(out):,} nós com suporte estatístico "
                f"({100*n_sup/max(len(out),1):.1f}%)")
    return out[ACCEL_COLUMNS]
