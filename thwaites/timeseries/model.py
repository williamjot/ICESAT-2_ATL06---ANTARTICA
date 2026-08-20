"""
thwaites.timeseries.model
=========================
Prioridade 3 (§4 do PLANO): modelo temporal com tendência, sazonalidade e
aceleração, com SELEÇÃO de modelo — não imposição.

MODELO COMPLETO (§4.3):

    h(x,y,t) = β0 + f(x,y) + β1·(t−t0) + ½·β2·(t−t0)²
               + a1·sin(2πt) + b1·cos(2πt) + γ_beam + ε

CONVENÇÃO DE ACELERAÇÃO (§4.3 exige convenção física inequívoca):
o termo quadrático é parametrizado como ½·β2·(t−t0)², de modo que

        β1 = dh/dt        [m/ano]
        β2 = d²h/dt²      [m/ano²]   (a aceleração, DIRETAMENTE)

Se o termo fosse β2·(t−t0)², a aceleração seria 2·β2 — é justamente essa
ambiguidade que a convenção acima elimina.

IDENTIFICABILIDADE (trava científica central deste módulo)
-----------------------------------------------------------------
§4.2: "Os dados JJA, isoladamente, não permitem estimar adequadamente um ciclo
sazonal anual." Com todas as observações dentro de uma janela de 3 meses, as
colunas sin(2πt)/cos(2πt) ficam quase colineares com a constante: o ajuste
retorna coeficientes enormes e sem sentido, que *parecem* um resultado.

Por isso `check_identifiability()` mede a cobertura de FASE do ciclo anual e o
condicionamento da matriz de projeto, e o ajuste RECUSA termos não suportados
pelos dados, em vez de devolver números espúrios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from thwaites.logging import get_logger

# Candidatos ATIVOS: foco em tendência × aceleração (§4.3/§4.4).
# A sazonalidade foi retirada do conjunto padrão porque a base atual é só JJA e
# o ciclo anual é NÃO-IDENTIFICÁVEL nessa amostragem (§4.2). O maquinário
# sazonal continua implementado e travado por `check_identifiability` — basta
# incluir os candidatos de SEASONAL_CANDIDATES quando houver base anual.
MODEL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "linear":        ("constant", "spatial", "linear"),
    "linear_accel":  ("constant", "spatial", "linear", "acceleration"),
    "linear_beam":   ("constant", "spatial", "linear", "beam"),
}

# Disponíveis só quando a cobertura de fase anual permitir (ver §4.2).
SEASONAL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "linear_seasonal":       ("constant", "spatial", "linear", "seasonal"),
    "linear_seasonal_accel": ("constant", "spatial", "linear", "seasonal", "acceleration"),
    "linear_seasonal_beam":  ("constant", "spatial", "linear", "seasonal", "beam"),
}


# --------------------------------------------------------- identificabilidade
def seasonal_phase_coverage(t) -> float:
    """
    Fração do ciclo anual coberta pelas observações (0–1).

    Usa a dispersão circular das fases: valores próximos de 0 significam que as
    observações estão concentradas numa janela estreita do ano (caso JJA) e a
    sazonalidade NÃO é identificável.
    """
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return 0.0
    phase = 2 * np.pi * (t % 1.0)
    R = np.hypot(np.mean(np.cos(phase)), np.mean(np.sin(phase)))
    # R≈1 -> fases concentradas; R≈0 -> bem espalhadas pelo ano
    return float(1.0 - R)


def check_identifiability(t, terms, min_phase_coverage: float = 0.35,
                          max_condition: float = 1e4,
                          min_years_for_accel: int = 5) -> dict:
    """
    Verifica se os termos pedidos são estimáveis com a amostragem temporal dada.

    Retorna {"ok": bool, "allowed_terms": (...), "reasons": {...}, ...}.
    NÃO ajusta nada — só decide o que pode ser ajustado.
    """
    t = np.asarray(t, dtype=float)
    reasons: dict[str, str] = {}
    allowed = list(terms)

    cov = seasonal_phase_coverage(t)
    if "seasonal" in allowed and cov < min_phase_coverage:
        allowed.remove("seasonal")
        reasons["seasonal"] = (
            f"cobertura de fase anual {cov:.3f} < {min_phase_coverage} — as "
            f"observações estão concentradas numa janela estreita do ano "
            f"(ex.: só JJA); sin/cos ficam quase colineares com a constante.")

    n_years = int(np.unique(np.floor(t)).size)
    if "acceleration" in allowed and n_years < min_years_for_accel:
        allowed.remove("acceleration")
        reasons["acceleration"] = (
            f"{n_years} anos distintos < {min_years_for_accel} — período curto "
            f"demais para separar aceleração de variabilidade interanual.")

    return {"ok": len(reasons) == 0, "allowed_terms": tuple(allowed),
            "requested_terms": tuple(terms), "reasons": reasons,
            "phase_coverage": cov, "n_years": n_years}


# ---------------------------------------------------------- matriz de projeto
def build_design_matrix(t, dx, dy, terms, t_ref: float,
                        beam=None, poly_order: int = 2) -> tuple[np.ndarray, list[str]]:
    """
    Monta a matriz de projeto e os nomes das colunas.

    `terms` ⊆ {constant, spatial, linear, acceleration, seasonal, beam}.
    """
    t = np.asarray(t, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    n = t.size
    dt = t - t_ref
    cols, names = [], []

    if "constant" in terms:
        cols.append(np.ones(n)); names.append("constant")
    if "linear" in terms:
        cols.append(dt); names.append("dhdt")
    if "acceleration" in terms:
        # convenção: ½·β2·dt²  =>  β2 = d²h/dt² diretamente
        cols.append(0.5 * dt ** 2); names.append("accel")
    if "seasonal" in terms:
        cols.append(np.sin(2 * np.pi * t)); names.append("sin_annual")
        cols.append(np.cos(2 * np.pi * t)); names.append("cos_annual")
    if "spatial" in terms:
        sx = max(float(np.std(dx)), 1.0)
        sy = max(float(np.std(dy)), 1.0)
        xn, yn = dx / sx, dy / sy
        cols += [xn, yn, xn * yn]; names += ["x", "y", "xy"]
        if poly_order >= 2:
            cols += [xn ** 2, yn ** 2]; names += ["x2", "y2"]
    if "beam" in terms and beam is not None:
        b = np.asarray(beam)
        uniq = np.unique(b)
        for u in uniq[1:]:            # primeiro feixe é a referência
            cols.append((b == u).astype(float)); names.append(f"beam_{u}")

    return np.column_stack(cols), names


# ------------------------------------------------------------------ ajuste
@dataclass
class FitResult:
    model: str
    terms: tuple
    names: list
    coef: np.ndarray
    cov: np.ndarray | None
    n: int
    k: int
    rss: float
    aicc: float
    condition: float
    resid: np.ndarray = field(repr=False, default=None)

    def value(self, name: str) -> float:
        return float(self.coef[self.names.index(name)]) if name in self.names else np.nan

    def stderr(self, name: str) -> float:
        if self.cov is None or name not in self.names:
            return np.nan
        i = self.names.index(name)
        return float(np.sqrt(max(self.cov[i, i], 0.0)))


def _aicc(n: int, k: int, rss: float) -> float:
    if n <= k + 1 or rss <= 0:
        return np.inf
    aic = n * np.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def fit_model(h, t, dx, dy, terms, t_ref, beam=None, weights=None,
              robust_iters: int = 0, n_sigma: float = 3.0) -> FitResult | None:
    """
    Ajusta um modelo. `robust_iters>0` faz rejeição robusta iterativa (MAD).

    Retorna None se o sistema for mal condicionado ou tiver graus de liberdade
    insuficientes — nunca devolve um ajuste que não se sustenta.
    """
    h = np.asarray(h, dtype=float)
    A, names = build_design_matrix(t, dx, dy, terms, t_ref, beam=beam)
    n, k = A.shape
    if n < k + 3:
        return None

    mask = np.ones(n, dtype=bool)
    coef = cov = None
    for _ in range(max(robust_iters, 0) + 1):
        Af, hf = A[mask], h[mask]
        if Af.shape[0] < k + 3:
            return None
        if weights is not None:
            w = np.asarray(weights, dtype=float)[mask]
            w = w / (np.mean(w) + 1e-30)
            sw = np.sqrt(w)
            Aw, hw = Af * sw[:, None], hf * sw
        else:
            Aw, hw = Af, hf

        cond = float(np.linalg.cond(Aw))
        if not np.isfinite(cond) or cond > 1e10:
            return None
        try:
            coef, *_ = np.linalg.lstsq(Aw, hw, rcond=None)
        except np.linalg.LinAlgError:
            return None

        resid_f = hf - Af @ coef
        dof = max(Af.shape[0] - k, 1)
        sigma2 = float(np.sum(resid_f ** 2) / dof)
        try:
            cov = np.linalg.inv(Aw.T @ Aw) * sigma2
        except np.linalg.LinAlgError:
            cov = None

        if robust_iters <= 0:
            break
        mad = 1.4826 * np.median(np.abs(resid_f - np.median(resid_f)))
        if mad < 1e-12:
            break
        bad = np.abs(resid_f) > n_sigma * mad
        if not bad.any():
            break
        idx = np.where(mask)[0][bad]
        mask[idx] = False

    resid = h[mask] - A[mask] @ coef
    rss = float(np.sum(resid ** 2))
    return FitResult(model="custom", terms=tuple(terms), names=names, coef=coef,
                     cov=cov, n=int(mask.sum()), k=k, rss=rss,
                     aicc=_aicc(int(mask.sum()), k, rss),
                     condition=float(np.linalg.cond(A[mask])), resid=resid)


# ------------------------------------------------------------ seleção (§4.4)
def leave_one_year_out(h, t, dx, dy, terms, t_ref, beam=None) -> float:
    """
    RMSE fora da amostra retendo um ANO inteiro por vez (§4.4).

    Reter observações ao acaso não testaria a capacidade temporal do modelo —
    por isso a unidade retida é o ano.
    """
    t = np.asarray(t, dtype=float)
    years = np.unique(np.floor(t))
    if years.size < 3:
        return np.nan
    errs = []
    for y in years:
        te = np.floor(t) == y
        tr = ~te
        if tr.sum() < 10 or te.sum() < 3:
            continue
        fit = fit_model(np.asarray(h)[tr], t[tr], np.asarray(dx)[tr],
                        np.asarray(dy)[tr], terms, t_ref,
                        beam=None if beam is None else np.asarray(beam)[tr])
        if fit is None:
            continue
        A, _ = build_design_matrix(t[te], np.asarray(dx)[te], np.asarray(dy)[te],
                                   terms, t_ref,
                                   beam=None if beam is None else np.asarray(beam)[te])
        if A.shape[1] != fit.coef.size:
            continue
        errs.append(np.asarray(h)[te] - A @ fit.coef)
    if not errs:
        return np.nan
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2)))


def residual_autocorrelation(resid, t) -> float:
    """Autocorrelação lag-1 dos resíduos ordenados no tempo (§4.4)."""
    r = np.asarray(resid, dtype=float)
    order = np.argsort(np.asarray(t, dtype=float))
    r = r[order]
    if r.size < 3:
        return np.nan
    r = r - np.mean(r)
    den = float(np.sum(r ** 2))
    return float(np.sum(r[1:] * r[:-1]) / den) if den > 0 else np.nan


def select_model(h, t, dx, dy, t_ref, beam=None, candidates=None,
                 weights=None, use_oos: bool = True) -> dict:
    """
    Compara modelos candidatos por AICc e validação temporal fora da amostra.

    Só considera candidatos cujos termos passem na checagem de
    identificabilidade — modelos não estimáveis são REJEITADOS, não ajustados.
    """
    cands = candidates or MODEL_CANDIDATES
    rows, fits = [], {}
    for name, terms in cands.items():
        ident = check_identifiability(t, terms)
        if set(ident["allowed_terms"]) != set(terms):
            rows.append({"model": name, "status": "não-identificável",
                         "reasons": "; ".join(ident["reasons"].values()),
                         "aicc": np.nan, "oos_rmse": np.nan})
            continue
        fit = fit_model(h, t, dx, dy, terms, t_ref, beam=beam, weights=weights)
        if fit is None:
            rows.append({"model": name, "status": "falhou", "reasons": "mal condicionado",
                         "aicc": np.nan, "oos_rmse": np.nan})
            continue
        oos = leave_one_year_out(h, t, dx, dy, terms, t_ref, beam) if use_oos else np.nan
        fits[name] = fit
        rows.append({"model": name, "status": "ok", "reasons": "",
                     "aicc": fit.aicc, "oos_rmse": oos, "k": fit.k, "n": fit.n,
                     "dhdt": fit.value("dhdt"), "dhdt_err": fit.stderr("dhdt"),
                     "accel": fit.value("accel"), "accel_err": fit.stderr("accel"),
                     "resid_ac1": residual_autocorrelation(fit.resid, t[-fit.resid.size:])})

    table = pd.DataFrame(rows)
    ok = table[table["status"] == "ok"]
    best_aicc = ok.nsmallest(1, "aicc")["model"].iloc[0] if not ok.empty else None
    valid_oos = ok.dropna(subset=["oos_rmse"])
    best_oos = valid_oos.nsmallest(1, "oos_rmse")["model"].iloc[0] if not valid_oos.empty else None
    return {"table": table, "fits": fits,
            "best_by_aicc": best_aicc, "best_by_oos": best_oos,
            "agree": bool(best_aicc is not None and best_aicc == best_oos)}


# ------------------------------------------------------------- sazonalidade
def seasonal_amplitude_phase(fit: FitResult) -> dict:
    """
    Amplitude e fase do ciclo anual a partir de a1·sin + b1·cos (§4.4).

    amplitude = √(a1²+b1²); fase = atan2(b1,a1) convertida para o dia do ano do
    máximo. Retorna NaN se o modelo não tem termo sazonal.
    """
    if "sin_annual" not in fit.names:
        return {"amplitude_m": np.nan, "phase_doy": np.nan}
    a1 = fit.value("sin_annual")
    b1 = fit.value("cos_annual")
    amp = float(np.hypot(a1, b1))
    phase = float(np.arctan2(b1, a1))         # rad
    doy = float((phase / (2 * np.pi)) % 1.0 * 365.25)
    return {"amplitude_m": amp, "phase_doy": doy, "a1": a1, "b1": b1}


# --------------------------------------------------------- JJA versus anual
def compare_jja_annual(nodes_jja: pd.DataFrame, nodes_annual: pd.DataFrame,
                       key=("x", "y"), bootstrap: int = 1000, seed: int = 0) -> dict:
    """
    Comparação PAREADA entre a tendência estimada só com JJA e com o ciclo
    anual (§4.5).

    §4.5 é explícito: "Não se deve pressupor que JJA seja conservador. Essa
    hipótese deverá ser testada." Esta função TESTA — o sinal da diferença é
    um resultado, não uma premissa.
    """
    k = list(key)
    m = nodes_jja[k + ["dhdt"]].merge(nodes_annual[k + ["dhdt"]], on=k,
                                      suffixes=("_jja", "_annual"))
    if m.empty:
        return {"n_paired": 0}
    d = (m["dhdt_jja"] - m["dhdt_annual"]).to_numpy()
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(d, size=(bootstrap, d.size), replace=True), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    med = float(np.median(d))
    # JJA "conservador" = subestima a perda = dh/dt de JJA MENOS negativo
    conservative = bool(med > 0 and lo > 0)
    return {
        "n_paired": int(len(m)),
        "median_diff_jja_minus_annual": med,
        "ci95": [float(lo), float(hi)],
        "significant": bool(not (lo <= 0 <= hi)),
        "jja_is_conservative": conservative,
        "interpretation": (
            "JJA subestima a perda (conservador)" if conservative else
            "JJA superestima a perda" if (med < 0 and hi < 0) else
            "diferença não distinguível de zero"),
    }
