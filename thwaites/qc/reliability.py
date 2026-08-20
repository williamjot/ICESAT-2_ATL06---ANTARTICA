"""
thwaites.qc.reliability
=======================
Classificação objetiva de cada nó de dh/dt em três níveis de confiabilidade.

Princípio: os limiares são PRÉ-DECLARADOS e derivados de exigências
estatísticas, não escolhidos depois de ver o resultado. Cada um responde a um
modo de falha específico observado neste projeto:

* `min_obs` — nós com poucas observações produzem valores extremos. Medido: os
  46 nós com dh/dt > +1 m/ano tinham 6.299 observações contra 155.818 da
  mediana geral.
* `min_years` — uma taxa exige épocas, não pontos. A amostra efetiva de um
  dh/dt é o número de ANOS distintos; foi por isso que o erro formal era
  otimista por fator ~50×.
* `max_rmse` — resíduo alto indica que uma superfície linear no tempo não
  descreve o dado (blunders, mistura de superfícies, sinal sazonal residual).
* `max_sigma` — incerteza da própria taxa, do jackknife.
* `min_snr` — |dh/dt|/σ. Um nó pode ser bem amostrado e ainda assim ter taxa
  indistinguível de zero; isso não o torna inválido, mas impede afirmar sinal.
* `min_tspan` — extensão temporal curta não resolve tendência.

A classe "aceitável com ressalvas" existe para não jogar fora nós utilizáveis em
estatística agregada mas inadequados para leitura pontual — descartá-los
enviesaria a cobertura espacial justamente nas margens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Limiares pré-declarados. Alterá-los é uma decisão metodológica: registre o
# motivo e refaça a análise de sensibilidade.
CRITERIA = {
    "confiavel": {
        "min_obs": 5000,        # amostragem espacial densa no raio de busca
        "min_years": 5,         # >=5 dos 7 anos representados
        "max_rmse_m": 1.0,      # ajuste descreve o dado
        "max_sigma_m_yr": 0.15,  # taxa bem determinada
        "min_snr": 2.0,         # taxa distinguível de zero a 2σ
        "min_tspan_yr": 4.0,    # base temporal longa
    },
    "aceitavel": {
        "min_obs": 500,
        "min_years": 4,         # mínimo para uma tendência ter sentido
        "max_rmse_m": 2.5,
        "max_sigma_m_yr": 0.5,
        "min_snr": 0.0,         # não exige significância
        "min_tspan_yr": 3.0,
    },
}

LABELS = {"confiavel": "confiável",
          "aceitavel": "aceitável com ressalvas",
          "nao_confiavel": "não confiável"}


def _years_col(nodes: pd.DataFrame) -> np.ndarray:
    """Número de épocas por nó, com degradação explícita se ausente."""
    if "n_years_node" in nodes.columns:
        return nodes["n_years_node"].to_numpy(dtype=float)
    # sem a contagem real, tspan é um limite SUPERIOR do nº de anos — usá-lo é
    # otimista, então fica registrado no relatório
    return nodes["tspan"].to_numpy(dtype=float)


def _passes(nodes: pd.DataFrame, thr: dict) -> np.ndarray:
    nobs = nodes["nobs"].to_numpy(dtype=float)
    rmse = nodes["rmse"].to_numpy(dtype=float)
    sig = nodes["dhdt_err"].to_numpy(dtype=float)
    dh = nodes["dhdt"].to_numpy(dtype=float)
    tsp = nodes["tspan"].to_numpy(dtype=float)
    yrs = _years_col(nodes)

    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.abs(dh) / sig

    ok = np.isfinite(dh) & np.isfinite(sig) & (sig > 0)
    ok &= nobs >= thr["min_obs"]
    ok &= yrs >= thr["min_years"]
    ok &= rmse <= thr["max_rmse_m"]
    ok &= sig <= thr["max_sigma_m_yr"]
    ok &= tsp >= thr["min_tspan_yr"]
    if thr["min_snr"] > 0:
        ok &= snr >= thr["min_snr"]
    return ok


def classify_nodes(nodes: pd.DataFrame, criteria: dict | None = None) -> pd.DataFrame:
    """
    Adiciona a coluna `reliability` e as colunas auxiliares `snr` e
    `n_years_used`. Não remove nada — a decisão de filtrar é de quem usa.
    """
    criteria = criteria or CRITERIA
    out = nodes.copy()

    sig = out["dhdt_err"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["snr"] = np.abs(out["dhdt"].to_numpy(dtype=float)) / sig
    out["n_years_used"] = _years_col(out)

    conf = _passes(out, criteria["confiavel"])
    acei = _passes(out, criteria["aceitavel"])

    lab = np.full(len(out), LABELS["nao_confiavel"], dtype=object)
    lab[acei] = LABELS["aceitavel"]
    lab[conf] = LABELS["confiavel"]     # confiável tem precedência
    out["reliability"] = lab
    return out


def reliability_report(nodes: pd.DataFrame, criteria: dict | None = None) -> dict:
    """Resumo por classe, com as estatísticas que justificam a separação."""
    criteria = criteria or CRITERIA
    d = nodes if "reliability" in nodes.columns else classify_nodes(nodes, criteria)
    rep = {"criteria": criteria,
           "n_total": int(len(d)),
           "uses_real_year_count": bool("n_years_node" in nodes.columns),
           "classes": {}}
    for lab in LABELS.values():
        s = d["reliability"] == lab
        if not s.any():
            rep["classes"][lab] = {"n": 0}
            continue
        g = d[s]
        rep["classes"][lab] = {
            "n": int(s.sum()),
            "pct": float(100 * s.mean()),
            "dhdt_median": float(g["dhdt"].median()),
            "dhdt_mean": float(g["dhdt"].mean()),
            "nobs_median": float(g["nobs"].median()),
            "rmse_median": float(g["rmse"].median()),
            "sigma_median": float(g["dhdt_err"].median()),
            "snr_median": float(np.nanmedian(g["snr"])),
            "thinning_pct": float(100 * (g["dhdt"] < 0).mean()),
        }
    return rep
