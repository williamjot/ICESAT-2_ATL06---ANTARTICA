"""
thwaites.uncertainty.error_correlation
======================================
Comprimento de correlação **do erro** de dh/dt — a escala que realmente governa
quantas amostras independentes existem ao integrar o balanço de massa.

Por que este módulo existe
--------------------------
`run_mass_balance` usava, como comprimento de correlação, o *range do variograma
do campo de dh/dt* (gravado em `interp_selection.json`). Isso confunde duas
grandezas distintas:

* o variograma do **sinal** descreve a estrutura glaciológica — costa afinando
  rápido, interior estável. Ele cresce quando o domínio cresce: ao expandir da
  Thwaites (93 mil km²) para o Amundsen (263 mil km²) o range foi de 41,8 km
  para 117,8 km, simplesmente porque o domínio maior contém mais do gradiente
  costa-interior;
* o que propaga na barra de erro é a correlação **do erro** (ruído do ATL06,
  resíduo de maré/DAC, amostragem temporal irregular). Não há razão física para
  ela acompanhar a extensão do domínio.

Usar o range do sinal levava a n_eff = A/(πL²) ≈ 6 amostras independentes em
todo o Amundsen, inflando σ(dM/dt) a 56% do próprio valor e derrubando o
resultado para ~1,8σ. Tratar todos os nós como independentes também é
indefensável e produz σ = 1,7 Gt/ano.

Estimadores implementados
-------------------------
`residual_correlation_length` — variograma dos RESÍDUOS da validação cruzada
espacial em blocos (predito − observado no nó deixado de fora). O resíduo é uma
realização do erro no local do nó, com o sinal de larga escala já removido pela
predição dos vizinhos.

`crossover_correlation_length` — variograma das diferenças de cruzamento, uma
medida de discrepância independente do interpolador.

Ambos devolvem também o variograma ajustado, para inspeção — nenhum número deste
módulo deve ser aceito sem olhar o ajuste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.interp.variogram import fit_variogram
from thwaites.logging import get_logger


def cv_residuals(nodes_df: pd.DataFrame, cfg: Config, method: str | None = None,
                 value_col: str = "dhdt", sigma_col: str = "dhdt_err") -> pd.DataFrame:
    """
    Resíduos da validação cruzada espacial em blocos, por nó.

    Cada nó é predito por um modelo treinado SEM o bloco que o contém, de modo
    que o resíduo (pred − obs) mede erro real, não ajuste em amostra.

    Devolve DataFrame com x, y, obs, pred, resid, var_pred.
    """
    from thwaites.interp.select import (PREDICTORS, _NEEDS_VARIOGRAM,
                                        spatial_block_folds)

    logger = get_logger()
    ic = cfg.interpolation
    method = method or ic.candidates[0]
    if method not in PREDICTORS:
        raise ValueError(f"método desconhecido: {method} (opções: {list(PREDICTORS)})")

    d = nodes_df[np.isfinite(nodes_df[value_col])]
    x = d["x"].to_numpy(float)
    y = d["y"].to_numpy(float)
    v = d[value_col].to_numpy(float)
    sig = d[sigma_col].to_numpy(float) if sigma_col in d else np.full_like(v, np.nan)

    fold = spatial_block_folds(x, y, ic.cv.block_km * 1000.0, ic.cv.n_folds, ic.cv.seed)
    parts = []
    for f in range(ic.cv.n_folds):
        test = fold == f
        train = ~test
        if test.sum() == 0 or train.sum() < max(ic.neighbors, 10):
            continue
        vparams = None
        if method in _NEEDS_VARIOGRAM:
            try:
                vparams = fit_variogram(x[train], y[train], v[train],
                                        n_lags=ic.variogram.n_lags,
                                        max_lag=ic.variogram.max_lag_m,
                                        seed=ic.cv.seed)
            except Exception as e:
                logger.warning(f"variograma falhou no fold {f} ({e}) — pulado.")
                continue
        p, va = PREDICTORS[method](x[train], y[train], v[train], sig[train],
                                   x[test], y[test], cfg, vparams)
        parts.append(pd.DataFrame({"x": x[test], "y": y[test], "obs": v[test],
                                   "pred": p, "var_pred": va}))

    if not parts:
        raise RuntimeError("CV não produziu nenhum fold válido.")
    out = pd.concat(parts, ignore_index=True)
    out["resid"] = out["pred"] - out["obs"]
    return out


def _range_from_variogram(x, y, values, cfg: Config, label: str) -> dict:
    """Ajusta o variograma e devolve range + diagnóstico, sem aceitar cegamente."""
    logger = get_logger()
    ic = cfg.interpolation
    vp = fit_variogram(x, y, values,
                       n_lags=ic.variogram.n_lags,
                       max_lag=ic.variogram.max_lag_m,
                       seed=ic.cv.seed)
    rng = float(vp["range_m"])
    nug = float(vp.get("nugget", np.nan))
    sill = float(vp.get("sill", np.nan))
    # fração de nugget: quanto do erro é puramente local (descorrelacionado).
    # Nugget alto => a correlação espacial do erro é fraca, e um L grande seria
    # uma leitura equivocada do ajuste.
    nug_frac = float(nug / sill) if np.isfinite(sill) and sill > 0 else float("nan")
    logger.info(f"variograma [{label}]: modelo {vp.get('model')} | "
                f"range {rng/1000:.1f} km | nugget {nug:.4f} | sill {sill:.4f} | "
                f"nugget/sill {nug_frac:.2f}")
    return {"range_m": rng, "nugget": nug, "sill": sill,
            "nugget_fraction": nug_frac, "model": vp.get("model"),
            "n_samples": int(np.size(values)), "variogram": vp}


def residual_correlation_length(nodes_df: pd.DataFrame, cfg: Config,
                                method: str | None = None) -> dict:
    """
    Comprimento de correlação estimado dos RESÍDUOS da CV em blocos.

    É o estimador preferido para propagar σ(dM/dt): mede a escala em que os
    ERROS se repetem, não a escala do sinal glaciológico.

    Cuidado de interpretação (declarado no retorno): a CV em blocos remove a
    estrutura até o tamanho do bloco, então este estimador NÃO enxerga
    correlação de erro em escala maior que `cv.block_km`. É um limite inferior
    para a escala de correlação do erro, e portanto um limite INFERIOR para σ.
    """
    res = cv_residuals(nodes_df, cfg, method=method)
    out = _range_from_variogram(res["x"].to_numpy(), res["y"].to_numpy(),
                                res["resid"].to_numpy(), cfg, "resíduos CV")
    out["estimator"] = "cv_residuals"
    out["method"] = method or cfg.interpolation.candidates[0]
    out["resid_rms"] = float(np.sqrt(np.mean(res["resid"] ** 2)))
    out["block_km"] = float(cfg.interpolation.cv.block_km)
    out["caveat"] = (
        "a CV em blocos remove estrutura acima do tamanho do bloco "
        f"({cfg.interpolation.cv.block_km:.0f} km); este range é um limite "
        "INFERIOR da correlação do erro e, portanto, sigma(dM/dt) daqui é um "
        "limite inferior")
    return out


def crossover_correlation_length(xovers: pd.DataFrame, cfg: Config,
                                 value_col: str = "dhdt") -> dict:
    """
    Comprimento de correlação das diferenças de cruzamento.

    Independe do interpolador, mas herda o caveat já registrado no projeto:
    crossovers usam o MESMO produto ATL06 e as mesmas correções, logo não são
    uma fonte externa — medem consistência interna de método.
    """
    d = xovers[np.isfinite(xovers[value_col])]
    if len(d) < 50:
        raise ValueError(f"apenas {len(d)} cruzamentos válidos — insuficiente.")
    v = d[value_col].to_numpy(float)
    # remove a média: o variograma deve ver a flutuação, não o nível
    out = _range_from_variogram(d["x"].to_numpy(), d["y"].to_numpy(),
                                v - np.mean(v), cfg, "cruzamentos")
    out["estimator"] = "crossovers"
    out["caveat"] = ("crossovers usam o mesmo produto ATL06 e as mesmas "
                     "correções — consistência interna, não validação externa")
    return out


def compare_correlation_lengths(nodes_df: pd.DataFrame, cfg: Config,
                                xovers: pd.DataFrame | None = None,
                                signal_range_m: float | None = None) -> dict:
    """
    Reúne as estimativas disponíveis lado a lado.

    Não escolhe sozinho um valor "certo": as escalas medem coisas diferentes e a
    escolha é uma decisão metodológica que deve ser explícita no texto. Devolve
    também o n_eff e o σ implicados por cada uma, que é o que torna a diferença
    concreta.
    """
    out = {"estimates": {}}
    try:
        out["estimates"]["cv_residuals"] = residual_correlation_length(nodes_df, cfg)
    except Exception as e:                                  # pragma: no cover
        out["estimates"]["cv_residuals"] = {"error": f"{type(e).__name__}: {e}"}
    if xovers is not None:
        try:
            out["estimates"]["crossovers"] = crossover_correlation_length(xovers, cfg)
        except Exception as e:                              # pragma: no cover
            out["estimates"]["crossovers"] = {"error": f"{type(e).__name__}: {e}"}
    if signal_range_m is not None:
        out["estimates"]["signal_field"] = {
            "range_m": float(signal_range_m),
            "estimator": "variograma do campo de dh/dt",
            "caveat": ("mede a estrutura do SINAL glaciológico, não do erro; "
                       "cresce com o tamanho do domínio — usar isto para "
                       "propagar erro superestima a correlação"),
        }
    return out
