"""
thwaites.timeseries.aliasing
============================
Quanto de um sinal PERIÓDICO vaza para a tendência estimada, dada a amostragem
temporal real dos dados?

Motivação
---------
A base é só JJA (meses 6-8). O ciclo anual é não-identificável nessa janela
(§4.2), então o modelo de produção não o ajusta — ele simplesmente não existe
no espaço de colunas. Um sinal periódico que o modelo não representa não
desaparece: ele se projeta sobre as colunas que existem, e a coluna `dt` é uma
delas. O resultado é um dh/dt espúrio.

Isso torna a afirmação "dh/dt só de inverno é estimativa conservadora do
derretimento anual" uma PREMISSA. Este módulo a converte em número medido.

Método
------
Segue o teste de resposta harmônica do ATBD ATL14/ATL15 r005 (§5): toma-se a
amostragem temporal REAL (as épocas que os dados de fato têm), substitui-se a
altura por um harmônico sintético de amplitude unitária, e roda-se o MESMO
estimador de produção. Como o harmônico não tem tendência, qualquer dh/dt
recuperado é vazamento puro.

Para um sinal de fase arbitrária,

    A·cos(2πt/P + φ) = A·cos(φ)·cos(2πt/P) − A·sin(φ)·sin(2πt/P)

de modo que, pela linearidade dos mínimos quadrados, a resposta em fase φ é
uma combinação das respostas às duas fases de referência. Logo:

    R = sqrt(b_cos² + b_sin²)     -> vazamento de PIOR CASO, por unidade de
                                     amplitude do sinal  [ (m/ano) / m ]
    R/sqrt(2)                     -> valor RMS sobre fase uniformemente sorteada

O viés em metros por ano é `R · A_sazonal`, com `A_sazonal` medido no GSFC-FDM
(ver `seasonal_amplitude`), não suposto.

Fidelidade
----------
`node_response` chama `_build_A`/`_lstsq_iter` de `thwaites.timeseries.dhdt` —
as mesmas funções da produção, inclusive a rejeição iterativa por MAD. Se o
estimador de produção rejeita pontos, o teste rejeita também, e o vazamento
medido é o do estimador que realmente
produziu os nossos números.
"""

from __future__ import annotations

import warnings

import numpy as np

from thwaites.timeseries.dhdt import _build_A, _lstsq_iter


def synthetic_harmonic(t: np.ndarray, period: float, phase: str) -> np.ndarray:
    """Harmônico de amplitude unitária nas épocas `t` (anos decimais)."""
    ang = 2.0 * np.pi * t / period
    if phase == "cos":
        return np.cos(ang)
    if phase == "sin":
        return np.sin(ang)
    raise ValueError(f"fase deve ser 'cos' ou 'sin', não {phase!r}")


def stretch_to_full_year(t: np.ndarray) -> np.ndarray:
    """
    Contrafactual de amostragem: as MESMAS observações, redistribuídas sobre o
    ano inteiro.

    A parte fracionária observada ocupa apenas [frac_min, frac_max] (JJA, ~0,42
    a 0,67). O remapeamento é linear e MONÓTONO, então preserva a estrutura de
    rajada das passagens (observações do mesmo sobrevoo continuam juntas) e o
    número de observações por ano — só amplia a cobertura de fase.

    NÃO é uma simulação de como o ICESat-2 amostraria o ano inteiro: é um
    contrafactual sobre as observações que temos, para isolar o efeito da
    COBERTURA DE FASE mantendo tudo o mais constante. Essa distinção precisa
    aparecer em qualquer texto que use o resultado.
    """
    yr = np.floor(t)
    frac = t - yr
    lo, hi = float(frac.min()), float(frac.max())
    if hi - lo < 1e-9:
        return t.copy()
    return yr + (frac - lo) / (hi - lo) * 0.999


def node_response(x, y, t, s, x0, y0, d, period: float) -> dict | None:
    """
    Vazamento de um harmônico de período `period` para o dh/dt, num nó.

    Roda o estimador de produção duas vezes (fases cos e sin) sobre a MESMA
    geometria e as MESMAS épocas, trocando só a altura pelo sinal sintético.

    Devolve dict com `b_cos`, `b_sin` e `R` (vazamento de pior caso, em
    (m/ano) por metro de amplitude), ou None se o nó não é ajustável.
    """
    n = len(t)
    if n < d.min_points:
        return None
    if float(t.max() - t.min()) < d.dt_min_years:
        return None

    dx, dy, dt = x - x0, y - y0, t - d.t_ref
    if d.use_weights:
        sv = np.where(s <= 0,
                      np.median(s[s > 0]) if (s > 0).any() else 0.05, s)
        wc = 1.0 / (sv ** 2 + 1e-12)
    else:
        wc = None

    A = _build_A(dx, dy, dt, d.poly_order, d.temp_order)

    out = {}
    for phase in ("cos", "sin"):
        z = synthetic_harmonic(t, period, phase)
        xhat, _, mask, _ = _lstsq_iter(A, z, wc, d.max_iter, d.n_sigma,
                                       d.resid_limit)
        if xhat is None or mask.sum() < d.min_points:
            return None
        out[f"b_{phase}"] = float(xhat[1])

    out["R"] = float(np.hypot(out["b_cos"], out["b_sin"]))
    out["nobs"] = int(n)
    return out


def jja_vs_annual_trend(h_a: np.ndarray, t: np.ndarray,
                        frac_lo: float, frac_hi: float,
                        whole_years: bool = True) -> dict:
    """
    Teste PAREADO direto: a tendência de `h_a` estimada só na janela JJA difere
    da estimada com o ano inteiro?

    Este teste responde a uma pergunta DIFERENTE da do harmônico. O harmônico
    mede vazamento de um ciclo estacionário para a tendência (aliasing). Aqui a
    pergunta é se dh/dt restrito ao inverno constitui uma estimativa
    conservadora da tendência anual; ela envolve o sinal real, incluindo
    a parte que a janela JJA simplesmente não observa (fusão de verão) e a
    variação interanual da amplitude sazonal.

    Só é possível porque o GSFC-FDM tem cobertura de ano inteiro. Vale para a
    componente de firn+SMB da altura de superfície — NÃO para a componente
    dinâmica, que o FDM não modela.

    `frac_lo`/`frac_hi` devem ser a janela de fase REALMENTE observada nos
    nossos dados, não a definição de calendário de JJA.

    Sinal: `diff = slope_jja − slope_anual` POSITIVO significa que JJA dá uma
    tendência menos negativa, isto é, subestima a perda — "conservador".
    """
    t = np.asarray(t, float)
    if whole_years:
        # Anos incompletos nas pontas enviesariam a comparação: a série do FDM
        # termina em junho, então incluir 2022 daria ao ajuste "anual" só a
        # metade fria do ano — deixaria de ser anual.
        #
        # O critério é de COBERTURA DE FASE, não `ceil`/`floor` da data: 2019
        # começa em 2019,007 e é um ano essencialmente completo; descartá-lo
        # por arredondamento jogaria fora um terço da série.
        yr = np.floor(t).astype(int)
        keep_years = []
        for y in np.unique(yr):
            fr = t[yr == y] - y
            if fr.min() <= 0.02 and fr.max() >= 0.98:
                keep_years.append(y)
        if len(keep_years) < 2:
            raise ValueError(
                f"menos de 2 anos com cobertura de fase completa em "
                f"{t.min():.3f}-{t.max():.3f}")
        span = np.isin(yr, keep_years)
    else:
        span = np.ones(t.size, bool)

    frac = t - np.floor(t)
    win = span & (frac >= frac_lo) & (frac <= frac_hi)
    if win.sum() < 8:
        raise ValueError(f"janela {frac_lo:.2f}-{frac_hi:.2f} tem {win.sum()} épocas")

    nt = h_a.shape[0]
    flat = np.asarray(h_a, float).reshape(nt, -1)
    good = np.isfinite(flat).all(axis=0)
    # Célula constante no tempo não carrega informação de tendência e é o
    # assinante típico do valor de preenchimento que escapou da máscara.
    # Incluí-la força slope=0 e ARRASTA A MEDIANA para zero.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # colunas todo-NaN
        good &= np.nanstd(flat, axis=0) > 1e-9

    def slopes(sel):
        ts = t[sel]
        G = np.column_stack([np.ones(ts.size), ts - ts.mean()])
        coef, *_ = np.linalg.lstsq(G, flat[np.ix_(sel, good)], rcond=None)
        return coef[1]

    s_ann = slopes(span)
    s_jja = slopes(win)
    diff = s_jja - s_ann

    return {
        "n_celulas": int(good.sum()),
        "n_celulas_descartadas": int(flat.shape[1] - good.sum()),
        "n_epocas_anual": int(span.sum()),
        "n_epocas_jja": int(win.sum()),
        "anos": [float(np.floor(t[span].min())), float(np.ceil(t[span].max()))],
        "janela_fase": [float(frac_lo), float(frac_hi)],
        "slope_anual_mediana": float(np.median(s_ann)),
        "slope_jja_mediana": float(np.median(s_jja)),
        "diff_mediana": float(np.median(diff)),
        "diff_p10": float(np.percentile(diff, 10)),
        "diff_p90": float(np.percentile(diff, 90)),
        "frac_celulas_jja_conservador": float(np.mean(diff > 0)),
    }


def seasonal_amplitude(h_a: np.ndarray, t: np.ndarray,
                       period: float = 1.0) -> np.ndarray:
    """
    Amplitude do harmônico de período `period` numa série de altura modelada.

    `h_a` tem forma (tempo, y, x) e `t` está em anos decimais. Ajusta, por
    célula, `h = c0 + c1·t + a·cos(2πt/P) + b·sin(2πt/P)` e devolve
    `sqrt(a²+b²)` — a amplitude, em metros.

    A tendência entra no ajuste de propósito: sem ela, a deriva de longo prazo
    da coluna de firn seria absorvida pelo harmônico e inflaria a amplitude.
    """
    t = np.asarray(t, float)
    ang = 2.0 * np.pi * t / period
    G = np.column_stack([np.ones_like(t), t - t.mean(),
                         np.cos(ang), np.sin(ang)])

    nt = h_a.shape[0]
    flat = np.asarray(h_a, float).reshape(nt, -1)
    good = np.isfinite(flat).all(axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # colunas todo-NaN
        good &= np.nanstd(flat, axis=0) > 1e-9   # ver nota em jja_vs_annual_trend

    amp = np.full(flat.shape[1], np.nan)
    if good.any():
        coef, *_ = np.linalg.lstsq(G, flat[:, good], rcond=None)
        amp[good] = np.hypot(coef[2], coef[3])
    return amp.reshape(h_a.shape[1:])
