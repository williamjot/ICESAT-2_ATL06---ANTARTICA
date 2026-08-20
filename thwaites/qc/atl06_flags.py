"""
thwaites.qc.atl06_flags
=======================
Filtro de qualidade baseado nos flags nativos do ATL06.

Significado dos flags — LIDO dos atributos do próprio produto (`description`,
`flag_values`, `flag_meanings`), não presumido:

`atl06_quality_summary`  0 = best_quality, 1 = potential_problem.
    A descrição oficial adverte: "Users who select only segments with zero
    values for this flag can be relatively certain of obtaining high-quality
    data, but will likely miss a significant [fraction]". É conservador.
    ATENÇÃO: é BINÁRIO. A config antiga usava `quality_max: 1`, que mantém
    tudo — o filtro nunca filtrou nada.

`h_robust_sprd`  RDE do desajuste entre as alturas dos fótons e o ajuste do
    segmento, em metros. Alto = superfície rugosa (crevasses) ou ruído. É o
    discriminante mais útil em geleira de fluxo rápido, onde o fraturamento
    degrada o ajuste.

`snr_significance`  probabilidade de a rotina de busca de sinal convergir ao
    SNR observado a partir de ruído aleatório. VALOR PEQUENO É BOM (baixa
    chance de blunder de detecção de superfície) — o filtro é um limite
    SUPERIOR, não inferior.

`n_fit_photons`  nº de fótons usados no ajuste. Poucos = estimativa frágil.

`msw_flag`  -1 cannot_determine | 0 no_layers | 1 layer_gt_3km |
    2 layer_between_1_and_3_km | 3 layer_lt_1km | 4 blow_snow_od_lt_0.5 |
    5 blow_snow_od_gt_0.5.
    ATENÇÃO: `msw == 1` é nuvem ACIMA de 3 km, praticamente inofensiva para a
    medida de superfície. Exigir `msw == 0` descartaria quase todo o dado sem
    ganho — o corte útil está nas camadas baixas e na neve soprada (>= 3).

`cloud_flg_asr`  0..2 = clear (alta/média/baixa confiança),
    3..5 = cloudy (baixa/média/alta). Derivado da refletância SOLAR aparente.
    VER `solar_elevation`: sob noite polar (JJA na Antártica) este flag pode
    não ser informativo, e aplicá-lo às cegas descartaria a maior parte da
    amostra por um motivo que não se sustenta.

`bsnow_conf`  confiança de neve soprada. Valores negativos indicam ausência de
    detecção/baixa confiança; positivos, detecção com confiança crescente.

`dem_h`  altura do DEM de referência embutido no produto. |h_li - dem_h| grande
    é um detector de blunder independente do nosso REMA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Nomes das colunas produzidas por pipelines/fetch_qc_flags.py
FLAG_COLUMNS = [
    "atl06_quality_summary", "segment_id", "sigma_geo_h",
    "h_robust_sprd", "n_fit_photons", "dh_fit_dx", "h_expected_rms",
    "snr", "snr_significance", "w_surface_window_final",
    "bsnow_conf", "bsnow_od", "bsnow_h", "cloud_flg_asr", "cloud_flg_atm",
    "msw_flag", "r_eff", "solar_elevation", "dem_h", "dem_flag",
]


def quality_mask(qc: pd.DataFrame, cfg, h_li: np.ndarray | None = None):
    """
    Máscara booleana de pontos que passam no controle de qualidade.

    Devolve `(keep, reasons)`, onde `reasons` conta quantos pontos CADA critério
    reprova isoladamente (não de forma exclusiva) — assim é possível ver qual
    critério domina a rejeição em vez de só o total.

    Todo critério é opcional: um limiar `None` na config desliga aquele teste,
    e isso fica registrado no relatório. Critérios desligados não são omissão —
    são decisão declarada.
    """
    q = cfg.atl06_qc
    n = len(qc)
    keep = np.ones(n, dtype=bool)
    reasons: dict[str, int] = {}

    def apply(name: str, bad: np.ndarray):
        bad = np.asarray(bad, dtype=bool)
        reasons[name] = int(bad.sum())
        keep[bad] = False

    if q.require_quality_summary_zero and "atl06_quality_summary" in qc:
        apply("atl06_quality_summary != 0",
              qc["atl06_quality_summary"].to_numpy() != 0)

    if q.max_h_robust_sprd_m is not None and "h_robust_sprd" in qc:
        v = qc["h_robust_sprd"].to_numpy()
        apply(f"h_robust_sprd > {q.max_h_robust_sprd_m}",
              np.isfinite(v) & (v > q.max_h_robust_sprd_m))

    if q.max_snr_significance is not None and "snr_significance" in qc:
        v = qc["snr_significance"].to_numpy()
        apply(f"snr_significance > {q.max_snr_significance}",
              np.isfinite(v) & (v > q.max_snr_significance))

    if q.min_n_fit_photons is not None and "n_fit_photons" in qc:
        v = qc["n_fit_photons"].to_numpy()
        apply(f"n_fit_photons < {q.min_n_fit_photons}",
              np.isfinite(v) & (v < q.min_n_fit_photons))

    if q.max_msw_flag is not None and "msw_flag" in qc:
        v = qc["msw_flag"].to_numpy()
        # -1 = cannot_determine: não é evidência de problema, não rejeita
        apply(f"msw_flag > {q.max_msw_flag}", np.isfinite(v) & (v > q.max_msw_flag))

    if q.max_cloud_flg_asr is not None and "cloud_flg_asr" in qc:
        v = qc["cloud_flg_asr"].to_numpy()
        apply(f"cloud_flg_asr > {q.max_cloud_flg_asr}",
              np.isfinite(v) & (v > q.max_cloud_flg_asr))

    if q.max_bsnow_conf is not None and "bsnow_conf" in qc:
        v = qc["bsnow_conf"].to_numpy()
        apply(f"bsnow_conf > {q.max_bsnow_conf}",
              np.isfinite(v) & (v > q.max_bsnow_conf))

    if (q.max_dem_diff_m is not None and "dem_h" in qc and h_li is not None):
        d = np.abs(np.asarray(h_li, dtype=float) - qc["dem_h"].to_numpy())
        apply(f"|h_li - dem_h| > {q.max_dem_diff_m}",
              np.isfinite(d) & (d > q.max_dem_diff_m))

    return keep, reasons


def summarize_flags(qc: pd.DataFrame) -> dict:
    """Distribuições dos flags, para escolher limiares com base em evidência."""
    out: dict = {"n": int(len(qc))}
    for c in ("atl06_quality_summary", "msw_flag", "cloud_flg_asr",
              "cloud_flg_atm", "bsnow_conf", "dem_flag"):
        if c in qc:
            v, k = np.unique(qc[c].to_numpy(), return_counts=True)
            out[c] = {int(a): int(b) for a, b in zip(v, k)}
    for c in ("h_robust_sprd", "snr_significance", "n_fit_photons",
              "w_surface_window_final", "solar_elevation", "bsnow_od",
              "h_expected_rms", "dh_fit_dx"):
        if c in qc:
            v = qc[c].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                out[c] = {
                    "p05": float(np.percentile(v, 5)),
                    "p50": float(np.percentile(v, 50)),
                    "p95": float(np.percentile(v, 95)),
                    "p99": float(np.percentile(v, 99)),
                    "max": float(v.max()),
                }
    return out
