"""
thwaites.corrections.firn
=========================
Correção de firn: separa mudança de ALTURA de mudança de MASSA.

FÍSICA
------
A altimetria mede a altura da superfície. Parte da variação não envolve massa
nenhuma: o firn (neve em compactação) muda de volume ao densificar. O ar sai da
coluna, a superfície baixa, e **nenhuma massa é perdida**.

A decomposição padrão (IMBIE; Smith et al. 2020) é:

    dh/dt = dh_gelo/dt + dFAC/dt

    FAC = firn air content [m] — a espessura equivalente de AR na coluna de firn

Logo, a altura equivalente em gelo e a massa são:

    dh_gelo/dt = dh/dt − dFAC/dt
    dM/dt      = ρ_gelo · dh_gelo/dt · A

Usar ρ=917 sobre o dh/dt bruto atribui a compactação de firn a perda de gelo.
O sinal do viés depende do sinal de
dFAC/dt: se o firn está compactando (dFAC/dt < 0), a superfície baixa sem perda
de massa e o balanço SUPERESTIMA a perda.

LIMITAÇÃO DE COBERTURA (declarar sempre)
----------------------------------------
O GSFC-FDM v1.2.1 termina em 30/06/2022 e o dh/dt do projeto vai a 2025. A taxa
de FAC é estimada na sobreposição (~3,5 de 7 invernos) e **extrapolada** para o
resto. `firn_rate_at` devolve a fração de cobertura temporal junto com a taxa,
para que essa extrapolação não passe silenciosa.

Se o dado não estiver disponível, `firn_sensitivity` quantifica o efeito de uma
FAIXA plausível de dFAC/dt no resultado final — transformar um viés não
quantificado numa faixa quantificada já é um ganho de rigor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger

# nomes candidatos da variável de firn air content no NetCDF do FDM
_FAC_CANDIDATES = ("FAC", "fac", "firn_air_content", "FirnAirContent")


def resolve_firn_path(cfg: Config) -> Path:
    p = Path(cfg.firn.path)
    return p if p.is_absolute() else cfg.paths.data_dir / p


def load_firn(cfg: Config, path=None):
    """
    Abre o recorte do FDM e devolve (x, y, time, fac).

    Não assume o nome da variável: tenta os candidatos e, se falhar, levanta
    erro LISTANDO o que existe no arquivo (em vez de adivinhar).
    """
    import xarray as xr

    p = Path(path) if path else resolve_firn_path(cfg)
    if not p.exists():
        raise FileNotFoundError(
            f"Recorte do FDM não encontrado: {p}\n"
            f"Rode pipelines/fetch_firn.py (GSFC-FDM v1.2.1, Zenodo 7221954).")
    ds = xr.open_dataset(p)

    name = cfg.firn.fac_var or next(
        (c for c in _FAC_CANDIDATES if c in ds.data_vars), None)
    if name is None or name not in ds.data_vars:
        avail = list(ds.data_vars)
        ds.close()
        raise ValueError(
            f"variável de FAC não encontrada em {p.name}. Disponíveis: {avail}. "
            f"Defina `firn.fac_var` na config.")

    xn = "x" if "x" in ds.coords else None
    yn = "y" if "y" in ds.coords else None
    if xn is None or yn is None:
        avail = list(ds.coords)
        ds.close()
        raise ValueError(f"coords x/y ausentes em {p.name}. Disponíveis: {avail}")

    fac = ds[name]
    x = np.asarray(ds[xn].values, dtype=float)
    y = np.asarray(ds[yn].values, dtype=float)
    t = ds["time"].values if "time" in ds.coords else None
    arr = np.asarray(fac.values, dtype=float)
    ds.close()
    return x, y, t, arr, name


def _decimal_year(times) -> np.ndarray:
    """
    Normaliza o eixo temporal para ANO DECIMAL.

    O GSFC-FDMv1.2.1 já traz `time` em anos decimais (units: "decimal years",
    1980,007 → 2022,484), então o caminho normal é passar direto. O ramo
    datetime64 fica para outros modelos de firn com calendário convencional.
    """
    arr = np.asarray(times)
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
        return arr.astype(float)          # já em anos decimais
    t = pd.to_datetime(arr)
    return (t.year + (t.dayofyear - 1) / 365.25).to_numpy(dtype=float)


def firn_rate_field(cfg: Config, path=None):
    """
    Taxa dFAC/dt [m/ano] por célula, por regressão linear no tempo.

    Retorna (x, y, rate_2d, info) com `info` incluindo a fração do período do
    projeto efetivamente coberta pelo modelo — a métrica que expõe a
    extrapolação.
    """
    logger = get_logger()
    x, y, times, fac, var = load_firn(cfg, path)
    if times is None or fac.ndim != 3:
        raise ValueError(f"esperava FAC com dimensão temporal; obtive shape {fac.shape}")

    ty = _decimal_year(times)
    y0, y1 = cfg.temporal.year_start, cfg.temporal.year_end + 1
    sel = (ty >= y0) & (ty < y1)
    if sel.sum() < 4:
        raise ValueError(f"sobreposição temporal insuficiente: {int(sel.sum())} épocas "
                         f"entre {y0} e {y1}")
    tt = ty[sel]
    ff = fac[sel, :, :]

    # regressão linear por célula (vetorizada)
    tc = tt - tt.mean()
    denom = float(np.sum(tc ** 2))
    rate = np.einsum("t,tij->ij", tc, ff - ff.mean(axis=0, keepdims=True)) / denom

    covered = (tt.max() - tt.min())
    requested = (cfg.temporal.year_end + 1) - cfg.temporal.year_start
    info = {
        "fac_variable": var,
        "n_epochs_used": int(sel.sum()),
        "model_period": [float(tt.min()), float(tt.max())],
        "project_period": [float(y0), float(y1)],
        "temporal_coverage_frac": float(covered / requested),
        "extrapolated": bool(tt.max() < cfg.temporal.year_end),
    }
    logger.info(
        f"dFAC/dt: {info['n_epochs_used']} épocas, {tt.min():.2f}–{tt.max():.2f} | "
        f"cobertura do período do projeto {100*info['temporal_coverage_frac']:.0f}% | "
        f"mediana {np.nanmedian(rate):+.4f} m/ano")
    if info["extrapolated"]:
        logger.warning(
            f"O FDM termina em {tt.max():.2f} e o projeto vai a "
            f"{cfg.temporal.year_end}. A taxa de FAC é EXTRAPOLADA — declarar.")
    return x, y, rate, info


def firn_rate_at(px, py, cfg: Config, path=None):
    """Amostra dFAC/dt [m/ano] nos pontos (x, y) EPSG:3031 (bilinear)."""
    from scipy.interpolate import RegularGridInterpolator

    x, y, rate, info = firn_rate_field(cfg, path)
    if y[0] > y[-1]:
        y = y[::-1]
        rate = rate[::-1, :]
    interp = RegularGridInterpolator((y, x), rate, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    pts = np.c_[np.asarray(py, float), np.asarray(px, float)]
    return interp(pts), info


def apply_firn_correction(grid_df: pd.DataFrame, cfg: Config,
                          value_col: str = "pred", path=None) -> tuple[pd.DataFrame, dict]:
    """
    Adiciona `dfac_dt` e `dhdt_ice` = dh/dt − dFAC/dt ao mapa interpolado.

    Onde o FDM não cobre, `dfac_dt` fica NaN e `dhdt_ice` cai para o dh/dt bruto
    (equivalente a assumir dFAC/dt = 0 ali) — registrado em `info`.
    """
    logger = get_logger()
    out = grid_df.copy()
    rate, info = firn_rate_at(out["x"].to_numpy(), out["y"].to_numpy(), cfg, path)
    out["dfac_dt"] = rate
    ok = np.isfinite(rate)
    out["dhdt_ice"] = out[value_col].to_numpy(float) - np.where(ok, rate, 0.0)

    info = dict(info)
    info["spatial_coverage_frac"] = float(np.mean(ok))
    info["dfac_dt_median"] = float(np.nanmedian(rate)) if ok.any() else None
    info["dhdt_raw_median"] = float(np.nanmedian(out[value_col]))
    info["dhdt_ice_median"] = float(np.nanmedian(out["dhdt_ice"]))
    logger.info(
        f"correção de firn: cobertura espacial {100*info['spatial_coverage_frac']:.0f}% | "
        f"dh/dt bruto {info['dhdt_raw_median']:+.4f} -> gelo-equivalente "
        f"{info['dhdt_ice_median']:+.4f} m/ano")
    return out, info


# ------------------------------------------------------------------- SMB
SMB_ANOMALY_CAVEAT = (
    "O GSFC-FDM fornece SMB_a = anomalia CUMULATIVA de SMB (m de gelo) "
    "relativa à climatologia de 1980-2019 — NÃO o SMB total. Sua derivada dá a "
    "TAXA DE ANOMALIA de SMB, não o SMB absoluto (que na Amundsen é ~0,3-1,0 m "
    "gelo/ano). Sem a climatologia de referência, a conservação de massa fecha "
    "em (ṁ_b − SMB_ref), não em ṁ_b.")


def smb_anomaly_rate_field(cfg: Config, path=None):
    """
    Taxa da ANOMALIA de SMB [m gelo/ano] por célula, do GSFC-FDM.

    ATENÇÃO (ver SMB_ANOMALY_CAVEAT): isto NÃO é o SMB total. `SMB_a` é uma
    anomalia cumulativa relativa à climatologia 1980-2019; derivando obtém-se
    quanto o SMB se afastou dessa referência, não o seu valor absoluto.
    """
    import xarray as xr

    logger = get_logger()
    p = Path(path) if path else resolve_firn_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} não existe (rode pipelines/fetch_firn.py).")
    ds = xr.open_dataset(p, decode_times=False)
    var = cfg.firn.smb_var
    if var not in ds.data_vars:
        avail = list(ds.data_vars)
        ds.close()
        raise ValueError(f"'{var}' ausente em {p.name}. Disponíveis: {avail}. "
                         f"Defina `firn.smb_var`.")
    ty = _decimal_year(ds["time"].values)
    arr = np.asarray(ds[var].values, dtype=float)
    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    ds.close()

    y0, y1 = cfg.temporal.year_start, cfg.temporal.year_end + 1
    sel = (ty >= y0) & (ty < y1)
    if sel.sum() < 4:
        raise ValueError(f"sobreposição temporal insuficiente: {int(sel.sum())} épocas")
    tt, ff = ty[sel], arr[sel, :, :]
    tc = tt - tt.mean()
    rate = np.einsum("t,tij->ij", tc, ff - ff.mean(axis=0, keepdims=True)) / float(np.sum(tc ** 2))

    info = {"smb_variable": var, "is_anomaly": True,
            "caveat": SMB_ANOMALY_CAVEAT,
            "n_epochs_used": int(sel.sum()),
            "model_period": [float(tt.min()), float(tt.max())],
            "rate_median_m_ice_yr": float(np.nanmedian(rate))}
    logger.info(f"taxa de ANOMALIA de SMB: mediana {info['rate_median_m_ice_yr']:+.4f} "
                f"m gelo/ano ({info['n_epochs_used']} épocas)")
    logger.warning(SMB_ANOMALY_CAVEAT)
    return x, y, rate, info


def resolve_smb_path(cfg: Config) -> Path:
    p = Path(cfg.firn.smb_path)
    return p if p.is_absolute() else cfg.paths.data_dir / p


def smb_total_field(cfg: Config, path=None):
    """
    SMB TOTAL médio [m gelo/ano] por célula — o termo que a conservação de
    massa realmente pede.

    DIFERENÇA CRÍTICA para `smb_anomaly_rate_field`: o arquivo de componentes do
    GSFC-FDM traz `SMB` já como TAXA ("meters of ice equivalent per year",
    definida como Snowfall−Sublimation + Rainfall − Runoff). Portanto aqui se
    tira a MÉDIA temporal — derivar seria errado. A variável `SMB_a` do outro
    arquivo é uma anomalia CUMULATIVA e exige derivada; confundir as duas
    trocaria ~1 m/ano por ~0,09 m/ano.
    """
    import xarray as xr

    logger = get_logger()
    p = Path(path) if path else resolve_smb_path(cfg)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} não existe. Rode: python pipelines/fetch_firn.py --which smb")
    ds = xr.open_dataset(p, decode_times=False)
    var = cfg.firn.smb_total_var
    if var not in ds.data_vars:
        avail = list(ds.data_vars)
        ds.close()
        raise ValueError(f"'{var}' ausente em {p.name}. Disponíveis: {avail}. "
                         f"Defina `firn.smb_total_var`.")
    units = str(ds[var].attrs.get("units", "")).lower()
    if "per year" not in units and "yr" not in units:
        logger.warning(f"unidade de {var} = '{units}' — esperava taxa por ano. "
                       f"Se for cumulativo, a média temporal está ERRADA.")

    ty = _decimal_year(ds["time"].values)
    arr = np.asarray(ds[var].values, dtype=float)
    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    ds.close()

    y0, y1 = cfg.temporal.year_start, cfg.temporal.year_end + 1
    sel = (ty >= y0) & (ty < y1)
    if sel.sum() < 4:
        raise ValueError(f"sobreposição temporal insuficiente: {int(sel.sum())} épocas")
    tt = ty[sel]
    mean = np.nanmean(arr[sel, :, :], axis=0)     # média da TAXA (não derivada)

    covered = tt.max() - tt.min()
    requested = y1 - y0
    info = {"smb_variable": var, "is_total": True, "units": units,
            "n_epochs_used": int(sel.sum()),
            "model_period": [float(tt.min()), float(tt.max())],
            "temporal_coverage_frac": float(covered / requested),
            "extrapolated": bool(tt.max() < cfg.temporal.year_end),
            "smb_median_m_ice_yr": float(np.nanmedian(mean))}
    logger.info(f"SMB TOTAL médio: mediana {info['smb_median_m_ice_yr']:+.4f} "
                f"m gelo/ano | {info['n_epochs_used']} épocas "
                f"({tt.min():.2f}–{tt.max():.2f})")
    return x, y, mean, info


def smb_total_at(px, py, cfg: Config, path=None):
    """Amostra o SMB total médio [m gelo/ano] nos pontos (x, y)."""
    from scipy.interpolate import RegularGridInterpolator

    x, y, mean, info = smb_total_field(cfg, path)
    if y[0] > y[-1]:
        y = y[::-1]
        mean = mean[::-1, :]
    interp = RegularGridInterpolator((y, x), mean, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    return interp(np.c_[np.asarray(py, float), np.asarray(px, float)]), info


def smb_anomaly_rate_at(px, py, cfg: Config, path=None):
    """Amostra a taxa de anomalia de SMB [m gelo/ano] nos pontos (x, y)."""
    from scipy.interpolate import RegularGridInterpolator

    x, y, rate, info = smb_anomaly_rate_field(cfg, path)
    if y[0] > y[-1]:
        y = y[::-1]
        rate = rate[::-1, :]
    interp = RegularGridInterpolator((y, x), rate, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    return interp(np.c_[np.asarray(py, float), np.asarray(px, float)]), info


# ---------------------------------------------------------------- sensibilidade
def firn_sensitivity(grid_df: pd.DataFrame, cfg: Config,
                     dfac_rates=(-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10),
                     value_col: str = "pred") -> pd.DataFrame:
    """
    Quanto uma FAIXA plausível de dFAC/dt muda o balanço de massa.

    Serve para quando o FDM não está disponível (ou para acompanhar a correção):
    converte um viés não quantificado numa faixa quantificada, que é declarável
    no artigo. Um dFAC/dt uniforme é uma simplificação grosseira — o objetivo é
    a ORDEM DE GRANDEZA do efeito, não um valor corrigido.
    """
    from thwaites.uncertainty.mass_balance import compute_mass_balance

    L = cfg.mass_balance.correlation_length_m or 20_000.0
    rows = []
    for r in dfac_rates:
        g = grid_df.copy()
        g[value_col] = g[value_col].to_numpy(float) - r
        res = compute_mass_balance(g, cfg, correlation_length_m=L,
                                   value_col=value_col)
        rows.append({"dfac_dt_m_yr": r,
                     "dMdt_Gt_yr": res["dMdt_Gt_yr"],
                     "sle_mm_yr": res["sle_mm_yr"],
                     "dhdt_ice_mean": res["dhdt_mean_m_yr"]})
    df = pd.DataFrame(rows)
    base = df.loc[df["dfac_dt_m_yr"] == 0.0, "dMdt_Gt_yr"]
    if len(base):
        df["delta_vs_zero_Gt_yr"] = df["dMdt_Gt_yr"] - float(base.iloc[0])
    return df
