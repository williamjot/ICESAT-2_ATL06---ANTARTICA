"""
thwaites.glaciology.advection
=============================
Termo de advecção e conversão Euleriano ↔ Lagrangiano do dh/dt.

FÍSICA
------
A derivada material (seguindo a parcela de gelo) e a derivada num ponto fixo se
relacionam por:

    Dh/Dt = ∂h/∂t + v·∇h
    └ Lagrangiano   └ Euleriano   └ termo de ADVECÇÃO

O ICESat-2 mede `∂h/∂t` (ponto fixo). O termo `v·∇h` é o quanto a topografia da
superfície "desfila" pelo ponto de medição enquanto o gelo escoa — no tronco da
Thwaites, a 2–4 km/ano sobre uma superfície inclinada, ele não é desprezível.

O QUE ESTE MÓDULO **NÃO** CORRIGE (importante)
----------------------------------------------
O balanço de massa do projeto usa a forma EULERIANA da conservação:

    ∂H/∂t + ∇·(H·v) = SMB − ṁ_b

Nela o termo ∇·(H·v) **já contabiliza a advecção**, e ∫(∂h/∂t)dA sobre uma
região É, por definição, a variação de volume daquela região. Ou seja: o número
em Gt/ano NÃO está contaminado por advecção, e aplicar a correção Lagrangiana
sobre ele seria contar o mesmo efeito duas vezes.

ONDE A ADVECÇÃO IMPORTA DE FATO
-------------------------------
  - interpretar o PADRÃO ESPACIAL de dh/dt como adelgaçamento dinâmico;
  - interpretar dh/dt PONTUAL como afinamento da coluna de gelo;
  - fechar a conservação na forma Lagrangiana (Moholdt et al. 2014; Shean et al.
    2019; Adusumilli et al. 2020), que evita o ∇H ruidoso de um DEM.

Por isso o produto principal aqui é a QUANTIFICAÇÃO do termo — quão grande ele é
em relação ao sinal — e não uma substituição do balanço de massa.

SENSIBILIDADE À SUAVIZAÇÃO
--------------------------
∇h de um DEM de 32 m é dominado por ruído e microtopografia. O gradiente precisa
ser suavizado à escala em que o dh/dt foi estimado (raio de busca ~15 km), senão
o termo de advecção vira ruído amplificado. `advection_sensitivity` varia essa
escala explicitamente, porque o resultado depende dela.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger


def surface_slope_from_rema(gx, gy, cfg: Config, smooth_km: float = 5.0,
                            tif_path=None, read_res_m: float = 500.0):
    """
    Gradiente da superfície (∂h/∂x, ∂h/∂y) numa grade regular, a partir do REMA.

    LEITURA DECIMADA: materializar a janela do REMA na resolução nativa de 32 m
    para a ROI + buffer exigiria ~284 milhões de pixels (2,3 GB em float64); as
    operações seguintes (preenchimento, dois `gaussian_filter`, `np.gradient`)
    elevariam o pico a ~14 GB numa máquina com ~3 GB livres.

    O absurdo era a resolução: o declive é usado numa escala de 5–15 km, sobre
    uma grade de dezenas de pontos. Ler a 32 m é ~1000× mais do que o cálculo
    aproveita. O rasterio decima durante a leitura (`out_shape`), então a
    resolução nativa nunca é materializada. `read_res_m=500` dá ~10-30 amostras
    por comprimento de suavização, folgado para estimar declive glaciológico.
    """
    import rasterio
    from rasterio.windows import from_bounds, Window
    from scipy.ndimage import gaussian_filter, map_coordinates
    from thwaites.corrections.slope import resolve_rema_path

    logger = get_logger()
    tif_path = tif_path or resolve_rema_path(cfg)
    gx = np.asarray(gx, float); gy = np.asarray(gy, float)
    pad = max(smooth_km * 4000.0, 20_000.0)

    with rasterio.open(tif_path) as src:
        win = from_bounds(gx.min() - pad, gy.min() - pad,
                          gx.max() + pad, gy.max() + pad, transform=src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height))
        native = float(abs(src.res[0]))
        # fator de decimação: nunca abaixo da resolução nativa
        step = max(int(round(read_res_m / native)), 1)
        out_h = max(int(win.height) // step, 2)
        out_w = max(int(win.width) // step, 2)
        logger.info(
            f"REMA: janela {int(win.height)}×{int(win.width)} @ {native:.0f} m "
            f"-> lida decimada em {out_h}×{out_w} @ ~{native*step:.0f} m "
            f"({out_h*out_w*8/1024**2:.0f} MB em vez de "
            f"{int(win.height)*int(win.width)*8/1024**3:.1f} GB)")
        band = src.read(1, window=win, out_shape=(out_h, out_w)).astype(np.float64)
        # transform da leitura DECIMADA (escala os pixels)
        tr = src.window_transform(win) * rasterio.Affine.scale(
            int(win.width) / out_w, int(win.height) / out_h)
        nodata = src.nodata
        res = native * (int(win.width) / out_w)

    if nodata is not None:
        band[band == nodata] = np.nan
    valid = np.isfinite(band)
    filled = np.where(valid, band, 0.0)
    sigma_px = max(smooth_km * 1000.0 / res, 1.0)
    num = gaussian_filter(filled, sigma=sigma_px)
    den = gaussian_filter(valid.astype(float), sigma=sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where(den > 1e-6, num / den, np.nan)

    # gradiente em m/m (eixo 0 = y decrescente no raster)
    dh_dy_px, dh_dx_px = np.gradient(smooth, res, res)
    dh_dy_px = -dh_dy_px          # raster: linha cresce para o sul

    inv = ~tr
    GX, GY = np.meshgrid(gx, gy)
    cols, rows = inv * (GX.ravel(), GY.ravel())
    coords = np.vstack([rows - 0.5, cols - 0.5])
    kw = dict(order=1, mode="constant", cval=np.nan)
    dhdx = map_coordinates(np.nan_to_num(dh_dx_px), coords, **kw).reshape(GX.shape)
    dhdy = map_coordinates(np.nan_to_num(dh_dy_px), coords, **kw).reshape(GX.shape)
    return dhdx, dhdy


def advection_term(gx, gy, cfg: Config, smooth_km: float = 5.0):
    """
    Termo de advecção v·∇h [m/ano] numa grade regular.

    Retorna (adv, info) com estatísticas — inclusive a razão entre o termo e o
    próprio sinal de dh/dt, que é o número que interessa.
    """
    from scipy.interpolate import RegularGridInterpolator
    from thwaites.glaciology.flux import load_velocity

    logger = get_logger()
    gx = np.asarray(gx, float); gy = np.asarray(gy, float)
    vx_x, vx_y, vx, vy = load_velocity(cfg)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.c_[GY.ravel(), GX.ravel()]
    kw = dict(method="linear", bounds_error=False, fill_value=np.nan)
    VX = RegularGridInterpolator((vx_y, vx_x), vx, **kw)(pts).reshape(GX.shape)
    VY = RegularGridInterpolator((vx_y, vx_x), vy, **kw)(pts).reshape(GX.shape)

    dhdx, dhdy = surface_slope_from_rema(gx, gy, cfg, smooth_km=smooth_km)
    adv = VX * dhdx + VY * dhdy

    fin = np.isfinite(adv)
    info = {
        "smooth_km": smooth_km,
        "n_valid": int(fin.sum()),
        "adv_median_m_yr": float(np.nanmedian(adv[fin])) if fin.any() else np.nan,
        "adv_abs_median_m_yr": float(np.nanmedian(np.abs(adv[fin]))) if fin.any() else np.nan,
        "adv_p90_abs_m_yr": float(np.nanpercentile(np.abs(adv[fin]), 90)) if fin.any() else np.nan,
        "speed_median_m_yr": float(np.nanmedian(np.hypot(VX, VY)[fin])) if fin.any() else np.nan,
        "slope_median": float(np.nanmedian(np.hypot(dhdx, dhdy)[fin])) if fin.any() else np.nan,
    }
    logger.info(
        f"advecção (suavização {smooth_km:.0f} km): |v·∇h| mediano "
        f"{info['adv_abs_median_m_yr']:.4f} m/ano (p90 {info['adv_p90_abs_m_yr']:.4f}) | "
        f"velocidade mediana {info['speed_median_m_yr']:.0f} m/ano | "
        f"declive mediano {info['slope_median']:.5f}")
    return adv, info


def to_lagrangian(dhdt_eulerian, advection):
    """
    Dh/Dt = ∂h/∂t + v·∇h

    NÃO use isto para recalcular o balanço de massa Euleriano — ver a nota no
    topo do módulo (contagem dupla da advecção).
    """
    return np.asarray(dhdt_eulerian, float) + np.asarray(advection, float)


def advection_sensitivity(gx, gy, cfg: Config,
                          smooth_scales=(2.0, 5.0, 10.0, 15.0)) -> pd.DataFrame:
    """
    Como o termo de advecção depende da escala de suavização do declive.

    Se |v·∇h| variar muito entre escalas, o termo é dominado por ruído do DEM e
    não deve ser usado como correção — só como ordem de grandeza.
    """
    rows = []
    for s in smooth_scales:
        _, info = advection_term(gx, gy, cfg, smooth_km=s)
        rows.append(info)
    return pd.DataFrame(rows)
