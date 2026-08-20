"""
thwaites.corrections.slope
==========================
Correção de slope por diferenciação de DEM de referência (REMA).

Δh = h_corr − REMA(x, y) remove a topografia estática (e seu declive), atacando
o alias de slope cross-track das trilhas não-repetentes do ICESat-2 (Schröder
et al. 2019; Howat et al. 2019). Como dh/dt é a derivada temporal, o offset do
epoch do REMA não afeta a taxa — só reduz o resíduo do ajuste (o RMSE de ~2,6 m
observado sem REMA), o que aperta a incerteza e permite raio de busca menor.

Amostragem BILINEAR do REMA (o offset cross-track é de dezenas de m; vizinho-
mais-próximo a 32 m introduziria erro de elevação comparável ao sinal).
Pontos sobre nodata ou com |Δh| > max_slope_ref_m (blunder) recebem h_res=NaN
e ficam de fora do ajuste — em vez de misturar altura absoluta e residual.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.grid.tiles import assign_xy
from thwaites.logging import get_logger


def resolve_rema_path(cfg: Config) -> Path:
    p = Path(cfg.slope.rema_path)
    return p if p.is_absolute() else cfg.paths.data_dir / p


def sample_rema_bilinear(x, y, rema_path: Path, block_km: float = 50.0,
                         margin_px: int = 4):
    """
    Amostra o REMA (bilinear) nos pontos (x,y) EPSG:3031. nodata -> NaN.

    LEITURA POR BLOCO: materializar o mosaico inteiro exigiria 1,28 GB em
    float64 na ROI da Thwaites e cerca de 3,1 GB numa ROI 2,4× maior, antes de
    carregar os pontos. Os pontos são binados em blocos de `block_km`; em cada
    bloco, somente a
    janela correspondente do raster é lida (~20-50 MB a 32 m). O resultado é
    numericamente equivalente à leitura integral.

    Diferente de `advection.py`, aqui NÃO se decima: a resolução nativa de 32 m
    é exatamente o que dá valor à correção de slope.
    """
    import rasterio
    from rasterio.windows import Window
    from scipy.ndimage import map_coordinates

    logger = get_logger()
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    block_m = block_km * 1000.0
    with rasterio.open(rema_path) as src:
        inv = ~src.transform
        nodata = src.nodata
        H, W = src.height, src.width

        # bin espacial dos pontos
        bi = np.floor(x / block_m).astype(np.int64)
        bj = np.floor(y / block_m).astype(np.int64)
        keys = bi * 100003 + bj
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[starts[1:], len(ks)]
        logger.info(f"REMA: amostrando {n:,} pontos em {len(starts)} blocos de "
                    f"{block_km:.0f} km (janela por bloco, sem carregar o mosaico)")

        for s, e in zip(starts, ends):
            idx = order[s:e]
            cols, rows = inv * (x[idx], y[idx])
            c0 = int(np.floor(cols.min())) - margin_px
            c1 = int(np.ceil(cols.max())) + margin_px
            r0 = int(np.floor(rows.min())) - margin_px
            r1 = int(np.ceil(rows.max())) + margin_px
            c0c, r0c = max(c0, 0), max(r0, 0)
            c1c, r1c = min(c1, W), min(r1, H)
            if c1c <= c0c or r1c <= r0c:
                continue                     # bloco fora do raster

            win = Window(c0c, r0c, c1c - c0c, r1c - r0c)
            band = src.read(1, window=win).astype(np.float64)
            valid = np.ones_like(band)
            if nodata is not None:
                bad = band == nodata
                valid[bad] = 0.0
                band[bad] = 0.0

            # coords relativas à janela; meio pixel porque map_coordinates
            # indexa CENTROS e o affine mapeia CANTOS
            ci = cols - c0c - 0.5
            ri = rows - r0c - 0.5
            coords = np.vstack([ri, ci])
            vals = map_coordinates(band, coords, order=1, mode="constant", cval=np.nan)
            wv = map_coordinates(valid, coords, order=1, mode="constant", cval=0.0)
            bh, bw = band.shape
            inside = (ri >= 0) & (ri <= bh - 1) & (ci >= 0) & (ci <= bw - 1)
            vals[~inside] = np.nan
            vals[wv < 0.999] = np.nan        # algum vizinho era nodata
            out[idx] = vals
            del band, valid
    return out


def apply_slope_reference(df: pd.DataFrame, cfg: Config,
                          rema_path: Path | None = None) -> pd.DataFrame:
    """
    Adiciona colunas `rema` (elevação amostrada) e `h_res` (= h_corr − REMA).
    h_res é NaN onde o REMA é inválido ou |Δh| > max_slope_ref_m.
    """
    logger = get_logger()
    rema_path = Path(rema_path) if rema_path else resolve_rema_path(cfg)
    if not rema_path.exists():
        raise FileNotFoundError(
            f"REMA não encontrado: {rema_path}\nRode pipelines/fetch_rema.py.")

    df = assign_xy(df, cfg)
    hcol = "h_corr" if "h_corr" in df.columns else "h_elv"
    rema = sample_rema_bilinear(df["x"].to_numpy(), df["y"].to_numpy(), rema_path)
    res = df[hcol].to_numpy(dtype=np.float64) - rema

    invalid = ~np.isfinite(res) | (np.abs(res) > cfg.slope.max_slope_ref_m)
    res = res.astype(np.float32)
    res[invalid] = np.nan

    out = df.copy()
    out["rema"] = rema.astype(np.float32)
    out["h_res"] = res
    n_ref = int(np.isfinite(res).sum())
    logger.info(f"slope (REMA): referenciados {n_ref:,}/{len(out):,} pontos "
                f"({100*n_ref/len(out):.1f}%) | Δh=h-REMA média {np.nanmean(res):+.3f} m, "
                f"desvio {np.nanstd(res):.3f} m. NOTA: esse desvio inclui a mudança real "
                f"acumulada desde o epoch do REMA (sinal, não ruído); o efeito na qualidade "
                f"do ajuste mede-se pelo RMSE POR NÓ do dh/dt, não por este desvio global.")
    return out
