"""
thwaites.qc.mask
================
Aplica a máscara BedMachine Antarctica aos pontos, por leitura de janela
(windowed reading) — nunca carrega o raster inteiro na RAM. Adiciona a coluna
`mask_class` (código BedMachine por ponto) e remove os pontos fora dos
valores mantidos (ex.: oceano = 0).

Usa recorte de janela e conversão vetorizada de coordenadas com `rowcol`.
A classificação dos pontos é feita em chunks para ser segura em memória com
dezenas de milhões de pontos.

[CONFIRMAR ao baixar o BedMachine]: caminho, formato e codificação exata do
'mask'. O código assume um GeoTIFF em EPSG:3031 lido na banda 1. Se você
baixar o NetCDF nativo, adaptamos a leitura (rasterio subdataset ou xarray).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger


def resolve_mask_path(cfg: Config) -> Path:
    """Resolve o caminho do BedMachine (relativo -> sob data/)."""
    p = Path(cfg.mask.bedmachine_path)
    if not p.is_absolute():
        p = cfg.paths.data_dir / p
    return p


def apply_bedmachine_mask(
    df: pd.DataFrame,
    cfg: Config,
    tif_path: str | Path | None = None,
    chunk: int | None = None,
) -> pd.DataFrame:
    """
    Classifica os pontos pela máscara BedMachine e filtra os mantidos.

    Parâmetros
    ----------
    df : DataFrame com colunas 'lon' e 'lat' (EPSG:4326).
    cfg : Config.
    tif_path : override do caminho do raster (para teste).
    chunk : nº de pontos por bloco de classificação (default: cfg.tiles.chunk).

    Retorna
    -------
    DataFrame filtrado (só valores mantidos) com nova coluna 'mask_class' (int).
    """
    import rasterio
    from rasterio.windows import from_bounds, Window
    from rasterio.transform import rowcol
    from pyproj import Transformer

    logger = get_logger()
    tif_path = Path(tif_path) if tif_path else resolve_mask_path(cfg)
    if not tif_path.exists():
        raise FileNotFoundError(
            f"BedMachine não encontrado: {tif_path}\n"
            f"Baixe o arquivo e ajuste 'mask.bedmachine_path' na config."
        )

    chunk = chunk or cfg.tiles.chunk
    keep = set(cfg.mask.keep_values)
    n = len(df)
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()

    to_polar = Transformer.from_crs(
        f"EPSG:{cfg.area.epsg_lonlat}", f"EPSG:{cfg.area.epsg_polar}", always_xy=True
    )

    # Janela do raster = bbox dos pontos (em 3031) + buffer de 10 km.
    x_all, y_all = to_polar.transform(lon, lat)
    bx0, bx1 = float(np.nanmin(x_all)) - 10_000, float(np.nanmax(x_all)) + 10_000
    by0, by1 = float(np.nanmin(y_all)) - 10_000, float(np.nanmax(y_all)) + 10_000

    with rasterio.open(tif_path) as src:
        window = from_bounds(bx0, by0, bx1, by1, transform=src.transform)
        window = window.intersection(Window(0, 0, src.width, src.height))
        band = src.read(1, window=window)
        full_transform = src.transform          # transform do RASTER COMPLETO
        logger.info(f"Máscara: janela {band.shape}, CRS {src.crs}, "
                    f"valores únicos {np.unique(band)[:15]}")

    # DETERMINISMO: o índice de pixel é calculado com o transform do RASTER
    # COMPLETO e só depois deslocado pela origem da janela. O transform da
    # JANELA faria o arredondamento de pontos sobre a fronteira entre pixels
    # depender da origem da janela e, portanto, do tamanho do lote no
    # processamento em streaming, quebrando a reprodutibilidade.
    row_off = int(window.row_off)
    col_off = int(window.col_off)

    mask_class = np.empty(n, dtype=np.int32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        xc, yc = x_all[s:e], y_all[s:e]
        rows, cols = rowcol(full_transform, xc, yc)
        rows = np.clip(np.asarray(rows) - row_off, 0, band.shape[0] - 1)
        cols = np.clip(np.asarray(cols) - col_off, 0, band.shape[1] - 1)
        mask_class[s:e] = band[rows, cols]

    keep_mask = np.isin(mask_class, list(keep))
    out = df.loc[keep_mask].copy()
    out["mask_class"] = mask_class[keep_mask]

    n_removed = n - len(out)
    logger.info(f"Máscara: mantidos {len(out):,}/{n:,} "
                f"({100*len(out)/max(n,1):.1f}%), removidos {n_removed:,} (fora de {sorted(keep)})")
    return out
