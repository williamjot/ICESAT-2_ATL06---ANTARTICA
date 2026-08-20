"""
pipelines/fetch_rema.py
=======================
Baixa o REMA (Reference Elevation Model of Antarctica, v2.0, 32 m) recortado à
ROI da Thwaites e grava um GeoTIFF local em EPSG:3031, para a correção de slope
(diferenciação por DEM de referência — Howat et al. 2019; Schröder et al. 2019).

Usa leitura REMOTA por janela dos COGs no bucket público da PGC (AWS) — só os
tiles que cobrem a ROI, recortados. Não baixa a Antártica inteira.

Grade de tiles v2.0 32m (derivada empiricamente dos bounds dos COGs):
    tile "R_C" cobre x=[C*100-3100, C*100-3000] km, y=[R*100-3100, R*100-3000] km.

Saída: data/REMA_thwaites_32m.tif  (config slope.rema_path)
Uso:   python pipelines/fetch_rema.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config

BUCKET = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/"
PREFIX = "rema/mosaics/v2.0/32m/"
TILE_KM = 100.0
ORIGIN_KM = -3100.0   # x_left = C*100 - 3100 ; y_bottom = R*100 - 3100


def roi_xy_bounds(cfg, buffer_m=30_000.0):
    """
    Extensão x/y (EPSG:3031) da ROI, derivada da CONFIG (+buffer).

    Derivar a extensão de `dhdt_nodes.parquet` seria circular: numa expansão da
    ROI, o REMA é necessário antes de haver nós e ficaria limitado ao domínio do
    produto já existente.
    """
    from thwaites.grid.reproject import to_polar

    roi = cfg.roi or cfg.area
    clon = np.array([roi.lon_min, roi.lon_max, roi.lon_min, roi.lon_max])
    clat = np.array([roi.lat_min, roi.lat_min, roi.lat_max, roi.lat_max])
    cx, cy = to_polar(clon, clat, cfg)
    return (float(np.min(cx)) - buffer_m, float(np.min(cy)) - buffer_m,
            float(np.max(cx)) + buffer_m, float(np.max(cy)) + buffer_m)


def tiles_for_bounds(xmin, ymin, xmax, ymax):
    c0 = int(np.floor((xmin / 1000 - ORIGIN_KM) / TILE_KM))
    c1 = int(np.floor((xmax / 1000 - ORIGIN_KM) / TILE_KM))
    r0 = int(np.floor((ymin / 1000 - ORIGIN_KM) / TILE_KM))
    r1 = int(np.floor((ymax / 1000 - ORIGIN_KM) / TILE_KM))
    return [f"{r:02d}_{c:02d}" for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def main():
    import rasterio
    from rasterio.merge import merge

    cfg = load_config()
    xmin, ymin, xmax, ymax = roi_xy_bounds(cfg)
    print(f"ROI (km): x[{xmin/1000:.0f},{xmax/1000:.0f}] y[{ymin/1000:.0f},{ymax/1000:.0f}]")
    tiles = tiles_for_bounds(xmin, ymin, xmax, ymax)
    print(f"Tiles candidatos: {len(tiles)} -> {tiles}")

    srcs = []
    for t in tiles:
        url = f"/vsicurl/{BUCKET}{PREFIX}{t}/{t}_32m_v2.0_dem.tif"
        try:
            srcs.append(rasterio.open(url))
        except Exception:
            print(f"  (tile {t} inexistente/sem cobertura — pulado)")
    if not srcs:
        raise SystemExit("Nenhum tile REMA disponível para a ROI.")
    print(f"Tiles abertos: {len(srcs)}. Mosaicando recortado à ROI...")

    mosaic, transform = merge(srcs, bounds=(xmin, ymin, xmax, ymax))
    nodata = srcs[0].nodata if srcs[0].nodata is not None else -9999.0
    for s in srcs:
        s.close()

    out = cfg.paths.data_dir / Path(cfg.slope.rema_path).name
    with rasterio.open(
        out, "w", driver="GTiff",
        height=mosaic.shape[1], width=mosaic.shape[2], count=1,
        dtype="float32", crs="EPSG:3031", transform=transform,
        nodata=nodata, compress="deflate", predictor=3, tiled=True,
    ) as dst:
        dst.write(mosaic[0].astype("float32"), 1)

    valid = mosaic[0][mosaic[0] != nodata]
    print(f"\nREMA ROI -> {out}  ({out.stat().st_size/1024**2:.0f} MB)")
    print(f"  shape {mosaic.shape[1]}x{mosaic.shape[2]} @ 32 m | "
          f"elevação {valid.min():.0f}..{valid.max():.0f} m | "
          f"nodata {100*np.mean(mosaic[0]==nodata):.1f}%")
    print("\nPróximo: python pipelines/run_slope.py")


if __name__ == "__main__":
    main()
