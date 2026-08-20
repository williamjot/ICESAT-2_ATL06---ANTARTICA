"""Teste da máscara BedMachine com um GeoTIFF sintético em EPSG:3031."""

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin
from pyproj import Transformer

from thwaites.qc.mask import apply_bedmachine_mask


def _make_synthetic_mask(tif_path):
    """
    Raster 20x20 @ 500 m em EPSG:3031. Colunas 0-4 = oceano(0),
    5-9 = flutuante(3), resto = aterrado(2). Retorna (transform, x0, y0, res).
    """
    res = 500.0
    W = H = 20
    x0, y0 = -1_600_000.0, -400_000.0   # canto superior-esquerdo (y para baixo)
    transform = from_origin(x0, y0, res, res)
    band = np.full((H, W), 2, dtype="int16")   # aterrado
    band[:, 0:5] = 0                            # oceano
    band[:, 5:10] = 3                           # flutuante
    with rasterio.open(
        tif_path, "w", driver="GTiff", height=H, width=W, count=1,
        dtype="int16", crs="EPSG:3031", transform=transform,
    ) as dst:
        dst.write(band, 1)
    return x0, y0, res


def _pixel_center_lonlat(x0, y0, res, col, row):
    x = x0 + (col + 0.5) * res
    y = y0 - (row + 0.5) * res
    to_ll = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(x, y)
    return lon, lat


def test_mask_classifies_and_filters(tmp_path, cfg):
    tif = tmp_path / "mask.tif"
    x0, y0, res = _make_synthetic_mask(tif)

    # um ponto em cada classe: oceano(col2), flutuante(col7), aterrado(col15)
    pts = [_pixel_center_lonlat(x0, y0, res, c, 10) for c in (2, 7, 15)]
    df = pd.DataFrame({
        "lon": [p[0] for p in pts],
        "lat": [p[1] for p in pts],
        "h_elv": np.array([100.0, 100.0, 100.0], dtype="float32"),
    })

    out = apply_bedmachine_mask(df, cfg, tif_path=tif)

    # oceano (0) removido; sobram flutuante e aterrado
    assert len(out) == 2
    classes = set(out["mask_class"].tolist())
    assert classes == {3, 2}
    assert "mask_class" in out.columns


def test_mask_missing_file(cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        apply_bedmachine_mask(
            pd.DataFrame({"lon": [-100.0], "lat": [-75.0], "h_elv": [1.0]}),
            cfg, tif_path=tmp_path / "inexistente.tif",
        )
