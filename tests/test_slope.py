"""Teste da correção de slope (diferenciação REMA) com um DEM sintético."""

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from thwaites.corrections.slope import apply_slope_reference, sample_rema_bilinear


def _make_linear_rema(path, x0=-1_500_000.0, y0=-400_000.0, res=1000.0, n=50,
                      nodata=-9999.0, nodata_cell=(5, 5)):
    """DEM linear z = 0.001x + 0.0005y + 1000 (bilinear amostra exato)."""
    cols = x0 + (np.arange(n) + 0.5) * res
    rows_y = y0 - (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(cols, rows_y)
    Z = (0.001 * X + 0.0005 * Y + 1000.0).astype("float32")
    Z[nodata_cell] = nodata
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1,
                       dtype="float32", crs="EPSG:3031",
                       transform=from_origin(x0, y0, res, res), nodata=nodata) as dst:
        dst.write(Z, 1)
    return lambda x, y: 0.001 * x + 0.0005 * y + 1000.0


def test_slope_residual_and_nodata(tmp_path, cfg):
    tif = tmp_path / "rema.tif"
    ztrue = _make_linear_rema(tif, nodata_cell=(5, 5))
    x0, y0, res = -1_500_000.0, -400_000.0, 1000.0

    def center(col, row):
        return x0 + (col + 0.5) * res, y0 - (row + 0.5) * res

    xa, ya = center(25, 25)   # ponto normal
    xb, yb = center(5, 5)     # ponto sobre nodata
    xc, yc = center(30, 20)   # blunder (Δh enorme)
    df = pd.DataFrame({
        "x": [xa, xb, xc], "y": [ya, yb, yc],
        "h_corr": [ztrue(xa, ya) + 0.5, ztrue(xb, yb) + 0.5, ztrue(xc, yc) + 500.0],
    })
    out = apply_slope_reference(df, cfg, rema_path=tif)

    assert "h_res" in out.columns and "rema" in out.columns
    h = out["h_res"].to_numpy()
    # ponto normal: resíduo = Δh imposto (0.5)
    assert np.isclose(h[0], 0.5, atol=1e-2)
    # nodata -> NaN
    assert np.isnan(h[1])
    # blunder (>max_slope_ref_m=200) -> NaN
    assert np.isnan(h[2])


def test_sample_bilinear_exact_on_linear(tmp_path, cfg):
    tif = tmp_path / "rema.tif"
    ztrue = _make_linear_rema(tif, nodata_cell=(0, 0))
    x0, y0, res = -1_500_000.0, -400_000.0, 1000.0
    # ponto fora do centro (entre pixels) — bilinear deve reproduzir o plano
    x = x0 + 12.3 * res
    y = y0 - 8.7 * res
    val = sample_rema_bilinear(np.array([x]), np.array([y]), tif)[0]
    assert np.isclose(val, ztrue(x, y), atol=1e-2)
