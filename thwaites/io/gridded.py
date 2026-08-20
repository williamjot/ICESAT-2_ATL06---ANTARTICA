"""
thwaites.io.gridded
===================
Recorte robusto de produtos em grade (NetCDF) à ROI do projeto.

POR QUE EXISTE: o primeiro recorte do GSFC-FDM falhou porque eu assumi
coordenadas `x`/`y` unidimensionais — e o produto as traz em **2D** (441×401),
com as dimensões na ordem `(time, x, y)`. Perder um download de 2,8 GB por uma
suposição de estrutura é caro; então o recorte passou a:

  1. aceitar coords 1D **ou** 2D;
  2. no caso 2D, **verificar** que a grade é regular antes de reduzir a eixos 1D
     (recortar por índice numa grade irregular daria resultado silenciosamente
     errado);
  3. aceitar tempo em **anos decimais** ou em calendário (datetime64);
  4. devolver sempre a convenção `(time, y, x)` com coords 1D.

Usado por `fetch_firn.py` e `fetch_velocity.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thwaites.config import Config
from thwaites.grid.reproject import to_polar

_X_NAMES = ("x", "easting", "X")
_Y_NAMES = ("y", "northing", "Y")


def roi_bounds_polar(cfg: Config, buffer_km: float = 100.0):
    """bbox da ROI em EPSG:3031 (+buffer)."""
    roi = cfg.roi or cfg.area
    clon = np.array([roi.lon_min, roi.lon_max, roi.lon_min, roi.lon_max])
    clat = np.array([roi.lat_min, roi.lat_min, roi.lat_max, roi.lat_max])
    cx, cy = to_polar(clon, clat, cfg)
    b = buffer_km * 1000.0
    return (float(np.min(cx)) - b, float(np.max(cx)) + b,
            float(np.min(cy)) - b, float(np.max(cy)) + b)


def _variation_axis(A: np.ndarray):
    """
    Ao longo de QUAL eixo esta coordenada 2D varia? (0, 1 ou None se ambíguo)

    Se `A[:, 0] == A[:, -1]` (duas colunas iguais), A não depende do índice de
    coluna; portanto, varia ao longo do eixo 0.
    """
    varies0 = not np.allclose(A[0, :], A[-1, :])   # linhas diferem => varia no eixo 0
    varies1 = not np.allclose(A[:, 0], A[:, -1])   # colunas diferem => varia no eixo 1
    if varies0 and not varies1:
        return 0
    if varies1 and not varies0:
        return 1
    return None


def _axes_1d(ds, xn: str, yn: str):
    """
    Extrai eixos 1D das coordenadas, aceitando 1D ou 2D regular.

    Retorna (ax, ay, axis_x, axis_y, ok, motivo). `ok=False` se a grade 2D não
    for regular — caso em que recortar por índice daria resultado
    silenciosamente errado.
    """
    X = np.asarray(ds[xn].values)
    Y = np.asarray(ds[yn].values)
    if X.ndim == 1 and Y.ndim == 1:
        return X.astype(float), Y.astype(float), None, None, True, ""
    if X.ndim != 2 or Y.ndim != 2:
        return None, None, None, None, False, f"ndim inesperado: x={X.ndim}, y={Y.ndim}"

    axx = _variation_axis(X)
    axy = _variation_axis(Y)
    if axx is None or axy is None:
        return None, None, None, None, False, "grade 2D não regular (coord varia nos dois eixos)"
    if axx == axy:
        return None, None, None, None, False, "x e y variam no MESMO eixo (grade inválida)"

    ax = (X[:, 0] if axx == 0 else X[0, :]).astype(float)
    ay = (Y[:, 0] if axy == 0 else Y[0, :]).astype(float)
    return ax, ay, axx, axy, True, ""


def crop_gridded_to_roi(nc_path, cfg: Config, variables=None,
                        year_start: int | None = None,
                        buffer_km: float = 100.0, verbose: bool = True):
    """
    Abre um NetCDF em grade de forma LAZY, recorta à ROI (+buffer) e ao período.

    Retorna um xarray.Dataset materializado com coords 1D e dims (time, y, x),
    ou None se não for possível recortar (com o motivo impresso).
    """
    import xarray as xr

    nc_path = Path(nc_path)
    x0, x1, y0, y1 = roi_bounds_polar(cfg, buffer_km)
    ds = xr.open_dataset(nc_path, chunks={}, decode_times=False)

    xn = next((c for c in _X_NAMES if c in ds.coords or c in ds.variables), None)
    yn = next((c for c in _Y_NAMES if c in ds.coords or c in ds.variables), None)
    if xn is None or yn is None:
        if verbose:
            print(f"        (sem coords x/y: coords={list(ds.coords)}) — pulado")
        ds.close()
        return None

    ax, ay, axx, axy, ok, why = _axes_1d(ds, xn, yn)
    if not ok:
        if verbose:
            print(f"        ({why} — recorte por índice inválido) — pulado")
        ds.close()
        return None

    # nome da DIMENSÃO correspondente a cada eixo (pode diferir do nome da coord)
    if axx is None:                      # coords já 1D
        dimx, dimy = xn, yn
    else:
        dimx = ds[xn].dims[axx]
        dimy = ds[yn].dims[axy]

    ix = np.flatnonzero((ax >= x0) & (ax <= x1))
    iy = np.flatnonzero((ay >= y0) & (ay <= y1))
    if ix.size == 0 or iy.size == 0:
        if verbose:
            print("        (ROI fora da grade) — pulado")
        ds.close()
        return None

    sel = {dimx: slice(int(ix[0]), int(ix[-1]) + 1),
           dimy: slice(int(iy[0]), int(iy[-1]) + 1)}

    if year_start is not None and "time" in ds.dims:
        t = np.asarray(ds["time"].values)
        if np.issubdtype(t.dtype, np.number):
            it = np.flatnonzero(t.astype(float) >= year_start)   # anos decimais
        else:
            import pandas as pd
            it = np.flatnonzero(pd.to_datetime(t).year >= year_start)
        if it.size:
            sel["time"] = slice(int(it[0]), int(it[-1]) + 1)

    sub = ds.isel(sel)
    if variables:
        keep = [v for v in variables if v in sub.data_vars]
        if keep:
            sub = sub[keep]

    sub = sub.drop_vars([c for c in (xn, yn) if c in sub.coords])
    sub = sub.assign_coords({dimx: (dimx, ax[ix]), dimy: (dimy, ay[iy])})
    if dimx != "x" or dimy != "y":
        sub = sub.rename({dimx: "x", dimy: "y"})
    out = sub.load()
    ds.close()

    dims = [d for d in ("time", "y", "x") if d in out.dims]
    out = out.transpose(*dims)
    if verbose:
        print(f"        recortado: {dict(out.sizes)} | vars {list(out.data_vars)}")
    return out
