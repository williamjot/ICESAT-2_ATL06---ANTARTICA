"""Diagnósticos oceânicos regionais derivados do GLORYS12V1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


RHO0 = 1027.0
CP0 = 3990.0


def layer_thickness(depth: np.ndarray) -> np.ndarray:
    """Espessura representada por cada nível central, limitada a 0–1000 m."""
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 1 or len(depth) < 2 or np.any(np.diff(depth) <= 0):
        raise ValueError("depth deve ser 1-D, crescente e conter ao menos 2 níveis")
    edges = np.empty(len(depth) + 1, dtype=float)
    edges[1:-1] = 0.5 * (depth[:-1] + depth[1:])
    edges[0] = max(0.0, depth[0] - 0.5 * (depth[1] - depth[0]))
    edges[-1] = min(1000.0, depth[-1] + 0.5 * (depth[-1] - depth[-2]))
    return np.maximum(np.diff(edges), 0.0)


def _weighted_mean(values, weights, axis=1):
    shape = [1] * values.ndim
    shape[axis] = len(weights)
    weight = np.asarray(weights).reshape(shape)
    valid = np.isfinite(values)
    numerator = np.nansum(values * weight, axis=axis)
    denominator = np.sum(np.where(valid, weight, 0.0), axis=axis)
    return np.divide(numerator, denominator,
                     out=np.full_like(numerator, np.nan, dtype=float),
                     where=denominator > 0)


def compute_jja_metrics(ds: xr.Dataset, *, mcdw_theta_min=0.0,
                        mcdw_salinity_min=34.5) -> xr.Dataset:
    """Calcula métricas anuais JJA usando TEOS-10 e níveis até 1000 m."""
    import gsw

    required = {"thetao", "so", "uo", "vo"}
    missing = required.difference(ds.data_vars)
    if missing:
        raise ValueError(f"variáveis GLORYS ausentes: {sorted(missing)}")
    for coordinate in ("time", "depth", "latitude", "longitude"):
        if coordinate not in ds.coords:
            raise ValueError(f"coordenada ausente: {coordinate}")
    months = set(int(value) for value in ds.time.dt.month.values)
    if not months.issubset({6, 7, 8}) or months != {6, 7, 8}:
        raise ValueError(f"entrada deve conter somente JJA completo; meses={months}")

    ordered = ds.sortby("depth")
    dims = ("time", "depth", "latitude", "longitude")
    pt = ordered.thetao.transpose(*dims).values.astype("float64")
    sp = ordered.so.transpose(*dims).values.astype("float64")
    uo = ordered.uo.transpose(*dims).values.astype("float64")
    vo = ordered.vo.transpose(*dims).values.astype("float64")
    depth = ordered.depth.values.astype(float)
    latitude = ordered.latitude.values.astype(float)
    longitude = ordered.longitude.values.astype(float)

    p = gsw.p_from_z(-depth[None, :, None, None],
                     latitude[None, None, :, None])
    sa = gsw.SA_from_SP(
        sp, p, longitude[None, None, None, :],
        latitude[None, None, :, None])
    ct = gsw.CT_from_pt(sa, pt)
    freezing = gsw.CT_freezing(sa, p, saturation_fraction=0.0)
    thermal = ct - freezing

    band = (depth >= 400.0) & (depth <= 1000.0)
    if band.sum() < 2:
        raise ValueError("menos de dois níveis entre 400 e 1000 m")
    thickness = layer_thickness(depth)[band]
    thermal_band = thermal[:, band]
    u_band, v_band = uo[:, band], vo[:, band]
    td_mean = _weighted_mean(thermal_band, thickness)
    u_mean = _weighted_mean(u_band, thickness)
    v_mean = _weighted_mean(v_band, thickness)
    speed = np.hypot(u_mean, v_mean)
    heat_content = RHO0 * CP0 * np.nansum(
        np.maximum(thermal_band, 0.0) * thickness[None, :, None, None], axis=1)
    valid_column = np.any(np.isfinite(thermal_band), axis=1)
    heat_content[~valid_column] = np.nan
    mcdw = ((pt[:, band] >= mcdw_theta_min) &
            (sp[:, band] >= mcdw_salinity_min))
    mcdw_thickness = np.sum(
        mcdw * thickness[None, :, None, None], axis=1).astype(float)
    mcdw_thickness[~valid_column] = np.nan

    gradient = np.gradient(ct, depth, axis=1)
    thermo_band = (depth >= 100.0) & (depth <= 1000.0)
    candidate = gradient[:, thermo_band]
    finite = np.any(np.isfinite(candidate), axis=1)
    index = np.argmax(np.where(np.isfinite(candidate), candidate, -np.inf), axis=1)
    thermocline = depth[thermo_band][index].astype(float)
    thermocline[~finite] = np.nan

    time = pd.DatetimeIndex(ordered.time.values)
    years = np.unique(time.year)
    monthly = {
        "thermal_driving_400_1000": td_mean,
        "potential_heat_content_400_1000": heat_content,
        "mcdw_thickness_400_1000": mcdw_thickness,
        "thermocline_depth": thermocline,
        "uo_400_1000": u_mean,
        "vo_400_1000": v_mean,
        "speed_400_1000": speed,
    }
    annual = {
        name: np.stack([np.nanmean(values[time.year == year], axis=0)
                        for year in years])
        for name, values in monthly.items()
    }
    result = xr.Dataset(
        {name: (("year", "latitude", "longitude"), values.astype("float32"))
         for name, values in annual.items()},
        coords={"year": years.astype("int16"),
                "latitude": latitude, "longitude": longitude},
        attrs={
            "title": "Diagnósticos oceânicos JJA do ASE",
            "source": "GLORYS12V1 monthly, GLOBAL_MULTIYEAR_PHY_001_030",
            "doi": "10.48670/moi-00021",
            "teos10": "SA_from_SP; CT_from_pt; CT_freezing",
            "mcdw_definition": (
                f"thetao >= {mcdw_theta_min} degC e so >= "
                f"{mcdw_salinity_min}; limiares candidatos, sujeitos a sensibilidade"),
        })
    result.thermal_driving_400_1000.attrs["units"] = "degree_Celsius"
    result.potential_heat_content_400_1000.attrs["units"] = "J m-2"
    result.mcdw_thickness_400_1000.attrs["units"] = "m"
    result.thermocline_depth.attrs["units"] = "m"
    for name in ("uo_400_1000", "vo_400_1000", "speed_400_1000"):
        result[name].attrs["units"] = "m s-1"
    return result

