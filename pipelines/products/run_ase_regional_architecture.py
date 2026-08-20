"""Arquitetura oceânica estática do ASE com IBCSO v2 e BedMachine v4.

Recorta e reprojeta somente a ROI aprovada para EPSG:3031, deriva calado,
espessura da coluna d'água sob plataformas e declividade batimétrica e gera um
mapa preliminar mantendo oceano azul e continente com o DEM sombreado.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.path import Path as MplPath
from pyproj import Transformer
from rasterio import band as rio_band
from rasterio import open as rio_open
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.viz.basemap import add_scale_bar, draw_basemap, draw_calving_fronts

LON_MIN, LON_MAX = -115.0, -95.0
LAT_MIN, LAT_MAX = -77.5, -73.0
RESOLUTION = 500.0
REFERENCE = ROOT / "data" / "ocean" / "regional" / "reference"
OUTPUT = ROOT / "outputs" / "mecanismo_oceanico_regional"


def roi_polygon(n=300):
    bottom_lon = np.linspace(LON_MIN, LON_MAX, n)
    right_lat = np.linspace(LAT_MIN, LAT_MAX, n)
    top_lon = np.linspace(LON_MAX, LON_MIN, n)
    left_lat = np.linspace(LAT_MAX, LAT_MIN, n)
    lon = np.r_[bottom_lon, np.full(n, LON_MAX), top_lon,
                np.full(n, LON_MIN), bottom_lon[:1]]
    lat = np.r_[np.full(n, LAT_MIN), right_lat, np.full(n, LAT_MAX),
                left_lat, LAT_MIN]
    x, y = Transformer.from_crs(4326, 3031, always_xy=True).transform(lon, lat)
    vertices = np.column_stack([x, y])
    extent = (float(np.min(x)), float(np.max(x)),
              float(np.min(y)), float(np.max(y)))
    return np.asarray(x), np.asarray(y), MplPath(vertices), extent


def target_grid(extent):
    x0 = np.floor(extent[0] / RESOLUTION) * RESOLUTION
    x1 = np.ceil(extent[1] / RESOLUTION) * RESOLUTION
    y0 = np.floor(extent[2] / RESOLUTION) * RESOLUTION
    y1 = np.ceil(extent[3] / RESOLUTION) * RESOLUTION
    width = int(round((x1 - x0) / RESOLUTION))
    height = int(round((y1 - y0) / RESOLUTION))
    transform = from_origin(x0, y1, RESOLUTION, RESOLUTION)
    x = x0 + (np.arange(width) + 0.5) * RESOLUTION
    y = y1 - (np.arange(height) + 0.5) * RESOLUTION
    return x, y, transform


def warp_layer(path, shape, transform, resampling, dtype="float32"):
    result = np.full(shape, np.nan if "float" in dtype else -32768, dtype=dtype)
    with rio_open(path) as src:
        reproject(
            source=rio_band(src, 1), destination=result,
            src_transform=src.transform, src_crs=src.crs,
            src_nodata=src.nodata, dst_transform=transform,
            dst_crs="EPSG:3031",
            dst_nodata=np.nan if "float" in dtype else -32768,
            resampling=resampling, num_threads=4)
    return result


def bedmachine_on_grid(path, x, y, transform, shape):
    with xr.open_dataset(path, decode_times=False) as source:
        margin = 2 * RESOLUTION
        xmin, xmax = float(x.min() - margin), float(x.max() + margin)
        ymin, ymax = float(y.min() - margin), float(y.max() + margin)
        x_slice = slice(xmin, xmax) if source.x[0] < source.x[-1] else slice(xmax, xmin)
        y_slice = slice(ymin, ymax) if source.y[0] < source.y[-1] else slice(ymax, ymin)
        subset = source.sel(x=x_slice, y=y_slice)
        bx = subset.x.values.astype(float)
        by = subset.y.values.astype(float)
        bm_transform = from_origin(
            bx.min() - RESOLUTION / 2, by.max() + RESOLUTION / 2,
            RESOLUTION, RESOLUTION)
        output = {}
        for name in ("bed", "surface", "thickness", "errbed", "mask"):
            source_values = subset[name].values
            # NetCDF é armazenado com y crescente; raster norte-para-sul requer flip.
            if by[0] < by[-1]:
                source_values = source_values[::-1, :]
            is_mask = name == "mask"
            destination = np.full(
                shape, -127 if is_mask else np.nan,
                dtype="int8" if is_mask else "float32")
            reproject(
                source=source_values, destination=destination,
                src_transform=bm_transform, src_crs="EPSG:3031",
                src_nodata=None, dst_transform=transform, dst_crs="EPSG:3031",
                dst_nodata=-127 if is_mask else np.nan,
                resampling=Resampling.nearest if is_mask else Resampling.bilinear,
                num_threads=4)
            output[name] = destination
    return output


def build_dataset():
    roi_x, roi_y, roi_path, extent = roi_polygon()
    x, y, transform = target_grid(extent)
    shape = (len(y), len(x))
    xx, yy = np.meshgrid(x, y)
    inside = roi_path.contains_points(
        np.column_stack([xx.ravel(), yy.ravel()]), radius=1.0).reshape(shape)

    ibcso_bed = warp_layer(
        REFERENCE / "IBCSO_v2_bed.tif", shape, transform,
        Resampling.bilinear, "float32")
    ibcso_rid = warp_layer(
        REFERENCE / "IBCSO_v2_RID.tif", shape, transform,
        Resampling.nearest, "int16")
    ibcso_tid = warp_layer(
        REFERENCE / "IBCSO_v2_TID.tif", shape, transform,
        Resampling.nearest, "int16")

    bedmachine = ROOT / "data" / (
        "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")
    bm = bedmachine_on_grid(bedmachine, x, y, transform, shape)
    floating = bm["mask"] == 3
    grounded = bm["mask"] == 2
    ocean = bm["mask"] == 0
    draft = np.where(floating, bm["surface"] - bm["thickness"], np.nan)
    water_column = np.where(
        floating, np.maximum(draft - bm["bed"], 0.0), np.nan)
    fused_bed = np.where(floating, bm["bed"], np.where(ocean, ibcso_bed, np.nan))
    dzdy, dzdx = np.gradient(fused_bed, RESOLUTION, RESOLUTION)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    variables = {
        "ibcso_bed": ibcso_bed,
        "ibcso_rid": ibcso_rid,
        "ibcso_tid": ibcso_tid,
        "bedmachine_bed": bm["bed"],
        "surface": bm["surface"],
        "thickness": bm["thickness"],
        "bed_uncertainty": bm["errbed"],
        "mask": bm["mask"],
        "ice_draft": draft,
        "water_column_thickness": water_column,
        "fused_bed": fused_bed,
        "bathy_slope": slope,
        "roi_mask": inside.astype("uint8"),
    }
    integer_fill = {"ibcso_rid": -32768, "ibcso_tid": -32768, "mask": -127}
    for name, values in variables.items():
        if name == "roi_mask":
            continue
        if name in integer_fill:
            variables[name] = np.where(inside, values, integer_fill[name])
        else:
            variables[name] = np.where(inside, values, np.nan)
    ds = xr.Dataset(
        {name: (("y", "x"), values) for name, values in variables.items()},
        coords={"x": x, "y": y},
        attrs={
            "title": "Arquitetura oceânica do Amundsen Sea Embayment",
            "crs": "EPSG:3031",
            "roi": "115W–95W; 77.5S–73S",
            "ibcso_doi": "10.1594/PANGAEA.937574",
            "bedmachine_doi": "10.5067/POJQI54A45HX",
            "note": "IBCSO no oceano aberto; BedMachine v4 sob gelo flutuante",
        })
    ds["ibcso_bed"].attrs.update(units="m", positive="up")
    ds["fused_bed"].attrs.update(units="m", positive="up")
    ds["ice_draft"].attrs.update(units="m", positive="up")
    ds["water_column_thickness"].attrs.update(units="m")
    ds["bathy_slope"].attrs.update(units="degree")
    return ds, roi_x, roi_y, extent


def plot_architecture(ds, roi_x, roi_y, extent, path):
    cfg = load_config("jja")
    fig, ax = plt.subplots(figsize=(11.2, 8.8), constrained_layout=True,
                           facecolor="white")
    draw_basemap(ax, cfg, *extent, target_px=900)
    xkm, ykm = ds.x.values / 1000, ds.y.values / 1000
    bathy = ds.ibcso_bed.values
    ocean = ds["mask"].values == 0
    contours = ax.contour(
        xkm, ykm, np.where(ocean, bathy, np.nan),
        levels=[-1500, -1000, -700, -500, -300],
        colors=["#234c6f", "#356b8f", "#4d82a4", "#6e9fba", "#91bbce"],
        linewidths=0.75, zorder=6)
    ax.clabel(contours, fmt="%d m", fontsize=7, inline=True)
    water = np.ma.masked_invalid(ds.water_column_thickness.values)
    artist = ax.pcolormesh(
        xkm, ykm, water, cmap="viridis", vmin=0, vmax=1000,
        shading="auto", alpha=0.82, rasterized=True, zorder=7)
    draw_calving_fronts(ax, cfg, 2022.5, color="#00509e", lw=0.9)
    ax.plot(roi_x / 1000, roi_y / 1000, color="#202020", lw=0.9,
            ls=(0, (4, 2)), zorder=13)
    ax.set_xlim(extent[0] / 1000, extent[1] / 1000)
    ax.set_ylim(extent[2] / 1000, extent[3] / 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    add_scale_bar(ax, 100)
    cbar = fig.colorbar(artist, ax=ax, shrink=0.78, pad=0.02, extend="max")
    cbar.set_label("espessura da coluna d'água sob a plataforma (m)")
    ax.set_title(
        "Arquitetura oceânica do Amundsen Sea Embayment\n"
        "batimetria IBCSO v2 · cavidades BedMachine v4 · EPSG:3031",
        loc="left", fontweight="bold")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ds, roi_x, roi_y, extent = build_dataset()
    netcdf = REFERENCE / "arquitetura_ase_500m_epsg3031.nc"
    encoding = {
        name: {"zlib": True, "complevel": 4,
               "dtype": "uint8" if name == "roi_mask" else "float32"}
        for name in ds.data_vars
    }
    # IDs e máscara preservam classes inteiras.
    for name, dtype in (("ibcso_rid", "int16"), ("ibcso_tid", "int16"),
                        ("mask", "int8")):
        encoding[name]["dtype"] = dtype
        encoding[name]["_FillValue"] = -127 if name == "mask" else -32768
    ds.to_netcdf(netcdf, encoding=encoding)
    figure = OUTPUT / "mapa_arquitetura_oceanica_ase.png"
    plot_architecture(ds, roi_x, roi_y, extent, figure)
    report = {
        "status": "ARQUITETURA_ESTATICA_CONCLUIDA",
        "roi": "115W–95W; 77.5S–73S",
        "resolution_m": RESOLUTION,
        "shape": {"y": int(ds.sizes["y"]), "x": int(ds.sizes["x"])},
        "floating_cells": int(np.isfinite(ds.water_column_thickness).sum()),
        "outputs": {"dataset": str(netcdf), "map": str(figure)},
        "limitations": [
            "contornos indicam geometria, não corrente observada",
            "incerteza batimétrica deve acompanhar extração de corredores",
            "IBCSO v2 possui cobertura direta incompleta; parte do campo é interpolada",
        ],
    }
    report_path = OUTPUT / "arquitetura_oceanica_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"Dataset -> {netcdf}")
    print(f"Mapa -> {figure}")
    print(f"Relatório -> {report_path}")


if __name__ == "__main__":
    main()
