"""Classificacao espaco-temporal da transicao grounded/floating.

O BedMachine v4 e somente um prior topologico longe da transicao. A faixa de
transicao usa linhas anuais NSIDC-0498 e a largura de flexao Up/Dn NSIDC-0778.
Sem linha local no ano, a envoltoria historica vira ``unknown``: nao se
interpola silenciosamente uma linha de aterramento.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from thwaites.qc.grounded_mask import BM_FLOATING_ICE, BM_GROUNDED_ICE

UNKNOWN = np.int8(0)
GROUNDED_CONFIDENT = np.int8(1)
GROUNDING_ZONE = np.int8(2)
FLOATING_CONFIDENT = np.int8(3)

STATE_NAMES = {0: "unknown", 1: "grounded_confident",
               2: "grounding_zone", 3: "floating_confident"}
SUPPORT_NAMES = {
    0: "unsupported",
    1: "bedmachine_v4_static_prior",
    2: "nsidc0498_exact_year_nsidc0778_width",
    3: "nsidc0498_exact_year_roi_width_fallback",
    4: "historical_gl_envelope_without_local_exact_year_line",
}


def normalize_glacier_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _coordinates(geometry, step_m: float) -> np.ndarray:
    from shapely import get_coordinates, segmentize
    if geometry is None or geometry.is_empty:
        return np.empty((0, 2), dtype=float)
    return get_coordinates(segmentize(geometry, step_m))[:, :2]


def derive_flexure_widths(gz, step_m: float = 250.0,
                          quantile: float = 0.95) -> tuple[dict, dict]:
    """Estima larguras Up-Dn simetricamente, por geleira."""
    from scipy.spatial import cKDTree
    rows, widths = [], {}
    ice = gz[(gz["Type"] == "Ice Sheet") & gz["Boundary"].isin(["Up", "Dn"])]
    for name, group in ice.groupby("Name"):
        up, dn = group[group["Boundary"] == "Up"], group[group["Boundary"] == "Dn"]
        if up.empty or dn.empty:
            continue
        up_xy = _coordinates(up.geometry.union_all(), step_m)
        dn_xy = _coordinates(dn.geometry.union_all(), step_m)
        if not len(up_xy) or not len(dn_xy):
            continue
        distances = np.concatenate([cKDTree(dn_xy).query(up_xy)[0],
                                    cKDTree(up_xy).query(dn_xy)[0]])
        key = normalize_glacier_name(name)
        width = float(np.quantile(distances, quantile))
        widths[key] = width
        rows.append({"name": str(name), "key": key,
                     "n_distances": int(len(distances)),
                     "p50_m": float(np.quantile(distances, .50)),
                     "p90_m": float(np.quantile(distances, .90)),
                     "p95_m": float(np.quantile(distances, .95)),
                     "selected_quantile": float(quantile),
                     "selected_width_m": width})
    if not rows:
        raise ValueError("NSIDC-0778 sem pares Up/Dn utilizaveis na ROI.")
    fallback = float(np.median([r["selected_width_m"] for r in rows]))
    return widths, {"by_glacier": rows, "roi_fallback_width_m": fallback}


def width_for_glacier(name: str, widths: dict[str, float], fallback: float):
    key = normalize_glacier_name(name)
    if key in widths:
        return widths[key], True
    if key.startswith("pineisland"):
        matches = [v for k, v in widths.items() if k.startswith("pineisland")]
        if matches:
            return max(matches), True
    hits = [v for k, v in widths.items() if k and k in key]
    return (hits[0], True) if len(hits) == 1 else (fallback, False)


@dataclass
class GroundingFields:
    sx: np.ndarray
    sy: np.ndarray
    years: np.ndarray
    exact_zone: np.ndarray
    exact_support: np.ndarray
    exact_radius_m: np.ndarray
    exact_distance_m: np.ndarray
    historical_zone: np.ndarray
    historical_radius_m: np.ndarray
    historical_distance_m: np.ndarray
    width_report: dict


def _rasterize(geometries, sx, sy, dtype="uint8", fill=0):
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    dx, dy = abs(float(sx[1] - sx[0])), abs(float(sy[1] - sy[0]))
    transform = from_origin(float(sx[0]) - dx / 2, float(sy[-1]) + dy / 2, dx, dy)
    north_up = rasterize(geometries, out_shape=(len(sy), len(sx)),
                         transform=transform, fill=fill, dtype=dtype,
                         all_touched=True)
    return np.flipud(north_up)


def build_grounding_fields(sx, sy, bedmachine_mask, gl, gz,
                           years=range(2019, 2026), step_m: float = 250.0,
                           width_quantile: float = .95,
                           positional_uncertainty_m: float = 500.0):
    """Constroi campos raster anuais alinhados ao BedMachine (EPSG:3031)."""
    from scipy.ndimage import distance_transform_edt
    if gl.crs.to_epsg() != 3031:
        gl = gl.to_crs(3031)
    if gz.crs.to_epsg() != 3031:
        gz = gz.to_crs(3031)
    widths, report = derive_flexure_widths(gz, step_m, width_quantile)
    fallback = report["roi_fallback_width_m"]
    years = np.asarray(list(years), dtype=np.int16)
    shape = (len(years), len(sy), len(sx))
    zone = np.zeros(shape, bool)
    support = np.zeros(shape, np.uint8)
    radius = np.zeros(shape, np.float32)
    distance = np.full(shape, np.nan, np.float32)
    pixel_m = abs(float(sx[1] - sx[0]))
    for k, year in enumerate(years):
        subset = gl[gl["Year"].astype(int) == int(year)]
        buffered, radius_shapes, support_shapes, line_shapes = [], [], [], []
        for row in subset.itertuples():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            width, matched = width_for_glacier(row.Glac_Name, widths, fallback)
            rad = float(width + positional_uncertainty_m)
            polygon = geom.buffer(rad)
            buffered.append((polygon, 1))
            radius_shapes.append((polygon, rad))
            support_shapes.append((polygon, 2 if matched else 3))
            line_shapes.append((geom, 1))
        if not buffered:
            continue
        zone[k] = _rasterize(buffered, sx, sy).astype(bool)
        radius[k] = _rasterize(radius_shapes, sx, sy, "float32")
        support[k] = _rasterize(support_shapes, sx, sy)
        line = _rasterize(line_shapes, sx, sy).astype(bool)
        distance[k] = distance_transform_edt(~line).astype(np.float32) * pixel_m
    historical_zone = zone.any(axis=0)
    historical_radius = radius.max(axis=0)
    historical_lines = np.isfinite(distance) & (distance <= pixel_m)
    line_union = historical_lines.any(axis=0)
    historical_distance = (distance_transform_edt(~line_union).astype(np.float32) * pixel_m
                           if line_union.any() else np.full(bedmachine_mask.shape, np.nan, np.float32))
    report.update({"positional_uncertainty_m": float(positional_uncertainty_m),
                   "transition_radius_definition": "p95(Up-Dn) + positional_uncertainty",
                   "years_requested": years.astype(int).tolist(),
                   "years_with_any_line": years[np.any(zone, axis=(1, 2))].astype(int).tolist()})
    return GroundingFields(np.asarray(sx), np.asarray(sy), years, zone, support,
                           radius, distance, historical_zone, historical_radius,
                           historical_distance, report)


def classify_points(x, y, t_year, mask_class, fields: GroundingFields):
    """Classifica pontos por vizinho de grade e devolve arrays compactos."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    years = np.floor(np.asarray(t_year, float)).astype(np.int16)
    mask_class = np.asarray(mask_class)
    j = np.clip(np.rint((x-fields.sx[0])/(fields.sx[1]-fields.sx[0])).astype(int), 0, len(fields.sx)-1)
    i = np.clip(np.rint((y-fields.sy[0])/(fields.sy[1]-fields.sy[0])).astype(int), 0, len(fields.sy)-1)
    state = np.full(len(x), UNKNOWN, np.int8)
    state[mask_class == BM_GROUNDED_ICE] = GROUNDED_CONFIDENT
    state[mask_class == BM_FLOATING_ICE] = FLOATING_CONFIDENT
    support = np.where(np.isin(mask_class, [BM_GROUNDED_ICE, BM_FLOATING_ICE]), 1, 0).astype(np.uint8)
    line_year = np.zeros(len(x), np.int16)
    radius = np.zeros(len(x), np.float32)
    distance = fields.historical_distance_m[i, j].astype(np.float32, copy=True)
    historical = fields.historical_zone[i, j]
    state[historical], support[historical] = UNKNOWN, 4
    radius[historical] = fields.historical_radius_m[i[historical], j[historical]]
    for k, year in enumerate(fields.years):
        same = years == year
        if not same.any():
            continue
        exact = same & fields.exact_zone[k, i, j]
        state[exact] = GROUNDING_ZONE
        support[exact] = fields.exact_support[k, i[exact], j[exact]]
        line_year[exact] = year
        radius[exact] = fields.exact_radius_m[k, i[exact], j[exact]]
        distance[same] = fields.exact_distance_m[k, i[same], j[same]]
    return {"grounding_state": state, "grounding_support": support,
            "grounding_line_year": line_year, "dist_observed_gl_m": distance,
            "transition_radius_m": radius}
