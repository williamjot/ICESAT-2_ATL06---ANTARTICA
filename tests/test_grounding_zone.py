import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from thwaites.qc.grounding_zone import (
    FLOATING_CONFIDENT, GROUNDED_CONFIDENT, GROUNDING_ZONE, UNKNOWN,
    build_grounding_fields, classify_points, derive_flexure_widths,
)


def _products():
    gl = gpd.GeoDataFrame(
        {"Glac_Name": ["Thwaites"], "Year": [2020]},
        geometry=[LineString([(0, -1000), (0, 1000)])], crs=3031)
    gz = gpd.GeoDataFrame(
        {"Name": ["Thwaites", "Thwaites"], "Type": ["Ice Sheet"] * 2,
         "Boundary": ["Up", "Dn"]},
        geometry=[LineString([(-500, -1000), (-500, 1000)]),
                  LineString([(500, -1000), (500, 1000)])], crs=3031)
    return gl, gz


def test_flexure_width_is_empirical():
    _, gz = _products()
    widths, report = derive_flexure_widths(gz, step_m=100, quantile=.95)
    assert 900 <= widths["thwaites"] <= 1100
    assert report["roi_fallback_width_m"] == widths["thwaites"]


def test_exact_year_and_missing_year_are_not_interpolated():
    gl, gz = _products()
    sx, sy = np.arange(-3000, 3001, 100.), np.arange(-2000, 2001, 100.)
    bm = np.where(sx[None, :] < 0, 2, 3).repeat(len(sy), axis=0)
    fields = build_grounding_fields(
        sx, sy, bm, gl, gz, years=[2020, 2021], step_m=100,
        width_quantile=.95, positional_uncertainty_m=100)
    exact = classify_points([-2000, 0, 2000], [0, 0, 0], [2020.5] * 3,
                            [2, 2, 3], fields)
    assert exact["grounding_state"].tolist() == [
        GROUNDED_CONFIDENT, GROUNDING_ZONE, FLOATING_CONFIDENT]
    missing = classify_points([-2000, 0, 2000], [0, 0, 0], [2021.5] * 3,
                              [2, 2, 3], fields)
    assert missing["grounding_state"].tolist() == [
        GROUNDED_CONFIDENT, UNKNOWN, FLOATING_CONFIDENT]
    assert missing["grounding_line_year"].tolist() == [0, 0, 0]
