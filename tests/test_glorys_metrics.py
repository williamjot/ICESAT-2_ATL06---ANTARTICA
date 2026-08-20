import numpy as np
import pandas as pd
import xarray as xr

from thwaites.ocean.glorys import compute_jja_metrics, layer_thickness


def _dataset():
    time = pd.to_datetime([
        "2019-06-15", "2019-07-15", "2019-08-15",
        "2020-06-15", "2020-07-15", "2020-08-15",
    ])
    depth = np.array([100.0, 300.0, 500.0, 700.0, 900.0])
    latitude = np.array([-75.5, -75.0])
    longitude = np.array([-110.0, -105.0])
    shape = (len(time), len(depth), len(latitude), len(longitude))
    warming = np.linspace(-1.2, 1.2, len(depth))[None, :, None, None]
    theta = np.broadcast_to(warming, shape).copy()
    salinity = np.full(shape, 34.7)
    u = np.full(shape, 0.03)
    v = np.full(shape, 0.04)
    return xr.Dataset(
        {"thetao": (("time", "depth", "latitude", "longitude"), theta),
         "so": (("time", "depth", "latitude", "longitude"), salinity),
         "uo": (("time", "depth", "latitude", "longitude"), u),
         "vo": (("time", "depth", "latitude", "longitude"), v)},
        coords={"time": time, "depth": depth,
                "latitude": latitude, "longitude": longitude})


def test_layer_thickness_is_positive():
    result = layer_thickness(np.array([10.0, 50.0, 200.0, 700.0, 900.0]))
    assert np.all(result > 0)
    assert result.sum() <= 1000.0


def test_compute_jja_metrics_shapes_and_units():
    result = compute_jja_metrics(_dataset())
    assert result.sizes == {"year": 2, "latitude": 2, "longitude": 2}
    assert np.all(result.potential_heat_content_400_1000.values > 0)
    assert np.allclose(result.speed_400_1000.values, 0.05, atol=1e-6)
    assert result.thermal_driving_400_1000.attrs["units"] == "degree_Celsius"


def test_rejects_non_jja_months():
    source = _dataset().copy()
    source = source.assign_coords(time=pd.to_datetime([
        "2019-05-15", "2019-07-15", "2019-08-15",
        "2020-06-15", "2020-07-15", "2020-08-15",
    ]))
    try:
        compute_jja_metrics(source)
    except ValueError as error:
        assert "JJA" in str(error)
    else:
        raise AssertionError("mês fora de JJA deveria ser rejeitado")

