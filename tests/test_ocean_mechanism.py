from pathlib import Path

import numpy as np
import pandas as pd

from thwaites.ocean.bas_melt import (
    freezing_temperature,
    harmonic_summary,
    load_bas_melt,
    practical_salinity_from_conductivity,
)


def test_freezing_temperature_is_pressure_and_salinity_dependent():
    surface = freezing_temperature(np.array([30.0, 34.0]), 0.0)
    deep = freezing_temperature(np.array([30.0, 34.0]), 500.0)
    assert surface[1] < surface[0]
    assert np.all(deep < surface)
    assert -2.5 < deep[1] < -1.5


def test_pss78_standard_seawater_reference_point():
    salinity = practical_salinity_from_conductivity(42.9140, 15.0, 0.0)
    assert np.isclose(salinity, 35.0, atol=1e-4)


def test_load_bas_melt_without_hash_for_fixture(tmp_path: Path):
    path = tmp_path / "melt.dat"
    path.write_text(
        "Timestamp\tEastward velocity\tNorthward velocity\tTemperature\tConductivity\n"
        "[ISO861]\t[cm/s]\t[cm/s]\t[deg C]\t[PSU]\n"
        "2020-01-01T00:00:00Z\t2\t-3\t-0.5\t28\n"
        "2020-01-01T02:00:00Z\t4\t0\t-0.4\t29\n",
        encoding="utf-8")
    data = load_bas_melt(path, validate_hash=False)
    assert list(data.columns).count("thermal_driving_c") == 1
    assert len(data) == 2
    assert np.all(data["thermal_driving_c"] > 0)
    assert data["salinity_psu"].between(30.0, 38.0).all()
    assert np.isclose(data.loc[0, "speed_cm_s"], np.hypot(2, -3))


def test_harmonic_summary_recovers_known_amplitude():
    time = pd.date_range("2020-01-01", periods=240, freq="2h", tz="UTC")
    hours = np.arange(len(time)) * 2.0
    values = 3.0 * np.sin(2 * np.pi * hours / 12.0)
    result = harmonic_summary(time, values, periods_hours=(12.0,))
    assert np.isclose(result["12h"]["amplitude"], 3.0, atol=1e-8)
    assert result["12h"]["r2"] > 0.999999
