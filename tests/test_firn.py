"""
Testes da correção de firn.

O teste central: a correção precisa mover a massa na direção FÍSICA correta.
Firn compactando (dFAC/dt < 0) baixa a superfície sem perder massa, então o
balanço bruto SUPERESTIMA a perda e a correção tem de reduzi-la.
"""

import numpy as np
import pandas as pd
import pytest

from thwaites.corrections.firn import (
    firn_sensitivity, apply_firn_correction, firn_rate_at, load_firn,
    resolve_firn_path,
)


def _grid(cfg, n=100, dhdt=-1.0, var=0.04):
    res = cfg.interpolation.grid_res_m
    return pd.DataFrame({
        "x": ((np.arange(n) % 10) * res).astype(float),
        "y": ((np.arange(n) // 10) * res).astype(float),
        "pred": np.full(n, dhdt), "var": np.full(n, var),
    })


def _make_firn_nc(path, dfac_dt=-0.05, x0=0.0, y0=0.0, n=12, res=12_500.0,
                  years=(2019, 2023)):
    """NetCDF sintético do FDM: FAC com tendência linear conhecida."""
    import xarray as xr

    x = x0 + np.arange(n) * res
    y = y0 + np.arange(n) * res
    t = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-01-01", freq="30D")
    ty = t.year + (t.dayofyear - 1) / 365.25
    fac = (10.0 + dfac_dt * (ty - ty[0]).to_numpy()[:, None, None]
           * np.ones((1, len(y), len(x))))
    ds = xr.Dataset({"FAC": (("time", "y", "x"), fac)},
                    coords={"time": t, "y": y, "x": x})
    ds.to_netcdf(path)
    return path


# --------------------------------------------------------------- sensibilidade
def test_sensitivity_spans_and_signs(cfg):
    g = _grid(cfg, dhdt=-1.0)
    s = firn_sensitivity(g, cfg, dfac_rates=(-0.1, 0.0, 0.1))
    assert len(s) == 3
    # dFAC/dt negativo (firn compactando) -> dh_gelo menos negativo -> perda MENOR
    row_neg = s[s["dfac_dt_m_yr"] == -0.1].iloc[0]
    row_zero = s[s["dfac_dt_m_yr"] == 0.0].iloc[0]
    row_pos = s[s["dfac_dt_m_yr"] == 0.1].iloc[0]
    assert row_neg["dMdt_Gt_yr"] > row_zero["dMdt_Gt_yr"] > row_pos["dMdt_Gt_yr"]
    assert "delta_vs_zero_Gt_yr" in s.columns


def test_sensitivity_magnitude_is_proportional(cfg):
    """O efeito tem de escalar com dFAC/dt (é uma subtração linear)."""
    g = _grid(cfg, dhdt=-1.0)
    s = firn_sensitivity(cfg=cfg, grid_df=g, dfac_rates=(0.0, 0.05, 0.10))
    d1 = abs(s.iloc[1]["delta_vs_zero_Gt_yr"])
    d2 = abs(s.iloc[2]["delta_vs_zero_Gt_yr"])
    assert np.isclose(d2 / d1, 2.0, rtol=0.02)


# ------------------------------------------------------------------- correção
def test_firn_rate_recovers_known_trend(cfg, tmp_path):
    p = _make_firn_nc(tmp_path / "firn.nc", dfac_dt=-0.05)
    px = np.array([30_000.0, 60_000.0])
    py = np.array([30_000.0, 60_000.0])
    rate, info = firn_rate_at(px, py, cfg, path=p)
    assert np.allclose(rate, -0.05, atol=1e-3)
    assert info["fac_variable"] == "FAC"
    assert info["n_epochs_used"] > 10


def test_correction_moves_mass_in_physical_direction(cfg, tmp_path):
    """Firn compactando => a correção REDUZ a perda estimada."""
    p = _make_firn_nc(tmp_path / "firn.nc", dfac_dt=-0.05)
    g = _grid(cfg, dhdt=-1.0)
    out, info = apply_firn_correction(g, cfg, path=p)
    assert "dhdt_ice" in out.columns and "dfac_dt" in out.columns
    # dh_gelo = -1.0 - (-0.05) = -0.95 -> menos negativo
    assert np.allclose(out["dhdt_ice"], -0.95, atol=1e-3)
    assert info["dhdt_ice_median"] > info["dhdt_raw_median"]


def test_extrapolation_is_flagged(cfg, tmp_path):
    """FDM que termina antes do fim do projeto tem de marcar extrapolação."""
    p = _make_firn_nc(tmp_path / "firn.nc", years=(2019, 2022))
    _, info = firn_rate_at(np.array([30_000.0]), np.array([30_000.0]), cfg, path=p)
    assert info["extrapolated"] is True
    assert info["temporal_coverage_frac"] < 1.0


def test_missing_variable_lists_available(cfg, tmp_path):
    """Não adivinha o nome da variável — reclama listando o que existe."""
    import xarray as xr
    p = tmp_path / "bad.nc"
    xr.Dataset({"outra_coisa": (("y", "x"), np.zeros((3, 3)))},
               coords={"y": [0, 1, 2], "x": [0, 1, 2]}).to_netcdf(p)
    with pytest.raises(ValueError) as e:
        load_firn(cfg, path=p)
    assert "outra_coisa" in str(e.value)


def test_missing_file_points_to_fetch(cfg):
    cfg.firn.path = "nao_existe.nc"
    with pytest.raises(FileNotFoundError) as e:
        load_firn(cfg)
    assert "fetch_firn" in str(e.value)
