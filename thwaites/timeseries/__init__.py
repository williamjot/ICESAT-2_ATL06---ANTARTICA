"""Análise temporal: dh/dt (fitsec), séries por nó e tendência formal."""

from thwaites.timeseries.dhdt import compute_tile_dhdt, run_dhdt
from thwaites.timeseries.build import build_node_series
from thwaites.timeseries.trend import mann_kendall_sen, compute_trends
from thwaites.timeseries.model import (
    fit_model, select_model, check_identifiability, compare_jja_annual,
)
from thwaites.timeseries.acceleration import (
    assess_acceleration, acceleration_field, AccelCriteria,
)

__all__ = ["compute_tile_dhdt", "run_dhdt",
           "build_node_series", "mann_kendall_sen", "compute_trends",
           "fit_model", "select_model", "check_identifiability", "compare_jja_annual",
           "assess_acceleration", "acceleration_field", "AccelCriteria"]
