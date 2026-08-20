"""Diagnósticos observacionais de vulnerabilidade glaciológica."""

from .vulnerability import (
    aggregate_basal_cells,
    along_flow_bed_slope,
    consensus_thinning,
    seasonal_basal_contrast,
    velocity_percent_trend,
)

__all__ = [
    "aggregate_basal_cells",
    "along_flow_bed_slope",
    "consensus_thinning",
    "seasonal_basal_contrast",
    "velocity_percent_trend",
]
