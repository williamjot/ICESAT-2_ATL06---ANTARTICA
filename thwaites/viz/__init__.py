"""Figuras: mapa de dh/dt, histograma, incerteza, tendência e diagramas glaciológicos."""

from thwaites.viz.figures import (
    fig_dhdt_map, fig_dhdt_hist, fig_trend_significance, fig_xover_validation,
    fig_uncertainty_map, fig_dhdt_with_confidence,
)
from thwaites.viz.glaciology import (
    fig_basal_melt_map, fig_dhdt_vs_velocity, fig_mass_budget,
    sample_velocity_at,
)

__all__ = ["fig_dhdt_map", "fig_dhdt_hist", "fig_trend_significance",
           "fig_xover_validation", "fig_uncertainty_map", "fig_dhdt_with_confidence",
           "fig_basal_melt_map", "fig_dhdt_vs_velocity", "fig_mass_budget",
           "sample_velocity_at"]
