"""Optional visualization. Requires matplotlib (``simulsi[viz]``) or Plotly (``simulsi[plotly]``)."""

from simulsi.visualization.animation import animate_series
from simulsi.visualization.plots import (
    plot_comparison,
    plot_convergence,
    plot_distribution,
    plot_entity_trajectories,
    plot_queue_length,
    plot_sensitivity,
    plot_series,
    plot_timeline,
    plot_utilization,
    save_figure,
)

__all__ = [
    "animate_series",
    "plot_comparison",
    "plot_convergence",
    "plot_distribution",
    "plot_entity_trajectories",
    "plot_queue_length",
    "plot_sensitivity",
    "plot_series",
    "plot_timeline",
    "plot_utilization",
    "save_figure",
]
