"""High-level analysis helpers shared by the HB-GRAPE notebooks."""
from __future__ import annotations

from .metrics import (
    HistorySeries,
    PulseSummary,
    ResultSummary,
    collect_history_series,
    load_result_summary,
    summarize_result,
)
from .trajectories import (
    TrajectoryBundle,
    TrajectoryRequest,
    compute_bloch_bundle,
    compute_population_trace,
    compute_trajectory_bundle,
)
from .plots import (
    plot_cost_components,
    plot_cost_histories,
    plot_population_traces,
    plot_pulse_overlay,
    plot_runtime_oracle_bars,
    render_step_size_dashboard,
    render_summary_table,
    save_figure,
)

__all__ = [
    "HistorySeries",
    "PulseSummary",
    "ResultSummary",
    "TrajectoryBundle",
    "TrajectoryRequest",
    "collect_history_series",
    "load_result_summary",
    "summarize_result",
    "compute_bloch_bundle",
    "compute_population_trace",
    "compute_trajectory_bundle",
    "plot_cost_histories",
    "plot_cost_components",
    "plot_runtime_oracle_bars",
    "plot_pulse_overlay",
    "plot_population_traces",
    "render_step_size_dashboard",
    "render_summary_table",
    "save_figure",
]
