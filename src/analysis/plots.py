"""Reusable plotting utilities built on top of :mod:`src.plotting`."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..plotting import save_figure as base_save_figure
from .metrics import ResultSummary
from .trajectories import TrajectoryBundle

DEFAULT_COLORS = plt.rcParams.get("axes.prop_cycle", None)
if DEFAULT_COLORS is not None:
    DEFAULT_COLORS = DEFAULT_COLORS.by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c"])
else:
    DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def _resolve_labels(summaries: Sequence[ResultSummary], labels: Sequence[str] | None) -> list[str]:
    if labels is not None:
        return list(labels)
    resolved = []
    for summary in summaries:
        label = summary.metadata.get("objective")
        if not label:
            label = summary.metadata.get("method")
        if not label:
            label = summary.run_name
        resolved.append(str(label))
    return resolved


def plot_cost_histories(
    summaries: Sequence[ResultSummary],
    *,
    ax: plt.Axes | None = None,
    logy: bool = True,
    labels: Sequence[str] | None = None,
) -> plt.Axes:
    """Overlay total cost trajectories for multiple results."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    labels = _resolve_labels(summaries, labels)
    for idx, summary in enumerate(summaries):
        history = summary.history.total
        iterations = summary.history.iterations
        if history is None:
            continue
        iters = np.arange(1, history.size + 1, dtype=np.int64) if iterations is None else iterations
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        plot_func = ax.semilogy if logy else ax.plot
        plot_func(iters, history, label=labels[idx], color=color)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Total cost vs iteration")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return ax


def plot_cost_components(
    summaries: Sequence[ResultSummary],
    *,
    ax: plt.Axes | None = None,
    labels: Sequence[str] | None = None,
) -> plt.Axes:
    """Plot terminal/power/neg penalty components on a log scale."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    labels = _resolve_labels(summaries, labels)
    for idx, summary in enumerate(summaries):
        history = summary.history
        iterations = history.iterations
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        if history.terminal is not None:
            iters = np.arange(1, history.terminal.size + 1, dtype=np.int64) if iterations is None else iterations
            ax.semilogy(iters, history.terminal, label=f"{labels[idx]}: terminal", color=color)
        if history.power_penalty is not None:
            iters = np.arange(1, history.power_penalty.size + 1, dtype=np.int64) if iterations is None else iterations
            ax.semilogy(iters, history.power_penalty, ls="--", color=color, alpha=0.8, label=f"{labels[idx]}: power")
        if history.neg_penalty is not None:
            iters = np.arange(1, history.neg_penalty.size + 1, dtype=np.int64) if iterations is None else iterations
            ax.semilogy(iters, history.neg_penalty, ls=":", color=color, alpha=0.8, label=f"{labels[idx]}: neg")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost component")
    ax.set_title("Cost component trajectories")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return ax


def plot_runtime_oracle_bars(
    summaries: Sequence[ResultSummary],
    *,
    axes: Sequence[plt.Axes] | None = None,
    labels: Sequence[str] | None = None,
) -> Sequence[plt.Axes]:
    """Render side-by-side bars for runtimes and oracle counts."""

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)
    else:
        fig = axes[0].figure

    labels = _resolve_labels(summaries, labels)
    runtimes = [summary.runtime_s for summary in summaries]
    oracle_calls = [summary.oracle_calls for summary in summaries]

    x = np.arange(len(summaries))
    colors = [DEFAULT_COLORS[idx % len(DEFAULT_COLORS)] for idx in range(len(summaries))]

    axes[0].bar(x, runtimes, color=colors)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_title("Runtime by method")
    axes[0].grid(True, axis="y", ls=":", alpha=0.4)

    axes[1].bar(x, oracle_calls, color=colors)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Oracle calls")
    axes[1].set_title("Oracle evaluations by method")
    axes[1].grid(True, axis="y", ls=":", alpha=0.4)

    fig.tight_layout()
    return axes


def plot_pulse_overlay(
    summaries: Sequence[ResultSummary],
    *,
    channel: str = "omega",
    ax: plt.Axes | None = None,
    labels: Sequence[str] | None = None,
) -> plt.Axes:
    """Overlay optimized pulses for the specified control channel."""

    if channel not in {"omega", "delta"}:
        raise ValueError("channel must be 'omega' or 'delta'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

    labels = _resolve_labels(summaries, labels)
    for idx, summary in enumerate(summaries):
        pulses = summary.pulses
        t_us = pulses.t_us
        if channel == "omega":
            waveform = pulses.omega
            baseline = pulses.omega_base
            ylabel = "Omega (rad/us)"
        else:
            waveform = pulses.delta if pulses.delta is not None else np.zeros_like(pulses.omega)
            baseline = pulses.delta_base
            ylabel = "Delta (rad/us)"

        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        ax.plot(t_us, waveform, label=labels[idx], color=color)
        if baseline is not None and idx == 0:
            ax.plot(t_us, baseline, color="#555555", ls="--", label="baseline")

    ax.set_xlabel("Time (µs)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{channel.capitalize()} pulses")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return ax


def plot_population_traces(
    bundles: Mapping[str, TrajectoryBundle],
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot excited-state population trajectories for labelled bundles."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

    for idx, (label, bundle) in enumerate(bundles.items()):
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        pop = np.asarray(bundle.pop_excited, dtype=np.float64)
        t_us = bundle.t_us
        if pop.shape[0] == t_us.size + 1:
            pop = pop[:-1]
        ax.plot(t_us[: pop.size], pop, label=label, color=color)

    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Excited population")
    ax.set_title("Population trajectories")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return ax


def render_summary_table(
    summaries: Sequence[ResultSummary],
    *,
    ax: plt.Axes | None = None,
    labels: Sequence[str] | None = None,
) -> plt.Axes:
    """Render a text table summarising key metrics."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure

    columns = [
        "label",
        "total",
        "terminal",
        "runtime_s",
        "iterations",
        "oracle_calls",
        "max|omega|",
        "area/pi",
    ]
    labels = _resolve_labels(summaries, labels)
    rows: list[list[str]] = []
    for label, summary in zip(labels, summaries):
        metrics = summary.metrics
        rows.append(
            [
                label,
                f"{metrics.get('total', np.nan):.3e}",
                f"{metrics.get('terminal', np.nan):.3e}",
                f"{summary.runtime_s:.2f}",
                f"{summary.iterations:d}",
                f"{summary.oracle_calls:d}",
                f"{summary.max_abs_omega:.3f}",
                f"{summary.area_omega_over_pi:.3f}",
            ]
        )

    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    fig.tight_layout()
    return ax


def render_step_size_dashboard(
    summaries: Sequence[ResultSummary],
    *,
    pulse_channel: str = "omega",
    labels: Sequence[str] | None = None,
    out_dir: Path | None = None,
    save_svg: bool = False,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Render a composite dashboard used by the step-size notebook."""

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    ax_cost = fig.add_subplot(gs[0, 0])
    ax_runtime = fig.add_subplot(gs[0, 1])
    ax_oracle = fig.add_subplot(gs[0, 2])
    ax_pulses = fig.add_subplot(gs[1, 0:2])
    ax_summary = fig.add_subplot(gs[1, 2])

    labels = _resolve_labels(summaries, labels)

    axis_map: dict[str, plt.Axes] = {}
    axis_map["cost"] = plot_cost_histories(summaries, ax=ax_cost, labels=labels)
    runtime_ax, oracle_ax = plot_runtime_oracle_bars(
        summaries, axes=(ax_runtime, ax_oracle), labels=labels
    )
    axis_map["runtime"] = runtime_ax
    axis_map["oracle"] = oracle_ax
    axis_map["pulses"] = plot_pulse_overlay(
        summaries, channel=pulse_channel, ax=ax_pulses, labels=labels
    )
    axis_map["summary_table"] = render_summary_table(summaries, ax=ax_summary, labels=labels)

    fig.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_dir, "step_size_dashboard", save_svg=save_svg)

    return fig, axis_map


def save_figure(fig: plt.Figure, directory: Path, stem: str, *, save_svg: bool = False, show: bool = False) -> None:
    """Proxy to :func:`src.plotting.save_figure` for convenience."""

    base_save_figure(fig, Path(directory), stem, save_svg=save_svg, show=show)


__all__ = [
    "plot_cost_histories",
    "plot_cost_components",
    "plot_runtime_oracle_bars",
    "plot_pulse_overlay",
    "plot_population_traces",
    "render_step_size_dashboard",
    "render_summary_table",
    "save_figure",
]
