"""Plotting utilities for GRAPE/CRAB workflows (us, rad/us)."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from .physics import fidelity_pure, propagate_piecewise_const
from .result import Result

__all__ = [
    "plot_cost_history",
    "plot_pulses",
    "plot_summary",
    "plot_penalties_history",
    "plot_robustness_heatmap",
]

_TWO_PI = 2.0 * np.pi
_HEATMAP_MIN_DEFAULT = 1e-3
_HEATMAP_MAX_DEFAULT = 1.0


def _ensure_figures_dir(result: Result) -> Path:
    """Return the figures directory for ``result``, creating it if necessary.

    Parameters
    ----------
    result : Result
        Optimization result whose artifact directory contains the figures folder.

    Returns
    -------
    pathlib.Path
        Absolute path to the figures directory.
    """

    figures_dir = Path(result.artifacts_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _save_figure(fig: plt.Figure, directory: Path, stem: str, save_svg: bool = False) -> None:
    """Save a figure to PNG (and optionally SVG) with consistent formatting.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure instance to export.
    directory : pathlib.Path
        Destination directory for the image files.
    stem : str
        Filename stem used for the exported images.
    save_svg : bool, optional
        When ``True`` also export an SVG alongside the PNG.
    """

    directory.mkdir(parents=True, exist_ok=True)
    png_path = directory / f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if save_svg:
        svg_path = directory / f"{stem}.svg"
        fig.savefig(svg_path, bbox_inches="tight")


def plot_cost_history(result: Result, *, save: bool = True, ax: Axes | None = None) -> Axes:
    """Plot the total cost trajectory for a run and optionally write it to disk.

    Parameters
    ----------
    result : Result
        Optimization result containing the recorded cost history.
    save : bool, optional
        When ``True`` export the plot to the run's figures directory.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on; create a new figure when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plotted convergence curve.
    """

    history = result.history
    total_series = history.get("total")
    if total_series is None:
        raise ValueError("Result history does not contain 'total' values.")
    total = np.asarray(total_series, dtype=float)
    if total.size == 0:
        raise ValueError("Result history provides empty 'total' values.")
    iter_series = history.get("iter")
    if iter_series is None:
        # Fallback to implicit iteration index when history omits explicit entries.
        iterations = np.arange(1, total.size + 1, dtype=float)
    else:
        iterations = np.asarray(iter_series, dtype=float)
        if iterations.size != total.size:
            iterations = np.arange(1, total.size + 1, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.semilogy(iterations, total, label="total cost")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Cost convergence")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    if save:
        figures_dir = _ensure_figures_dir(result)
        _save_figure(fig, figures_dir, "cost_history")

    return ax


def plot_penalties_history(result: Result, *, save: bool = True, ax: Axes | None = None) -> Axes:
    """Plot logged penalty terms on a log axis, handling missing histories.

    Parameters
    ----------
    result : Result
        Optimization result containing penalty traces (if recorded).
    save : bool, optional
        When ``True`` export the plot to the run's figures directory.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on; create a new figure when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the rendered penalty curves or placeholder text.
    """

    history = result.history
    penalty_keys = [key for key in history.keys() if key.endswith("_penalty")]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if not penalty_keys:
        ax.axis("off")
        ax.text(0.5, 0.5, "No penalty terms recorded", ha="center", va="center", fontsize=11)
        if save:
            figures_dir = _ensure_figures_dir(result)
            _save_figure(fig, figures_dir, "penalties_history")
        return ax

    iter_series = history.get("iter")
    first_series = np.asarray(history[penalty_keys[0]], dtype=float)
    if iter_series is None:
        # Fallback to implicit iteration index when history omits explicit entries.
        iterations = np.arange(1, first_series.size + 1, dtype=float)
    else:
        iterations = np.asarray(iter_series, dtype=float)
        if iterations.size != first_series.size:
            iterations = np.arange(1, first_series.size + 1, dtype=float)

    for key in penalty_keys:
        values = np.asarray(history[key], dtype=float)
        positive = np.clip(values, 1e-12, None)
        label = key.replace("_", " ")
        ax.semilogy(iterations, positive, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Penalty value")
    ax.set_title("Penalty terms")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    if save:
        figures_dir = _ensure_figures_dir(result)
        _save_figure(fig, figures_dir, "penalties_history")

    return ax


def plot_pulses(result: Result, *, save: bool = True, axes: Sequence[Axes] | None = None) -> Sequence[Axes]:
    """Plot optimized pulses against their baselines for quick inspection.

    Parameters
    ----------
    result : Result
        Optimization result containing pulse data and baselines.
    save : bool, optional
        When ``True`` export the figure to the run's figures directory.
    axes : Sequence[matplotlib.axes.Axes], optional
        Pre-existing axes to reuse; create a new figure when ``None``.

    Returns
    -------
    Sequence[matplotlib.axes.Axes]
        Axes displaying the optimized and baseline controls.
    """

    pulses = result.pulses
    t_us = np.asarray(pulses["t_us"], dtype=float)
    omega = np.asarray(pulses["omega"], dtype=float)
    omega_base = np.asarray(pulses.get("omega_base", np.zeros_like(omega)), dtype=float)
    delta = pulses.get("delta")
    delta_base = pulses.get("delta_base")

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    else:
        fig = axes[0].figure
    ax_omega, ax_delta = axes

    ax_omega.plot(t_us, omega, label="Optimized Omega(t)")
    ax_omega.plot(t_us, omega_base, ls="--", label="Baseline Omega(t)")
    ax_omega.set_ylabel("Omega (rad/us)")
    ax_omega.grid(True, ls=":", alpha=0.6)
    ax_omega.legend()

    if delta is not None:
        ax_delta.plot(t_us, np.asarray(delta, dtype=float), label="Optimized Delta(t)")
    if delta_base is not None:
        ax_delta.plot(t_us, np.asarray(delta_base, dtype=float), ls="--", label="Baseline Delta(t)")
    ax_delta.set_xlabel("Time ($\mu$s)")
    ax_delta.set_ylabel("Delta (rad/us)")
    ax_delta.grid(True, ls=":", alpha=0.6)
    if delta is not None or delta_base is not None:
        ax_delta.legend()

    fig.tight_layout()

    if save:
        figures_dir = _ensure_figures_dir(result)
        _save_figure(fig, figures_dir, "pulses")

    return axes


def plot_summary(result: Result, *, save: bool = True, ax: Axes | None = None) -> Axes:
    """Render a compact metrics table summarizing the optimization outcome.

    Parameters
    ----------
    result : Result
        Optimization result whose scalar metrics should be displayed.
    save : bool, optional
        When ``True`` export the table to the run's figures directory.
    ax : matplotlib.axes.Axes, optional
        Existing axes to render into; create a new figure when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the rendered metrics table.
    """

    metrics = result.final_metrics
    rows = [
        ("terminal", metrics.get("terminal")),
        ("path", metrics.get("path")),
        ("ensemble", metrics.get("ensemble")),
        ("power_penalty", metrics.get("power_penalty")),
        ("neg_penalty", metrics.get("neg_penalty")),
        ("total", metrics.get("total")),
        ("runtime_s", metrics.get("runtime_s")),
    ]
    display_rows = [(name, None if value is None else f"{value:.6g}") for name, value in rows if value is not None]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 2.6))
    else:
        fig = ax.figure

    ax.axis("off")
    table = ax.table(
        cellText=[[name, val] for name, val in display_rows],
        colLabels=["Metric", "Value"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.set_title("Results summary")

    if save:
        figures_dir = _ensure_figures_dir(result)
        _save_figure(fig, figures_dir, "summary")

    return ax


def plot_robustness_heatmap(
    pulse: Mapping[str, np.ndarray],
    t_us: np.ndarray,
    delta_base: np.ndarray,
    detuning_MHz_grid: np.ndarray,
    area_pi_grid: np.ndarray,
    label: str,
    *,
    psi0: np.ndarray,
    target: np.ndarray,
    save_dir: Path,
    ax: Axes | None = None,
    save: bool = True,
    log: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    save_svg: bool = True,
    add_colorbar: bool = True,
) -> tuple[Axes, np.ndarray]:
    """Map terminal infidelity across detuning and pulse-area perturbations.

    Parameters
    ----------
    pulse : Mapping[str, numpy.ndarray]
        Dictionary containing at least ``omega`` (and optionally ``delta``) pulse envelopes.
    t_us : numpy.ndarray
        Time grid in microseconds matching the pulse arrays.
    delta_base : numpy.ndarray
        Baseline detuning used when the pulse does not supply ``delta`` explicitly.
    detuning_MHz_grid : numpy.ndarray
        Grid of detuning offsets in MHz on which robustness is evaluated.
    area_pi_grid : numpy.ndarray
        Grid of scaling factors applied to the pulse area (in units of ``pi``).
    label : str
        Identifier used when saving the generated figures and arrays.
    psi0 : numpy.ndarray
        Initial state for propagation.
    target : numpy.ndarray
        Target state used to evaluate terminal fidelity.
    save_dir : pathlib.Path
        Directory where figures and robustness arrays are stored.
    ax : matplotlib.axes.Axes, optional
        Existing axes to reuse; create a new figure when ``None``.
    save : bool, optional
        Whether to persist the rendered figure and robustness data.
    log : bool, optional
        If ``True``, use logarithmic colour scaling.
    vmin, vmax : float, optional
        Explicit colour limits; defaults computed from positive entries.
    cmap : str, optional
        Matplotlib colormap name used for the heatmap.
    save_svg : bool, optional
        When ``True`` also export a vector SVG image.
    add_colorbar : bool, optional
        Toggle the colourbar display.

    Returns
    -------
    tuple[matplotlib.axes.Axes, numpy.ndarray]
        The axes containing the plot and the computed infidelity grid.
    """

    """Generate a robustness heatmap over detuning offsets and pulse areas."""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    omega = np.asarray(pulse["omega"], dtype=float)
    delta = np.asarray(pulse.get("delta", delta_base), dtype=float)
    t_us = np.asarray(t_us, dtype=float)
    delta_base = np.asarray(delta_base, dtype=float)
    detuning_MHz_grid = np.asarray(detuning_MHz_grid, dtype=float)
    area_pi_grid = np.asarray(area_pi_grid, dtype=float)

    if omega.shape != t_us.shape:
        raise ValueError("Omega pulse must share the same shape as the time grid.")
    if delta.shape != t_us.shape:
        raise ValueError("Delta pulse must share the same shape as the time grid.")

    dt_us = float(np.diff(t_us).mean())
    base_area_pi = np.trapz(np.abs(omega), t_us) / np.pi
    if base_area_pi <= 0.0:
        raise ValueError("Baseline pulse area must be positive.")

    psi0 = np.asarray(psi0, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)

    detuning_offsets = detuning_MHz_grid * _TWO_PI
    area_count = area_pi_grid.size
    detuning_count = detuning_MHz_grid.size
    infidelity = np.empty((area_count, detuning_count), dtype=float)

    omega_norm = omega / base_area_pi

    for i, area_scale in enumerate(area_pi_grid):
        omega_scaled = area_scale * omega_norm
        for j, detuning in enumerate(detuning_offsets):
            delta_shifted = delta + detuning
            prop = propagate_piecewise_const(
                omega_scaled,
                delta_shifted,
                dt_us,
                psi0=psi0,
            )
            psi_T = prop["psi_T"]
            fidelity = fidelity_pure(psi_T, target)
            infidelity[i, j] = max(1.0 - fidelity, 0.0)

    X, Y = np.meshgrid(detuning_MHz_grid, area_pi_grid)

    finite_vals = infidelity[np.isfinite(infidelity) & (infidelity > 0.0)]
    # Ensure colour-scale bounds remain positive for logarithmic normalisation.
    if vmin is None:
        lower_bound = max(float(finite_vals.min(initial=_HEATMAP_MIN_DEFAULT)), _HEATMAP_MIN_DEFAULT)
    else:
        lower_bound = float(vmin)
    if not np.isfinite(lower_bound) or lower_bound <= 0.0:
        lower_bound = _HEATMAP_MIN_DEFAULT

    if vmax is None:
        upper_bound = max(float(finite_vals.max(initial=_HEATMAP_MAX_DEFAULT)), _HEATMAP_MAX_DEFAULT)
    else:
        upper_bound = float(vmax)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        upper_bound = _HEATMAP_MAX_DEFAULT
    if upper_bound <= lower_bound:
        upper_bound = max(lower_bound * 10.0, _HEATMAP_MAX_DEFAULT)

    norm = LogNorm(vmin=lower_bound, vmax=upper_bound) if log else None

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    c = ax.pcolormesh(X, Y, infidelity, shading="auto", cmap=cmap, norm=norm)
    ax.set_xlabel("Detuning offset Delta_0 (MHz)")
    ax.set_ylabel("Pulse area (pi units)")
    ax.set_title(f"Terminal infidelity - {label}")
    if add_colorbar:
        cb = fig.colorbar(c, ax=ax)
        cb.set_label("Terminal infidelity")

    if save:
        stem = f"heatmap_terminal_vs_detuning_area_{label}"
        _save_figure(fig, save_dir, stem, save_svg=save_svg)
        np.savez(
            save_dir / f"robustness_terminal_vs_detuning_area_{label}.npz",
            detuning_MHz=detuning_MHz_grid,
            area_pi=area_pi_grid,
            infidelity=infidelity,
        )

    return ax, infidelity
