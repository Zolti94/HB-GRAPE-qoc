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


def _ensure_figures_dir(result: Result) -> Path:
    figures_dir = Path(result.artifacts_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _save_figure(fig: plt.Figure, directory: Path, stem: str, save_svg: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    png_path = directory / f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if save_svg:
        svg_path = directory / f"{stem}.svg"
        fig.savefig(svg_path, bbox_inches="tight")


def plot_cost_history(result: Result, *, save: bool = True, ax: Axes | None = None) -> Axes:
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
    if vmin is None:
        vmin = float(finite_vals.min(initial=1e-8))
    if vmax is None:
        vmax = float(finite_vals.max(initial=1.0))

    norm = LogNorm(vmin=max(vmin, 1e-8), vmax=max(vmax, 1e-6)) if log else None

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
