"""Shared optimizer utilities for GRAPE coefficient optimizers (us / rad-per-us).

This module now focuses on bookkeeping helpers (history tracking, gradient
utilities, result containers) and re-exports the higher-level problem and
objective helpers located in :mod:`optimizers.problem` and
:mod:`optimizers.objectives`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np
import numpy.typing as npt

from .problem import GrapeControlProblem, build_grape_problem
from .objectives import evaluate_problem

NDArrayFloat = npt.NDArray[np.float64]

__all__ = [
    "NDArrayFloat",
    "StepStats",
    "make_step_stats",
    "OptimizerState",
    "GrapeControlProblem",
    "OptimizationOutput",
    "clip_gradients",
    "safe_norm",
    "build_grape_problem",
    "evaluate_problem",
    "history_to_arrays",
]


@dataclass(slots=True)
class StepStats:
    """Per-iteration metrics recorded by optimizers for logging."""

    iteration: int
    total: float
    terminal: float
    path: float
    ensemble: float
    power_penalty: float
    neg_penalty: float
    grad_norm: float
    step_norm: float
    lr: float
    wall_time_s: float
    calls_per_iter: int


def _init_history() -> Dict[str, list[Any]]:
    """Initialise the history dictionary used by :class:`OptimizerState`."""

    return {
        "iter": [],
        "total": [],
        "terminal": [],
        "path": [],
        "ensemble": [],
        "power_penalty": [],
        "neg_penalty": [],
        "grad_norm": [],
        "step_norm": [],
        "lr": [],
        "wall_time_s": [],
        "calls_per_iter": [],
    }


def make_step_stats(
    iteration: int,
    cost: Mapping[str, Any],
    grad_norm: float,
    step_norm: float,
    lr_value: float,
    wall_time: float,
    calls: int,
) -> StepStats:
    """Convenience wrapper to populate :class:`StepStats` from scalar metrics."""

    return StepStats(
        iteration=int(iteration),
        total=float(cost.get("total", 0.0)),
        terminal=float(cost.get("terminal", 0.0)),
        path=float(cost.get("path", 0.0)),
        ensemble=float(cost.get("ensemble", 0.0)),
        power_penalty=float(cost.get("power_penalty", 0.0)),
        neg_penalty=float(cost.get("neg_penalty", 0.0)),
        grad_norm=float(grad_norm),
        step_norm=float(step_norm),
        lr=float(lr_value),
        wall_time_s=float(wall_time),
        calls_per_iter=int(calls),
    )


@dataclass(slots=True)
class OptimizerState:
    """Mutable container tracking coefficients, gradients, and history."""

    coeffs: NDArrayFloat
    grad: NDArrayFloat
    history: Dict[str, list[Any]] = field(default_factory=_init_history)
    runtime_s: float = 0.0
    status: str = "in_progress"

    def record(self, stats: StepStats) -> None:
        """Append :class:`StepStats` values to the internal history lists."""

        self.history["iter"].append(int(stats.iteration))
        self.history["total"].append(float(stats.total))
        self.history["terminal"].append(float(stats.terminal))
        self.history["path"].append(float(stats.path))
        self.history["ensemble"].append(float(stats.ensemble))
        self.history["power_penalty"].append(float(stats.power_penalty))
        self.history["neg_penalty"].append(float(stats.neg_penalty))
        self.history["grad_norm"].append(float(stats.grad_norm))
        self.history["step_norm"].append(float(stats.step_norm))
        self.history["lr"].append(float(stats.lr))
        self.history["wall_time_s"].append(float(stats.wall_time_s))
        self.history["calls_per_iter"].append(int(stats.calls_per_iter))


@dataclass(slots=True)
class OptimizationOutput:
    """Structured result returned by optimizer front-ends."""

    coeffs: NDArrayFloat
    omega: NDArrayFloat
    delta: Optional[NDArrayFloat]
    cost_terms: Dict[str, float]
    history: Dict[str, NDArrayFloat]
    runtime_s: float
    optimizer_state: Dict[str, Any]
    extras: Dict[str, Any] | None = None


def clip_gradients(grad: NDArrayFloat, *, max_norm: float | None = None) -> NDArrayFloat:
    """Optionally clip gradient magnitude to ``max_norm``."""

    vec = np.asarray(grad, dtype=np.float64)
    if max_norm is None or max_norm <= 0.0:
        return vec
    norm = np.linalg.norm(vec)
    if norm == 0.0 or norm <= max_norm:
        return vec
    return vec * (max_norm / norm)


def safe_norm(arr: NDArrayFloat) -> float:
    """Return Euclidean norm, or ``inf`` if ``arr`` contains non-finite values."""

    vec = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(vec).all():
        return float("inf")
    return float(np.linalg.norm(vec))


def history_to_arrays(history: Dict[str, list[Any]]) -> Dict[str, NDArrayFloat]:
    """Convert iteration histories into NumPy arrays for serialization."""

    arrays: Dict[str, NDArrayFloat] = {}
    for key, values in history.items():
        if key in {"iter", "calls_per_iter"}:
            arrays[key] = np.asarray(values, dtype=np.int32)
        else:
            arrays[key] = np.asarray(values, dtype=np.float64)
    return arrays
