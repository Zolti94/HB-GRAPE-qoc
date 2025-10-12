"""Propagation helpers shared by multi-notebook analysis."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from ..crab_notebook_utils import (
    bloch_coordinates,
    ground_state_projectors,
    population_excited,
)
from ..physics import propagate_piecewise_const
from .metrics import PulseSummary, ResultSummary


def _as_array(payload: Any, *, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(payload, dtype=dtype)


@dataclass(slots=True)
class TrajectoryRequest:
    """Context needed to propagate pulses."""

    psi0: np.ndarray
    dt_us: float | None = None
    t_us: np.ndarray | None = None
    omega_base: np.ndarray | None = None
    delta_base: np.ndarray | None = None


@dataclass(slots=True)
class TrajectoryBundle:
    """Container holding propagated trajectories and Bloch data."""

    t_us: np.ndarray
    psi_path: np.ndarray
    rho_path: np.ndarray
    pop_excited: np.ndarray
    bloch: np.ndarray | None = None
    bloch_ground: np.ndarray | None = None


def _prepare_dt(request: TrajectoryRequest, pulses: PulseSummary) -> tuple[np.ndarray, float]:
    t_us = request.t_us if request.t_us is not None else pulses.t_us
    if t_us.size == 0:
        raise ValueError("Time grid is empty; cannot propagate trajectories.")
    if request.dt_us is not None:
        dt_us = float(request.dt_us)
    else:
        if t_us.size < 2:
            raise ValueError("Need at least two samples to infer dt_us.")
        dt_us = float(np.mean(np.diff(t_us)))
    return _as_array(t_us), dt_us


def compute_trajectory_bundle(
    summary: ResultSummary,
    request: TrajectoryRequest,
    *,
    use_delta_baseline_when_missing: bool = True,
) -> TrajectoryBundle:
    """Propagate the optimized pulse stored in ``summary``."""

    pulses = summary.pulses
    t_us, dt_us = _prepare_dt(request, pulses)

    omega = _as_array(pulses.omega)
    delta = pulses.delta
    if delta is None and use_delta_baseline_when_missing:
        delta = pulses.delta_base
    if delta is None:
        delta_arr = np.zeros_like(omega)
    else:
        delta_arr = _as_array(delta)

    propagation = propagate_piecewise_const(
        omega,
        delta_arr,
        dt_us,
        psi0=np.asarray(request.psi0, dtype=np.complex128),
    )
    rho_path = np.asarray(propagation["rho_path"])
    psi_path = np.asarray(propagation["psi_path"])
    pop = population_excited(rho_path)
    bundle = TrajectoryBundle(
        t_us=t_us,
        psi_path=psi_path,
        rho_path=rho_path,
        pop_excited=np.asarray(pop, dtype=np.float64),
    )
    return bundle


def compute_population_trace(summary: ResultSummary, request: TrajectoryRequest) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning time grid and population trace."""

    bundle = compute_trajectory_bundle(summary, request)
    return bundle.t_us, bundle.pop_excited


def compute_bloch_bundle(
    summary: ResultSummary,
    request: TrajectoryRequest,
    *,
    include_baseline: bool = True,
) -> Mapping[str, TrajectoryBundle]:
    """Propagate optimized (and optionally baseline) pulses, returning Bloch data."""

    optimized = compute_trajectory_bundle(summary, request)
    optimized = replace(
        optimized,
        bloch=bloch_coordinates(optimized.rho_path),
        bloch_ground=_compute_ground_path(summary.pulses.omega, summary.pulses.delta, request),
    )

    payload: dict[str, TrajectoryBundle] = {"optimized": optimized}

    if include_baseline and request.omega_base is not None:
        baseline_delta = request.delta_base
        omega_base = _as_array(request.omega_base)
        if baseline_delta is None:
            baseline_delta = np.zeros_like(omega_base, dtype=np.float64)
        baseline = propagate_piecewise_const(
            omega_base,
            _as_array(baseline_delta),
            float(request.dt_us)
            if request.dt_us is not None
            else (float(optimized.t_us[1] - optimized.t_us[0]) if optimized.t_us.size > 1 else 0.0),
            psi0=np.asarray(request.psi0, dtype=np.complex128),
        )
        rho_base = np.asarray(baseline["rho_path"])
        bundle = TrajectoryBundle(
            t_us=optimized.t_us,
            psi_path=np.asarray(baseline["psi_path"]),
            rho_path=rho_base,
            pop_excited=np.asarray(population_excited(rho_base), dtype=np.float64),
            bloch=bloch_coordinates(rho_base),
            bloch_ground=_compute_ground_path(request.omega_base, request.delta_base, request),
        )
        payload["baseline"] = bundle

    return payload


def _compute_ground_path(
    omega: np.ndarray | None,
    delta: np.ndarray | None,
    request: TrajectoryRequest,
) -> np.ndarray | None:
    if omega is None:
        return None
    omega_arr = _as_array(omega)
    if delta is None:
        if request.delta_base is None:
            return None
        delta_arr = _as_array(request.delta_base)
    else:
        delta_arr = _as_array(delta)
    projectors = ground_state_projectors(omega_arr, delta_arr)
    return bloch_coordinates(projectors)
