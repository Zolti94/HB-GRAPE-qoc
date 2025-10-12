"""Problem construction utilities for GRAPE coefficient optimizers.

This module isolates the data-flow from experiment configuration → baseline
arrays → `GrapeControlProblem`.  The intent is to keep the optimizer front-ends
focused on iteration logic while this module is responsible for wiring the
physics inputs (time grids, baselines, harmonic bases, states) and producing the
immutable problem description plus initial coefficients that optimizers mutate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ..baselines import GrapeBaselineConfig, build_grape_baseline
from ..config import ExperimentConfig, PenaltyConfig
from ..controls import coeffs_to_control, crab_linear_basis
from ..physics import rho_to_state

NDArrayFloat = npt.NDArray[np.float64]
NDArrayComplex = npt.NDArray[np.complex128]

__all__ = ["GrapeControlProblem", "build_grape_problem"]

_TWO_PI = 2.0 * np.pi


@dataclass(slots=True)
class GrapeControlProblem:
    """Problem definition capturing bases, baselines, and metadata for GRAPE.

    The dataclass binds together the static quantities required by all
    optimizers: time grid, baseline controls, harmonic bases, states, penalties,
    and bookkeeping metadata.  Optimizer front-ends treat ``coeffs_init`` as the
    starting point in coefficient space and call :meth:`controls_from_coeffs` /
    :meth:`gradients_to_coeffs` to shuttle between coefficient vectors and the
    time-domain control representation.
    """

    t_us: NDArrayFloat
    dt_us: float
    omega_base: NDArrayFloat
    delta_base: Optional[NDArrayFloat]
    basis_omega: NDArrayFloat
    basis_delta: Optional[NDArrayFloat]
    psi0: NDArrayComplex
    psi_target: NDArrayComplex
    penalties: PenaltyConfig
    optimize_delta: bool
    metadata: Dict[str, Any]
    coeffs_init: NDArrayFloat
    objective: str = "terminal"
    path_settings: Dict[str, Any] = field(default_factory=dict)
    ensemble_settings: Dict[str, Any] = field(default_factory=dict)
    omega_slice: slice = field(init=False)
    delta_slice: Optional[slice] = field(init=False)
    num_coeffs: int = field(init=False)
    t_total_us: float = field(init=False)

    def __post_init__(self) -> None:
        k_omega = self.basis_omega.shape[1]
        self.omega_slice = slice(0, k_omega)
        if self.optimize_delta and self.basis_delta is not None:
            k_delta = self.basis_delta.shape[1]
            self.delta_slice = slice(k_omega, k_omega + k_delta)
            self.num_coeffs = k_omega + k_delta
        else:
            self.delta_slice = None
            self.num_coeffs = k_omega
        if self.t_us.size > 1:
            self.t_total_us = float(self.t_us[-1] - self.t_us[0])
        else:
            self.t_total_us = float(self.dt_us)

    @property
    def k_omega(self) -> int:
        return self.basis_omega.shape[1]

    @property
    def k_delta(self) -> int:
        return 0 if self.delta_slice is None or self.basis_delta is None else self.basis_delta.shape[1]

    def split_coeffs(self, coeffs: NDArrayFloat) -> tuple[NDArrayFloat, Optional[NDArrayFloat]]:
        """Split concatenated coefficients into omega and optional delta slices."""

        coeffs = np.asarray(coeffs, dtype=np.float64)
        c_omega = coeffs[self.omega_slice]
        if self.delta_slice is not None and self.basis_delta is not None:
            c_delta = coeffs[self.delta_slice]
        else:
            c_delta = None
        return c_omega, c_delta

    def controls_from_coeffs(self, coeffs: NDArrayFloat) -> tuple[NDArrayFloat, Optional[NDArrayFloat]]:
        """Construct control waveforms from coefficients and baselines."""

        c_omega, c_delta = self.split_coeffs(coeffs)
        omega = coeffs_to_control(self.basis_omega, c_omega, base=self.omega_base)
        if c_delta is not None and self.basis_delta is not None and self.delta_base is not None:
            delta = coeffs_to_control(self.basis_delta, c_delta, base=self.delta_base)
        else:
            delta = self.delta_base
        return omega, None if delta is None else np.asarray(delta, dtype=np.float64)

    def gradients_to_coeffs(
        self,
        grad_omega: NDArrayFloat,
        grad_delta: Optional[NDArrayFloat],
    ) -> NDArrayFloat:
        """Project gradients from the time domain back into coefficient space."""

        pieces = [self.basis_omega.T @ np.asarray(grad_omega, dtype=np.float64)]
        if self.delta_slice is not None and self.basis_delta is not None and grad_delta is not None:
            pieces.append(self.basis_delta.T @ np.asarray(grad_delta, dtype=np.float64))
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


def _initial_coefficients(problem: GrapeControlProblem, options: Mapping[str, Any]) -> NDArrayFloat:
    """Derive initial coefficients while honouring user overrides."""

    coeffs = problem.coeffs_init
    if options.get("coeffs_init") is not None:
        coeffs = np.asarray(options["coeffs_init"], dtype=np.float64)
    elif options.get("coeffs_init_omega") is not None:
        coeffs = np.zeros(problem.num_coeffs, dtype=np.float64)
        c_omega = np.asarray(options["coeffs_init_omega"], dtype=np.float64)
        if c_omega.shape[0] != problem.k_omega:
            raise ValueError("coeffs_init_omega length mismatch.")
        coeffs[problem.omega_slice] = c_omega
        if problem.delta_slice is not None:
            init_delta = options.get("coeffs_init_delta")
            if init_delta is not None:
                c_delta = np.asarray(init_delta, dtype=np.float64)
                if c_delta.shape[0] != problem.k_delta:
                    raise ValueError("coeffs_init_delta length mismatch.")
                coeffs[problem.delta_slice] = c_delta
    return coeffs.astype(np.float64, copy=True)


def build_grape_problem(config: ExperimentConfig) -> Tuple[GrapeControlProblem, NDArrayFloat, Dict[str, Any]]:
    """Create a :class:`GrapeControlProblem` from an experiment configuration.

    Workflow
    --------
    1. Deserialize the baseline config from ``config.baseline`` and generate the
       deterministic arrays used across notebooks/optimizers.
    2. Validate and, when requested, truncate/override the harmonic bases.
    3. Infer optimizer metadata (objective choice, path/ensemble settings,
       penalty weights) and assemble the immutable :class:`GrapeControlProblem`.
    4. Produce the initial coefficient vector returned alongside the problem so
       optimizer front-ends can begin iterating without re-reading the baseline.

    Returns
    -------
    Tuple[GrapeControlProblem, NDArrayFloat, Dict[str, Any]]
        ``(problem, coeffs0, metadata)`` where ``metadata`` merges baseline
        summaries with optimizer selections for downstream reporting.
    """

    spec = config.baseline
    if spec.params is None:
        raise ValueError("ExperimentConfig.baseline.params must define a GRAPE baseline.")
    baseline_cfg = GrapeBaselineConfig.from_dict(spec.params)
    arrays, baseline_meta = build_grape_baseline(baseline_cfg)
    metadata = dict(baseline_meta)
    metadata.setdefault("baseline_name", spec.name)
    metadata.setdefault("baseline_config", baseline_cfg.to_dict())

    t_us = arrays["t"].astype(np.float64)
    dt_us = float(np.asarray(arrays.get("dt", np.diff(t_us)).squeeze(), dtype=np.float64))
    omega_base = arrays["Omega0"].astype(np.float64)
    delta_base = arrays.get("Delta0")
    if delta_base is not None:
        delta_base = delta_base.astype(np.float64)

    basis_omega = arrays["CRAB_BASIS_OMEGA"].astype(np.float64)
    basis_delta = arrays.get("CRAB_BASIS_DELTA")
    if basis_delta is not None:
        basis_delta = basis_delta.astype(np.float64)

    if basis_omega.shape[0] != t_us.shape[0]:
        raise ValueError("Omega basis rows must match time samples.")
    if basis_delta is not None and basis_delta.shape[0] != t_us.shape[0]:
        raise ValueError("Delta basis rows must match time samples.")

    psi0 = rho_to_state(arrays.get("rho0", np.array([1.0, 0.0], dtype=np.complex128)))
    psi_target = rho_to_state(arrays.get("target", np.array([0.0, 1.0], dtype=np.complex128)))

    options = dict(config.optimizer_options)
    metadata = dict(metadata)
    objective = str(config.metadata.get("objective", options.get("objective", "terminal"))).lower()
    if objective not in {"terminal", "path", "ensemble"}:
        objective = "terminal"

    delta_max_rad_per_us = float(np.abs(delta_base).max()) if delta_base is not None else 0.0
    delta_max_MHz = delta_max_rad_per_us / _TWO_PI

    path_defaults = {"reference": "adiabatic_ground_state"}
    path_settings = dict(config.metadata.get("path_params", options.get("path_params", {})))
    path_defaults.update({k: v for k, v in path_settings.items() if v is not None})

    ensemble_defaults = {
        "beta_min": 0.9,
        "beta_max": 1.1,
        "num_beta": 5,
        "detuning_MHz_min": -delta_max_MHz,
        "detuning_MHz_max": delta_max_MHz,
        "num_detuning": 5,
    }
    ensemble_settings = dict(config.metadata.get("ensemble_params", options.get("ensemble_params", {})))
    ensemble_defaults.update({k: v for k, v in ensemble_settings.items() if v is not None})

    optimize_delta = bool(options.get("optimize_delta", basis_delta is not None))

    if "omegas_rad_per_us" in options:
        omegas = np.asarray(options["omegas_rad_per_us"], dtype=np.float64)
        if omegas.size == 0:
            raise ValueError("omegas_rad_per_us must be non-empty when provided.")
        phases = np.asarray(options.get("phases", np.zeros_like(omegas)), dtype=np.float64)
        if phases.shape != omegas.shape:
            raise ValueError("phases length must match omegas_rad_per_us length.")
        basis_omega = crab_linear_basis(t_us, omegas.size, omegas, phases)
    else:
        K = int(options.get("K", basis_omega.shape[1]))
        if K <= 0:
            raise ValueError("K must be positive.")
        if K < basis_omega.shape[1]:
            basis_omega = basis_omega[:, :K]

    if optimize_delta and basis_delta is not None:
        if "delta_omegas_rad_per_us" in options:
            delta_omegas = np.asarray(options["delta_omegas_rad_per_us"], dtype=np.float64)
            if delta_omegas.size == 0:
                raise ValueError("delta_omegas_rad_per_us must be non-empty when provided.")
            delta_phases = np.asarray(options.get("delta_phases", np.zeros_like(delta_omegas)), dtype=np.float64)
            if delta_phases.shape != delta_omegas.shape:
                raise ValueError("delta_phases length must match delta_omegas_rad_per_us length.")
            basis_delta = crab_linear_basis(t_us, delta_omegas.size, delta_omegas, delta_phases)
        else:
            K_delta = int(options.get("K_delta", basis_delta.shape[1]))
            if K_delta <= 0:
                raise ValueError("K_delta must be positive.")
            if K_delta < basis_delta.shape[1]:
                basis_delta = basis_delta[:, :K_delta]
    else:
        basis_delta = None
        optimize_delta = False

    penalties = config.penalties

    coeffs_init = np.zeros(basis_omega.shape[1] + (basis_delta.shape[1] if basis_delta is not None else 0), dtype=np.float64)

    problem = GrapeControlProblem(
        t_us=t_us,
        dt_us=dt_us,
        omega_base=omega_base,
        delta_base=delta_base,
        basis_omega=basis_omega,
        basis_delta=basis_delta if optimize_delta else None,
        psi0=psi0,
        psi_target=psi_target,
        penalties=penalties,
        optimize_delta=optimize_delta,
        metadata=metadata,
        coeffs_init=coeffs_init,
        objective=objective,
        path_settings=path_defaults,
        ensemble_settings=ensemble_defaults,
    )

    problem.metadata.update(
        {
            "objective": objective,
            "omega_basis_shape": problem.basis_omega.shape,
            "delta_basis_shape": None if problem.basis_delta is None else problem.basis_delta.shape,
            "omegas_rad_per_us": options.get("omegas_rad_per_us"),
            "delta_omegas_rad_per_us": options.get("delta_omegas_rad_per_us"),
            "path_settings": path_defaults,
            "ensemble_settings": ensemble_defaults,
        }
    )

    coeffs0 = _initial_coefficients(problem, config.optimizer_options)
    if coeffs0.shape[0] != problem.num_coeffs:
        raise ValueError("Initial coefficient vector size mismatch with basis dimensions.")

    problem.coeffs_init = coeffs0.copy()
    return problem, coeffs0, metadata
