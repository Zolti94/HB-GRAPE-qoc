"""Shared optimizer utilities for CRAB coefficient optimizers (us / rad-per-us)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import numpy.typing as npt

from ..config import ExperimentConfig, PenaltyConfig
from ..controls import coeffs_to_control
from ..cost import accumulate_cost_and_grads

NDArrayFloat = npt.NDArray[np.float64]
NDArrayComplex = npt.NDArray[np.complex128]

__all__ = [
    "NDArrayFloat",
    "StepStats",
    "OptimizerState",
    "CrabProblem",
    "OptimizationOutput",
    "clip_gradients",
    "safe_norm",
    "load_crab_problem",
    "evaluate_problem",
    "history_to_arrays",
]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BASELINES: Dict[str, Path] = {
    "default": _PROJECT_ROOT / "data" / "baselines" / "_baseline_crab",
    "crab": _PROJECT_ROOT / "data" / "baselines" / "_baseline_crab",
}


def _rho_to_state(rho: NDArrayComplex) -> NDArrayComplex:
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape == (2,):
        vec = rho
    elif rho.shape == (2, 2):
        vals, vecs = np.linalg.eigh(rho)
        idx = int(np.argmax(vals))
        vec = vecs[:, idx]
    else:
        raise ValueError("rho must be a state vector (2,) or density matrix (2,2).")
    # Fix global phase for reproducibility.
    phase = np.exp(-1j * np.angle(vec[0])) if abs(vec[0]) > 1e-12 else 1.0
    return (vec * phase).astype(np.complex128)


@dataclass(slots=True)
class StepStats:
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


@dataclass(slots=True)
class OptimizerState:
    coeffs: NDArrayFloat
    grad: NDArrayFloat
    history: Dict[str, list[Any]] = field(default_factory=_init_history)
    runtime_s: float = 0.0
    status: str = "in_progress"

    def record(self, stats: StepStats) -> None:
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
class CrabProblem:
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
    omega_slice: slice = field(init=False)
    delta_slice: Optional[slice] = field(init=False)
    num_coeffs: int = field(init=False)

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

    @property
    def k_omega(self) -> int:
        return self.basis_omega.shape[1]

    @property
    def k_delta(self) -> int:
        return 0 if self.delta_slice is None else self.basis_delta.shape[1]

    def split_coeffs(self, coeffs: NDArrayFloat) -> tuple[NDArrayFloat, Optional[NDArrayFloat]]:
        coeffs = np.asarray(coeffs, dtype=np.float64)
        c_omega = coeffs[self.omega_slice]
        if self.delta_slice is not None and self.basis_delta is not None:
            c_delta = coeffs[self.delta_slice]
        else:
            c_delta = None
        return c_omega, c_delta

    def controls_from_coeffs(self, coeffs: NDArrayFloat) -> tuple[NDArrayFloat, Optional[NDArrayFloat]]:
        c_omega, c_delta = self.split_coeffs(coeffs)
        omega = coeffs_to_control(self.basis_omega, c_omega, base=self.omega_base)
        if c_delta is not None and self.basis_delta is not None and self.delta_base is not None:
            delta = coeffs_to_control(self.basis_delta, c_delta, base=self.delta_base)
        else:
            delta = self.delta_base
        return omega, delta

    def gradients_to_coeffs(
        self,
        grad_omega: NDArrayFloat,
        grad_delta: Optional[NDArrayFloat],
    ) -> NDArrayFloat:
        pieces = [self.basis_omega.T @ np.asarray(grad_omega, dtype=np.float64)]
        if self.delta_slice is not None and self.basis_delta is not None and grad_delta is not None:
            pieces.append(self.basis_delta.T @ np.asarray(grad_delta, dtype=np.float64))
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


@dataclass(slots=True)
class OptimizationOutput:
    coeffs: NDArrayFloat
    omega: NDArrayFloat
    delta: Optional[NDArrayFloat]
    cost_terms: Dict[str, float]
    history: Dict[str, NDArrayFloat]
    runtime_s: float
    optimizer_state: Dict[str, Any]


def clip_gradients(grad: NDArrayFloat, *, max_norm: float | None = None) -> NDArrayFloat:
    vec = np.asarray(grad, dtype=np.float64)
    if max_norm is None or max_norm <= 0.0:
        return vec
    norm = np.linalg.norm(vec)
    if norm == 0.0 or norm <= max_norm:
        return vec
    return vec * (max_norm / norm)


def safe_norm(arr: NDArrayFloat) -> float:
    vec = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(vec).all():
        return float("inf")
    return float(np.linalg.norm(vec))


def _resolve_baseline_dir(config: ExperimentConfig) -> Path:
    spec = config.baseline
    if spec.path is not None:
        return Path(spec.path)
    name = (spec.name or "default").lower()
    if name in _DEFAULT_BASELINES:
        return _DEFAULT_BASELINES[name]
    candidate = _PROJECT_ROOT / "data" / "baselines" / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Baseline '{spec.name}' not found. Provide baseline.path in config.")


def _load_arrays(base_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    arrays_path = base_dir / "arrays.npz"
    metadata_path = base_dir / "metadata.json"
    with np.load(arrays_path, allow_pickle=True) as npz:
        arrays = {k: np.array(npz[k]) for k in npz.files}
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["baseline_dir"] = str(base_dir)
    return arrays, metadata


def _initial_coefficients(problem: CrabProblem, options: Mapping[str, Any]) -> NDArrayFloat:
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


def load_crab_problem(config: ExperimentConfig) -> tuple[CrabProblem, NDArrayFloat, Dict[str, Any]]:
    base_dir = _resolve_baseline_dir(config)
    arrays, metadata = _load_arrays(base_dir)

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

    psi0 = _rho_to_state(arrays.get("rho0", np.array([1.0, 0.0], dtype=np.complex128)))
    psi_target = _rho_to_state(arrays.get("target", np.array([0.0, 1.0], dtype=np.complex128)))

    optimize_delta = bool(config.optimizer_options.get("optimize_delta", basis_delta is not None))

    penalties = config.penalties

    coeffs_init = np.zeros(basis_omega.shape[1] + (basis_delta.shape[1] if optimize_delta and basis_delta is not None else 0), dtype=np.float64)

    problem = CrabProblem(
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
    )

    coeffs0 = _initial_coefficients(problem, config.optimizer_options)
    if coeffs0.shape[0] != problem.num_coeffs:
        raise ValueError("Initial coefficient vector size mismatch with basis dimensions.")

    problem.coeffs_init = coeffs0.copy()
    return problem, coeffs0, metadata


def evaluate_problem(
    problem: CrabProblem,
    coeffs: NDArrayFloat,
    *,
    neg_epsilon: float = 1e-6,
) -> tuple[Dict[str, float], NDArrayFloat, Dict[str, Any]]:
    omega, delta = problem.controls_from_coeffs(coeffs)
    raw_cost, grad_dict = accumulate_cost_and_grads(
        omega,
        delta if delta is not None else np.asarray(problem.delta_base, dtype=np.float64) if problem.delta_base is not None else np.zeros_like(omega),
        problem.dt_us,
        psi0=problem.psi0,
        psi_target=problem.psi_target,
        w_power=problem.penalties.power_weight,
        w_neg=problem.penalties.neg_weight,
        neg_epsilon=neg_epsilon,
    )
    cost_dict = {k: float(v) for k, v in raw_cost.items()}
    grad_coeffs = problem.gradients_to_coeffs(
        grad_dict.get("dJ/dOmega", np.zeros_like(omega)),
        grad_dict.get("dJ/dDelta") if problem.delta_slice is not None else None,
    )
    extras = {
        "omega": omega,
        "delta": delta,
        "grad_time": grad_dict,
    }
    return cost_dict, grad_coeffs, extras


def history_to_arrays(history: Dict[str, list[Any]]) -> Dict[str, NDArrayFloat]:
    arrays: Dict[str, NDArrayFloat] = {}
    for key, values in history.items():
        if key == "iter" or key == "calls_per_iter":
            arrays[key] = np.asarray(values, dtype=np.int32)
        else:
            arrays[key] = np.asarray(values, dtype=np.float64)
    return arrays
