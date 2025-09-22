"""Shared optimizer utilities for CRAB coefficient optimizers (us / rad-per-us)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import numpy.typing as npt

from ..config import ExperimentConfig, PenaltyConfig
from ..controls import coeffs_to_control, crab_linear_basis
from ..cost import accumulate_cost_and_grads, penalty_terms, terminal_cost_and_grad
from ..physics import propagate_piecewise_const, fidelity_pure
from ..crab_notebook_utils import ground_state_projectors

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

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
SIGMA_X_HALF = 0.5 * SIGMA_X
SIGMA_Z_HALF = 0.5 * SIGMA_Z
_TWO_PI = 2.0 * np.pi


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
        return omega, None if delta is None else np.asarray(delta, dtype=np.float64)

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
    extras: Dict[str, Any] | None = None


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
        objective=objective,
        path_settings=path_defaults,
        ensemble_settings=ensemble_defaults,
    )

    problem.metadata.update({
        "objective": objective,
        "omega_basis_shape": problem.basis_omega.shape,
        "delta_basis_shape": None if problem.basis_delta is None else problem.basis_delta.shape,
        "omegas_rad_per_us": options.get("omegas_rad_per_us"),
        "delta_omegas_rad_per_us": options.get("delta_omegas_rad_per_us"),
        "path_settings": path_defaults,
        "ensemble_settings": ensemble_defaults,
    })

    coeffs0 = _initial_coefficients(problem, config.optimizer_options)
    if coeffs0.shape[0] != problem.num_coeffs:
        raise ValueError("Initial coefficient vector size mismatch with basis dimensions.")

    problem.coeffs_init = coeffs0.copy()
    return problem, coeffs0, metadata


def _evaluate_terminal(
    problem: CrabProblem,
    omega: NDArrayFloat,
    delta: Optional[NDArrayFloat],
    neg_epsilon: float,
) -> tuple[Dict[str, float], NDArrayFloat, Dict[str, Any]]:
    delta_eval = delta if delta is not None else (
        np.asarray(problem.delta_base, dtype=np.float64) if problem.delta_base is not None else np.zeros_like(omega)
    )
    raw_cost, grad_dict = accumulate_cost_and_grads(
        omega,
        delta_eval,
        problem.dt_us,
        psi0=problem.psi0,
        psi_target=problem.psi_target,
        w_power=problem.penalties.power_weight,
        w_neg=problem.penalties.neg_weight,
        neg_epsilon=neg_epsilon,
    )
    cost = {
        "terminal": float(raw_cost.get("terminal", 0.0)),
        "path": 0.0,
        "ensemble": 0.0,
        "power_penalty": float(raw_cost.get("power_penalty", 0.0)),
        "neg_penalty": float(raw_cost.get("neg_penalty", 0.0)),
        "total": float(raw_cost.get("total", 0.0)),
        "terminal_eval": float(raw_cost.get("terminal", 0.0)),
    }
    grad_coeffs = problem.gradients_to_coeffs(
        grad_dict.get("dJ/dOmega", np.zeros_like(omega)),
        grad_dict.get("dJ/dDelta") if problem.delta_slice is not None else None,
    )
    extras = {
        "omega": omega,
        "delta": delta,
        "grad_time": grad_dict,
        "oracle_calls": 1,
        "terminal_infidelity": cost["terminal"],
    }
    return cost, grad_coeffs, extras


def _evaluate_path(
    problem: CrabProblem,
    omega: NDArrayFloat,
    delta: Optional[NDArrayFloat],
) -> tuple[Dict[str, float], NDArrayFloat, Dict[str, Any]]:
    delta_eval = delta if delta is not None else (
        np.asarray(problem.delta_base, dtype=np.float64) if problem.delta_base is not None else np.zeros_like(omega)
    )
    prop = propagate_piecewise_const(omega, delta_eval, problem.dt_us, psi0=problem.psi0)
    rho_path = np.asarray(prop["rho_path"], dtype=np.complex128)
    U_hist = np.asarray(prop["U_hist"], dtype=np.complex128)
    rhos = rho_path[:-1]
    projectors = ground_state_projectors(omega, delta_eval)
    overlaps = np.real(np.einsum("kij,kji->k", projectors, rhos))
    total_time = problem.t_total_us if problem.t_total_us > 0.0 else problem.dt_us * max(len(omega), 1)
    path_fidelity = float(np.clip((problem.dt_us / total_time) * overlaps.sum(), 0.0, 1.0))
    path_infidelity = 1.0 - path_fidelity

    final_state = rho_path[-1]
    final_fidelity = float(np.clip(np.real(np.trace(final_state @ problem.psi_target @ problem.psi_target.conj().T)), 0.0, 1.0))
    final_infidelity = 1.0 - final_fidelity

    pen_power, pen_neg, grad_penalty_time = penalty_terms(
        omega,
        problem.dt_us,
        power_weight=problem.penalties.power_weight,
        neg_weight=problem.penalties.neg_weight,
        neg_kappa=problem.penalties.neg_kappa,
    )

    Nt = omega.size
    lams = np.zeros((Nt, 2, 2), dtype=np.complex128)
    for k in range(Nt - 2, -1, -1):
        lam_next = lams[k + 1]
        U = U_hist[k]
        lams[k] = U.conj().T @ (lam_next + (problem.dt_us / total_time) * projectors[k]) @ U

    gO_time = np.zeros_like(omega, dtype=np.float64)
    gD_time = np.zeros_like(delta_eval, dtype=np.float64)
    for k in range(Nt - 1):
        rho_k = rhos[k]
        lam_next = lams[k + 1]
        gO_time[k] = -np.imag(np.trace(lam_next @ (SIGMA_X_HALF @ rho_k - rho_k @ SIGMA_X_HALF))) * problem.dt_us
        gD_time[k] = -np.imag(np.trace(lam_next @ (SIGMA_Z_HALF @ rho_k - rho_k @ SIGMA_Z_HALF))) * problem.dt_us

    gO_time_total = gO_time + grad_penalty_time
    gD_time_total = gD_time if problem.delta_slice is not None else None
    grad_coeffs = problem.gradients_to_coeffs(gO_time_total, gD_time_total)

    cost = {
        "terminal": float(final_infidelity),
        "path": float(path_infidelity),
        "ensemble": 0.0,
        "power_penalty": float(pen_power),
        "neg_penalty": float(pen_neg),
        "total": float(path_infidelity + pen_power + pen_neg),
        "terminal_eval": float(final_infidelity),
    }

    grad_time = {
        "dJ/dOmega": gO_time_total,
        "dJ/dDelta": np.zeros_like(delta_eval) if gD_time_total is None else gD_time_total,
    }

    extras = {
        "omega": omega,
        "delta": delta,
        "grad_time": grad_time,
        "oracle_calls": 1,
        "path_infidelity": float(path_infidelity),
        "path_fidelity": float(path_fidelity),
        "terminal_infidelity": float(final_infidelity),
    }
    return cost, grad_coeffs, extras


def _evaluate_ensemble(
    problem: CrabProblem,
    omega: NDArrayFloat,
    delta: Optional[NDArrayFloat],
    neg_epsilon: float,
) -> tuple[Dict[str, float], NDArrayFloat, Dict[str, Any]]:
    delta_eval = delta if delta is not None else (
        np.asarray(problem.delta_base, dtype=np.float64) if problem.delta_base is not None else np.zeros_like(omega)
    )
    settings = problem.ensemble_settings
    beta_vals = np.linspace(float(settings["beta_min"]), float(settings["beta_max"]), int(settings["num_beta"]))
    detuning_vals = np.linspace(
        float(settings["detuning_MHz_min"]),
        float(settings["detuning_MHz_max"]),
        int(settings["num_detuning"]),
    )
    detuning_offsets = detuning_vals * _TWO_PI
    ensemble_size = beta_vals.size * detuning_vals.size

    pen_power, pen_neg, grad_penalty_time = penalty_terms(
        omega,
        problem.dt_us,
        power_weight=problem.penalties.power_weight,
        neg_weight=problem.penalties.neg_weight,
        neg_kappa=problem.penalties.neg_kappa,
    )

    base_infidelity, _, _ = terminal_cost_and_grad(
        omega,
        delta_eval,
        problem.psi0,
        problem.dt_us,
        problem.psi_target,
        power_weight=0.0,
        neg_weight=0.0,
    )
    base_infidelity = float(base_infidelity)

    gO_acc = np.zeros_like(omega, dtype=np.float64)
    gD_acc = np.zeros_like(delta_eval, dtype=np.float64)
    inf_sum = 0.0
    fid_sum = 0.0
    fid_sq_sum = 0.0

    for beta in beta_vals:
        omega_mod = beta * omega
        for detuning in detuning_offsets:
            delta_mod = delta_eval + detuning
            sample_infidelity, gO_time, gD_time = terminal_cost_and_grad(
                omega_mod,
                delta_mod,
                problem.psi0,
                problem.dt_us,
                problem.psi_target,
                power_weight=0.0,
                neg_weight=0.0,
            )
            inf_sum += float(sample_infidelity)
            sample_fidelity = float(np.clip(1.0 - sample_infidelity, 0.0, 1.0))
            fid_sum += sample_fidelity
            fid_sq_sum += sample_fidelity * sample_fidelity
            gO_acc += beta * np.asarray(gO_time, dtype=np.float64)
            if problem.delta_slice is not None:
                gD_acc += np.asarray(gD_time, dtype=np.float64)

    inv_members = 1.0 / float(ensemble_size)
    mean_final_infidelity = float(np.clip(inf_sum * inv_members, 0.0, 1.0))
    mean_final_fidelity = float(np.clip(fid_sum * inv_members, 0.0, 1.0))
    variance = max(fid_sq_sum * inv_members - mean_final_fidelity * mean_final_fidelity, 0.0)
    std_final_fidelity = math.sqrt(variance)

    gO_time_mean = gO_acc * inv_members + grad_penalty_time
    if problem.delta_slice is not None:
        gD_time_mean = gD_acc * inv_members
    else:
        gD_time_mean = None

    grad_coeffs = problem.gradients_to_coeffs(gO_time_mean, gD_time_mean)

    cost = {
        "terminal": float(base_infidelity),
        "path": 0.0,
        "ensemble": float(mean_final_infidelity),
        "power_penalty": float(pen_power),
        "neg_penalty": float(pen_neg),
        "total": float(mean_final_infidelity + pen_power + pen_neg),
        "terminal_eval": float(base_infidelity),
    }

    grad_time = {
        "dJ/dOmega": gO_time_mean,
        "dJ/dDelta": np.zeros_like(delta_eval) if gD_time_mean is None else gD_time_mean,
    }

    extras = {
        "omega": omega,
        "delta": delta,
        "grad_time": grad_time,
        "oracle_calls": int(ensemble_size),
        "mean_final_infidelity": float(mean_final_infidelity),
        "mean_final_fidelity": float(mean_final_fidelity),
        "std_final_fidelity": float(std_final_fidelity),
        "terminal_infidelity": float(base_infidelity),
    }
    return cost, grad_coeffs, extras


def evaluate_problem(
    problem: CrabProblem,
    coeffs: NDArrayFloat,
    *,
    neg_epsilon: float = 1e-6,
) -> tuple[Dict[str, float], NDArrayFloat, Dict[str, Any]]:
    omega, delta = problem.controls_from_coeffs(coeffs)
    objective = problem.objective
    if objective == "path":
        cost_dict, grad_coeffs, extras = _evaluate_path(problem, omega, delta)
    elif objective == "ensemble":
        cost_dict, grad_coeffs, extras = _evaluate_ensemble(problem, omega, delta, neg_epsilon)
    else:
        cost_dict, grad_coeffs, extras = _evaluate_terminal(problem, omega, delta, neg_epsilon)
    extras.setdefault("objective", objective)
    extras.setdefault("omega", omega)
    extras.setdefault("delta", delta)
    return cost_dict, grad_coeffs, extras


def history_to_arrays(history: Dict[str, list[Any]]) -> Dict[str, NDArrayFloat]:
    arrays: Dict[str, NDArrayFloat] = {}
    for key, values in history.items():
        if key == "iter" or key == "calls_per_iter":
            arrays[key] = np.asarray(values, dtype=np.int32)
        else:
            arrays[key] = np.asarray(values, dtype=np.float64)
    return arrays
