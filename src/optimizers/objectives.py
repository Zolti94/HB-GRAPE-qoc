"""Objective evaluation helpers for GRAPE coefficient optimizers.

Each objective maps the coefficient vector → time-domain controls → physics
simulation → scalar cost/gradient.  Centralising this logic keeps the optimizer
loops lightweight and makes it easier to reason about the data flow between
controls, penalties, and diagnostic extras returned to notebooks.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ..cost import accumulate_cost_and_grads
from ..penalties import compute_penalties
from ..physics import propagate_piecewise_const, fidelity_pure, SIGMA_X, SIGMA_Z
from ..crab_notebook_utils import ground_state_projectors
from .problem import GrapeControlProblem

NDArrayFloat = npt.NDArray[np.float64]

__all__ = ["evaluate_problem"]

SIGMA_X_HALF = 0.5 * SIGMA_X
SIGMA_Z_HALF = 0.5 * SIGMA_Z
_TWO_PI = 2.0 * np.pi


def _evaluate_terminal(
    problem: GrapeControlProblem,
    omega: NDArrayFloat,
    delta: Optional[NDArrayFloat],
    neg_kappa: float,
) -> Tuple[Dict[str, float], NDArrayFloat, Dict[str, Any]]:
    """Evaluate terminal objective value and gradients for given controls."""

    delta_eval = (
        delta
        if delta is not None
        else (
            np.asarray(problem.delta_base, dtype=np.float64)
            if problem.delta_base is not None
            else np.zeros_like(omega)
        )
    )

    cost_dict, grad_dict = accumulate_cost_and_grads(
        omega,
        delta_eval,
        problem.dt_us,
        psi0=problem.psi0,
        psi_target=problem.psi_target,
        w_power=problem.penalties.power_weight,
        w_neg=problem.penalties.neg_weight,
        neg_kappa=neg_kappa,
    )

    gO_time = np.asarray(grad_dict.get("dJ/dOmega", np.zeros_like(omega)), dtype=np.float64)
    gD_time_full = np.asarray(grad_dict.get("dJ/dDelta", np.zeros_like(delta_eval)), dtype=np.float64)
    gD_for_coeffs = gD_time_full if problem.delta_slice is not None else None

    grad_coeffs = problem.gradients_to_coeffs(gO_time, gD_for_coeffs)

    grad_time = {
        "dJ/dOmega": gO_time,
        "dJ/dDelta": gD_time_full if gD_for_coeffs is not None else np.zeros_like(delta_eval),
    }

    cost = {
        "terminal": float(cost_dict.get("terminal", 0.0)),
        "path": 0.0,
        "ensemble": 0.0,
        "power_penalty": float(cost_dict.get("power_penalty", 0.0)),
        "neg_penalty": float(cost_dict.get("neg_penalty", 0.0)),
        "total": float(cost_dict.get("total", cost_dict.get("terminal", 0.0))),
        "terminal_eval": float(cost_dict.get("terminal", 0.0)),
    }

    extras = {
        "omega": omega,
        "delta": delta,
        "grad_time": grad_time,
        "oracle_calls": 1,
        "terminal_infidelity": cost["terminal"],
    }
    return cost, grad_coeffs, extras


def _evaluate_path(
    problem: GrapeControlProblem,
    omega: NDArrayFloat,
    delta: Optional[NDArrayFloat],
    neg_kappa: float,
) -> Tuple[Dict[str, float], NDArrayFloat, Dict[str, Any]]:
    """Evaluate path objective using instantaneous ground-state projectors."""

    delta_eval = (
        delta
        if delta is not None
        else (
            np.asarray(problem.delta_base, dtype=np.float64)
            if problem.delta_base is not None
            else np.zeros_like(omega)
        )
    )
    prop = propagate_piecewise_const(omega, delta_eval, problem.dt_us, psi0=problem.psi0)
    rho_path = np.asarray(prop["rho_path"], dtype=np.complex128)
    U_hist = np.asarray(prop["U_hist"], dtype=np.complex128)
    rhos = rho_path[:-1]
    projectors = ground_state_projectors(omega, delta_eval)
    total_time = problem.t_total_us if problem.t_total_us > 0.0 else problem.dt_us * max(len(omega), 1)
    dt_over_total = problem.dt_us / total_time if total_time > 0.0 else 0.0
    grad_proj_omega = np.zeros_like(omega, dtype=np.float64)
    grad_proj_delta = np.zeros_like(delta_eval, dtype=np.float64)
    for idx, (om, de) in enumerate(zip(omega, delta_eval)):
        norm = float(np.hypot(om, de))
        if norm < 1e-12:
            continue
        norm3 = norm * norm * norm
        dnx_domega = (de * de) / norm3
        dnz_domega = -om * de / norm3
        dnx_ddelta = -om * de / norm3
        dnz_ddelta = (om * om) / norm3
        dP_domega = -0.5 * (dnx_domega * SIGMA_X + dnz_domega * SIGMA_Z)
        dP_ddelta = -0.5 * (dnx_ddelta * SIGMA_X + dnz_ddelta * SIGMA_Z)
        rho_k = rhos[idx]
        grad_proj_omega[idx] = -dt_over_total * float(np.real(np.trace(dP_domega @ rho_k)))
        grad_proj_delta[idx] = -dt_over_total * float(np.real(np.trace(dP_ddelta @ rho_k)))
    overlaps = np.real(np.einsum("kij,kji->k", projectors, rhos))
    path_fidelity = float(np.clip((problem.dt_us / total_time) * overlaps.sum(), 0.0, 1.0))
    path_infidelity = 1.0 - path_fidelity
    psi_T = rho_path[-1]
    if isinstance(psi_T, np.ndarray) and psi_T.ndim == 1:
        overlap = np.vdot(problem.psi_target, psi_T)
        final_fidelity = float(np.clip(np.real(overlap * overlap.conjugate()), 0.0, 1.0))
    else:
        proj = np.outer(problem.psi_target, np.conjugate(problem.psi_target))
        final_fidelity = float(np.clip(np.real(np.trace(psi_T @ proj)), 0.0, 1.0))
    final_infidelity = 1.0 - final_fidelity

    pen_power, pen_neg, grad_pen_omega, grad_pen_delta = compute_penalties(
        omega,
        delta_eval,
        problem.dt_us,
        power_weight=problem.penalties.power_weight,
        neg_weight=problem.penalties.neg_weight,
        neg_kappa=neg_kappa,
    )

    Nt = omega.size
    lams = np.zeros((Nt + 1, 2, 2), dtype=np.complex128)
    lams[-1] = (problem.dt_us / total_time) * projectors[-1]
    for k in range(Nt - 1, -1, -1):
        U = U_hist[k]
        lams[k] = U.conj().T @ (lams[k + 1]) @ U + (problem.dt_us / total_time) * projectors[k]
    gO_time = np.zeros_like(omega, dtype=np.float64)
    gD_time = np.zeros_like(delta_eval, dtype=np.float64)
    for k in range(Nt):
        rho_k = rhos[k]
        lam_next = lams[k]
        gO_time[k] = -np.imag(np.trace(lam_next @ (SIGMA_X_HALF @ rho_k - rho_k @ SIGMA_X_HALF))) * problem.dt_us
        gD_time[k] = -np.imag(np.trace(lam_next @ (SIGMA_Z_HALF @ rho_k - rho_k @ SIGMA_Z_HALF))) * problem.dt_us

    gO_time += grad_proj_omega
    gD_time += grad_proj_delta

    gO_time_total = gO_time + grad_pen_omega
    if problem.delta_slice is not None:
        gD_time_total = gD_time + grad_pen_delta
    else:
        gD_time_total = None

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
    problem: GrapeControlProblem,
    omega: NDArrayFloat,
    delta: Optional[NDArrayFloat],
    neg_kappa: float,
) -> Tuple[Dict[str, float], NDArrayFloat, Dict[str, Any]]:
    """Evaluate ensemble objective by marginalising over amplitude/detuning samples."""

    delta_eval = (
        delta
        if delta is not None
        else (
            np.asarray(problem.delta_base, dtype=np.float64)
            if problem.delta_base is not None
            else np.zeros_like(omega)
        )
    )
    settings = problem.ensemble_settings
    beta_vals = np.linspace(float(settings["beta_min"]), float(settings["beta_max"]), int(settings["num_beta"]))
    beta_weights = np.exp(-0.5 * ((beta_vals - 1.0) / 0.1) ** 2)
    beta_weights = beta_weights / beta_weights.sum()
    detuning_vals = np.linspace(
        float(settings["detuning_MHz_min"]),
        float(settings["detuning_MHz_max"]),
        int(settings["num_detuning"]),
    )
    detuning_offsets = detuning_vals * _TWO_PI
    delta_ref = float(delta_eval[0]) if delta_eval.size else 0.0
    sigma_detuning = 0.1 * abs(delta_ref)
    if sigma_detuning <= 0.0:
        sigma_detuning = 0.1
    detuning_weights = np.exp(-0.5 * (detuning_offsets / sigma_detuning) ** 2)
    detuning_weights = detuning_weights / detuning_weights.sum()
    ensemble_size = beta_vals.size * detuning_vals.size

    pen_power, pen_neg, grad_pen_omega, grad_pen_delta = compute_penalties(
        omega,
        delta_eval,
        problem.dt_us,
        power_weight=problem.penalties.power_weight,
        neg_weight=problem.penalties.neg_weight,
        neg_kappa=neg_kappa,
    )

    base_cost, _ = accumulate_cost_and_grads(
        omega,
        delta_eval,
        problem.dt_us,
        psi0=problem.psi0,
        psi_target=problem.psi_target,
        w_power=0.0,
        w_neg=0.0,
        neg_kappa=neg_kappa,
    )
    base_infidelity = float(base_cost.get("terminal", 0.0))

    gO_acc = np.zeros_like(omega, dtype=np.float64)
    gD_acc = np.zeros_like(delta_eval, dtype=np.float64)
    inf_sum = 0.0
    fid_sum = 0.0
    fid_sq_sum = 0.0

    for beta, beta_weight in zip(beta_vals, beta_weights):
        omega_mod = beta * omega
        for detuning, det_weight in zip(detuning_offsets, detuning_weights):
            weight = float(beta_weight * det_weight)
            delta_mod = delta_eval + detuning
            sample_cost, sample_grad = accumulate_cost_and_grads(
                omega_mod,
                delta_mod,
                problem.dt_us,
                psi0=problem.psi0,
                psi_target=problem.psi_target,
                w_power=0.0,
                w_neg=0.0,
                neg_kappa=neg_kappa,
            )
            sample_infidelity = float(sample_cost.get("terminal", 0.0))
            gO_time = np.asarray(sample_grad.get("dJ/dOmega", np.zeros_like(omega_mod)), dtype=np.float64)
            gD_time = np.asarray(sample_grad.get("dJ/dDelta", np.zeros_like(delta_mod)), dtype=np.float64)
            inf_sum += weight * sample_infidelity
            sample_fidelity = float(np.clip(1.0 - sample_infidelity, 0.0, 1.0))
            fid_sum += weight * sample_fidelity
            fid_sq_sum += weight * sample_fidelity * sample_fidelity
            gO_acc += weight * beta * gO_time
            if problem.delta_slice is not None:
                gD_acc += weight * gD_time

    mean_final_infidelity = float(np.clip(inf_sum, 0.0, 1.0))
    mean_final_fidelity = float(np.clip(fid_sum, 0.0, 1.0))
    variance = max(fid_sq_sum - mean_final_fidelity * mean_final_fidelity, 0.0)
    std_final_fidelity = math.sqrt(variance)

    gO_time_mean = gO_acc + grad_pen_omega
    if problem.delta_slice is not None:
        gD_time_mean = gD_acc + grad_pen_delta
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
    problem: GrapeControlProblem,
    coeffs: NDArrayFloat,
    *,
    neg_kappa: float = 10.0,
) -> Tuple[Dict[str, float], NDArrayFloat, Dict[str, Any]]:
    """Evaluate the problem objective for ``coeffs`` and return cost, gradient, extras.

    The function performs a full pipeline pass:

    ``coeffs`` → ``(omega, delta)`` via :meth:`GrapeControlProblem.controls_from_coeffs`
    → objective-specific physics/penalty evaluation → ``(cost, grad_coeffs, extras)``.

    Parameters
    ----------
    problem : GrapeControlProblem
        Immutable problem description produced by :func:`build_grape_problem`.
    coeffs : numpy.ndarray
        Current coefficient vector supplied by the optimizer.
    neg_kappa : float, optional
        Smoothed negativity parameter used by the penalty helper.

    Returns
    -------
    tuple
        ``(cost_dict, grad_coeffs, extras)`` where ``cost_dict`` aggregates cost
        components, ``grad_coeffs`` is the gradient in coefficient space, and
        ``extras`` includes diagnostics (controls, oracle counts, fidelities).
    """

    omega, delta = problem.controls_from_coeffs(coeffs)
    objective = problem.objective
    if objective == "path":
        cost_dict, grad_coeffs, extras = _evaluate_path(problem, omega, delta, neg_kappa)
    elif objective == "ensemble":
        cost_dict, grad_coeffs, extras = _evaluate_ensemble(problem, omega, delta, neg_kappa)
    else:
        cost_dict, grad_coeffs, extras = _evaluate_terminal(problem, omega, delta, neg_kappa)
    extras.setdefault("objective", objective)
    extras.setdefault("omega", omega)
    extras.setdefault("delta", delta)
    return cost_dict, grad_coeffs, extras
