"""Armijo backtracking line-search optimizer for GRAPE coefficients."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np

from ..config import ExperimentConfig
from ..artifacts import ArtifactPaths
from .base import (
    GrapeControlProblem,
    OptimizationOutput,
    OptimizerState,
    StepStats,
    clip_gradients,
    evaluate_problem,
    history_to_arrays,
    safe_norm,
)


def _make_step_stats(
    iteration: int,
    cost: Dict[str, float],
    grad_norm: float,
    step_norm: float,
    lr_value: float,
    wall_time: float,
    calls: int,
) -> StepStats:
    """Package scalar metrics for logging and serialization."""

    return StepStats(
        iteration=iteration,
        total=float(cost.get("total", 0.0)),
        terminal=float(cost.get("terminal", 0.0)),
        path=float(cost.get("path", 0.0)),
        ensemble=float(cost.get("ensemble", 0.0)),
        power_penalty=float(cost.get("power_penalty", 0.0)),
        neg_penalty=float(cost.get("neg_penalty", 0.0)),
        grad_norm=grad_norm,
        step_norm=step_norm,
        lr=lr_value,
        wall_time_s=wall_time,
        calls_per_iter=calls,
    )


def optimize_linesearch(
    config: ExperimentConfig,
    _paths: ArtifactPaths,
    problem: GrapeControlProblem,
    *,
    coeffs0: np.ndarray | None = None,
) -> OptimizationOutput:
    """Run Armijo backtracking line search on GRAPE coefficients.
    
    Parameters
    ----------
    config : ExperimentConfig
    Experiment configuration supplying optimizer options.
    _paths : ArtifactPaths
    Artifact locations (unused by optimizer front-end).
    problem : GrapeControlProblem
    GRAPE problem definition containing basis and metadata.
    coeffs0 : numpy.ndarray, optional
    Optional initial coefficients overriding ``problem.coeffs_init``.
    
    Returns
    -------
    OptimizationOutput
    Structured result containing coefficients, metrics, and history.
    """


    options = dict(config.optimizer_options)
    max_iters = int(options.get("max_iters", 200))
    alpha0 = float(options.get("alpha0", options.get("learning_rate", 0.1)))
    beta = float(options.get("ls_beta", 0.5))
    sigma = float(options.get("ls_sigma", 1e-4))
    max_backtracks = int(options.get("ls_max_backtracks", 12))
    grad_clip = options.get("grad_clip")
    grad_tol = float(options.get("grad_tol", 1e-4))
    rtol = float(options.get("rtol", 1e-5))
    neg_kappa = float(options.get("neg_kappa", options.get("neg_epsilon", 10.0)))
    max_time_s = options.get("max_time_s")

    coeffs = (coeffs0.copy() if coeffs0 is not None else problem.coeffs_init.copy()).astype(np.float64)

    state = OptimizerState(coeffs=coeffs.copy(), grad=np.zeros_like(coeffs))
    start_time = time.perf_counter()
    prev_total: float | None = None
    status = "completed"
    total_backtracks = 0

    for iteration in range(1, max_iters + 1):
        iter_start = time.perf_counter()
        cost_dict, grad_coeffs, extras = evaluate_problem(problem, coeffs, neg_kappa=neg_kappa)
        total_cost = float(cost_dict.get("total", 0.0))
        grad_norm = safe_norm(grad_coeffs)
        calls = int(extras.get("oracle_calls", 1))

        if not np.isfinite(total_cost) or not np.isfinite(grad_coeffs).all():
            status = "failed_nonfinite"
            stats = _make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0, time.perf_counter() - iter_start, calls)
            state.record(stats)
            break

        if grad_norm <= grad_tol:
            status = "converged_grad"
            stats = _make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0, time.perf_counter() - iter_start, calls)
            state.record(stats)
            break

        if prev_total is not None:
            rel_impr = abs(prev_total - total_cost) / max(1.0, abs(prev_total))
            if rel_impr <= rtol:
                # Terminate early when relative improvement stalls.
                status = "converged_rtol"
                stats = _make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0, time.perf_counter() - iter_start, calls)
                state.record(stats)
                break

        clipped_grad = clip_gradients(grad_coeffs, max_norm=grad_clip)
        state.grad = clipped_grad
        direction = -clipped_grad
        grad_dot_dir = float(np.dot(clipped_grad, direction))
        if grad_dot_dir >= 0.0:
            status = "armijo_direction"
            stats = _make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0, time.perf_counter() - iter_start, calls)
            state.record(stats)
            break

        alpha = alpha0
        accepted = False
        calls_this_iter = calls
        candidate_coeffs = coeffs
        candidate_cost = cost_dict
        candidate_extras = extras

        for bt in range(max_backtracks + 1):
            # Backtrack along the search direction until Armijo condition holds.
            candidate_coeffs = coeffs + alpha * direction
            candidate_cost, candidate_grad, candidate_extras = evaluate_problem(
                problem, candidate_coeffs, neg_kappa=neg_kappa
            )
            calls_bt = int(candidate_extras.get("oracle_calls", 1))
            calls_this_iter += calls_bt
            total_backtracks += 1 if bt > 0 else 0
            lhs = float(candidate_cost.get("total", 0.0))
            rhs = total_cost + sigma * alpha * grad_dot_dir
            if lhs <= rhs:
                accepted = True
                break
            alpha *= beta

        if not accepted:
            status = "armijo_failed"
            stats = _make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0, time.perf_counter() - iter_start, calls_this_iter)
            state.record(stats)
            break

        coeffs = candidate_coeffs
        step_norm = safe_norm(alpha * direction)
        stats = _make_step_stats(iteration, cost_dict, grad_norm, step_norm, alpha, time.perf_counter() - iter_start, calls_this_iter)
        state.record(stats)
        prev_total = total_cost

        state.runtime_s = time.perf_counter() - start_time
        if max_time_s is not None and state.runtime_s >= float(max_time_s):
            status = "time_limit"
            break
    else:
        status = "max_iters"

    state.status = status
    final_cost, final_grad_coeffs, extras = evaluate_problem(problem, coeffs, neg_kappa=neg_kappa)
    state.runtime_s = time.perf_counter() - start_time
    history = history_to_arrays(state.history)

    omega = np.asarray(extras.get("omega"), dtype=np.float64)
    delta_arr = extras.get("delta")
    delta = None if delta_arr is None else np.asarray(delta_arr, dtype=np.float64)

    cost_terms: Dict[str, float] = {
        "terminal": float(final_cost.get("terminal", 0.0)),
        "path": float(final_cost.get("path", 0.0)),
        "ensemble": float(final_cost.get("ensemble", 0.0)),
        "power_penalty": float(final_cost.get("power_penalty", 0.0)),
        "neg_penalty": float(final_cost.get("neg_penalty", 0.0)),
        "total": float(final_cost.get("total", 0.0)),
        "terminal_eval": float(final_cost.get("terminal_eval", final_cost.get("terminal", 0.0))),
        "runtime_s": float(state.runtime_s),
        "total_backtracks": total_backtracks,
    }

    optimizer_state = {
        "status": status,
        "iterations": int(history["iter"].shape[0]),
        "grad_norm_final": safe_norm(final_grad_coeffs),
        "alpha0": alpha0,
        "beta": beta,
        "sigma": sigma,
        "objective": extras.get("objective", problem.objective),
        "total_backtracks": total_backtracks,
    }

    return OptimizationOutput(
        coeffs=coeffs.astype(np.float64),
        omega=omega,
        delta=delta,
        cost_terms=cost_terms,
        history=history,
        runtime_s=float(state.runtime_s),
        optimizer_state=optimizer_state,
        extras=extras,
    )

