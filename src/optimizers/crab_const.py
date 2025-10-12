"""Constant-step gradient descent on GRAPE coefficients."""
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
    make_step_stats,
    safe_norm,
)


def optimize_const(
    config: ExperimentConfig,
    _paths: ArtifactPaths,
    problem: GrapeControlProblem,
    *,
    coeffs0: np.ndarray | None = None,
) -> OptimizationOutput:
    """Run constant-step gradient descent on GRAPE coefficients.
    
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
    base_lr = float(options.get("learning_rate", options.get("alpha", 0.05)))
    lr_decay = float(options.get("lr_decay", 1.0))
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

    for iteration in range(1, max_iters + 1):
        iter_start = time.perf_counter()
        cost_dict, grad_coeffs, extras = evaluate_problem(problem, coeffs, neg_kappa=neg_kappa)
        total_cost = float(cost_dict.get("total", 0.0))
        grad_norm = safe_norm(grad_coeffs)
        calls = int(extras.get("oracle_calls", 1))

        if not np.isfinite(total_cost) or not np.isfinite(grad_coeffs).all():
            status = "failed_nonfinite"
            stats = make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0, time.perf_counter() - iter_start, calls)
            state.record(stats)
            break

        if grad_norm <= grad_tol:
            status = "converged_grad"
            stats = make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0, time.perf_counter() - iter_start, calls)
            state.record(stats)
            break

        if prev_total is not None:
            rel_impr = abs(prev_total - total_cost) / max(1.0, abs(prev_total))
            if rel_impr <= rtol:
                # Terminate early when relative improvement stalls.
                status = "converged_rtol"
                stats = make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0, time.perf_counter() - iter_start, calls)
                state.record(stats)
                break

        lr_t = base_lr * (lr_decay ** (iteration - 1))
        clipped_grad = clip_gradients(grad_coeffs, max_norm=grad_clip)
        state.grad = clipped_grad

        step = -lr_t * clipped_grad
        coeffs = coeffs + step
        step_norm = safe_norm(step)

        stats = make_step_stats(iteration, cost_dict, grad_norm, step_norm, lr_t, time.perf_counter() - iter_start, calls)
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
    }

    optimizer_state = {
        "status": status,
        "iterations": int(history["iter"].shape[0]),
        "grad_norm_final": safe_norm(final_grad_coeffs),
        "learning_rate": base_lr,
        "lr_decay": lr_decay,
        "objective": extras.get("objective", problem.objective),
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

