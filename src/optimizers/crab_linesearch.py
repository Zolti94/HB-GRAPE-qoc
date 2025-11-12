"""Armijo backtracking line-search optimizer for GRAPE coefficients."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
from requests import options

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


def optimize_linesearch(
    config: ExperimentConfig,
    _paths: ArtifactPaths,
    problem: GrapeControlProblem,
    *,
    coeffs0: np.ndarray | None = None,
) -> OptimizationOutput:
    """Run Armijo backtracking line search on GRAPE coefficients."""

    options = dict(config.optimizer_options)

    # --- options (robust parsing, no breaking asserts) ------------------------
    max_iters = int(options.get("max_iters", 200))
    alpha0 = float(options.get("alpha0", options.get("learning_rate", 0.1)))
    beta = float(options.get("ls_beta", 0.5))
    sigma = float(options.get("ls_sigma", 1e-4))
    max_backtracks = int(options.get("ls_max_backtracks", 12))
    grad_clip_opt = options.get("grad_clip", None)
    grad_clip = float(grad_clip_opt) if grad_clip_opt is not None else None
    grad_tol = float(options.get("grad_tol", 1e-10))
    rtol = float(options.get("rtol", 1e-10))
    neg_kappa = float(options.get("neg_kappa", options.get("neg_epsilon", 10.0)))
    max_time_s = options.get("max_time_s")
    max_time_s = float(max_time_s) if max_time_s is not None else None

    # sanitize critical params (avoid crashing on bad config)
    if not (0.0 < beta < 1.0):
        beta = 0.5
    if not (0.0 < sigma < 0.5):
        sigma = 1e-4
    if not (alpha0 > 0.0):
        alpha0 = 0.1

    # --- init -----------------------------------------------------------------
    coeffs = (coeffs0.copy() if coeffs0 is not None else problem.coeffs_init.copy()).astype(np.float64)
    state = OptimizerState(coeffs=coeffs.copy(), grad=np.zeros_like(coeffs))
    start_time = time.perf_counter()
    prev_total: float | None = None
    status = "completed"
    total_backtracks = 0
    cached_triplet = None  # reuse (cost, grad, extras) after acceptance
    cur_alpha0 = float(alpha0)  # warm-start next iteration's initial step

    # --- main loop ------------------------------------------------------------
    for iteration in range(1, max_iters + 1):
        iter_t0 = time.perf_counter()
        if (max_time_s is not None) and ((iter_t0 - start_time) >= max_time_s):
            status = "time_limit"
            break

        # Evaluate current point (reuse cached from previous acceptance if available)
        if cached_triplet is None:
            cost_dict, grad_coeffs, extras = evaluate_problem(problem, coeffs, neg_kappa=neg_kappa)
        else:
            cost_dict, grad_coeffs, extras = cached_triplet
            cached_triplet = None

        total_cost = float(cost_dict.get("total", 0.0))
        grad_norm = safe_norm(grad_coeffs)
        calls = int(extras.get("oracle_calls", 1))

        # Non-finite guard
        if not np.isfinite(total_cost) or not np.isfinite(grad_coeffs).all():
            status = "failed_nonfinite"
            stats = make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0,
                                    time.perf_counter() - iter_t0, calls)
            state.record(stats)
            break

        # Gradient tolerance
        if grad_norm <= grad_tol:
            status = "converged_grad"
            stats = make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0,
                                    time.perf_counter() - iter_t0, calls)
            state.record(stats)
            break

        # Relative improvement stall (uses previous accepted cost)
        if prev_total is not None:
            rel_impr = abs(prev_total - total_cost) / max(abs(prev_total), 1e-12)
            if rel_impr <= rtol:
                status = "converged_rtol"
                stats = make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0,
                                        time.perf_counter() - iter_t0, calls)
                state.record(stats)
                break

        # Choose descent direction (optionally clip gradient for robustness)
        raw_grad = grad_coeffs
        if grad_clip is not None:
            clipped_grad = clip_gradients(raw_grad, max_norm=grad_clip)
        else:
            clipped_grad = raw_grad

        direction = -clipped_grad
        state.grad = clipped_grad  # keep state consistent with actual step direction

        # Correct Armijo inner product: grad(x)^T p
        gtd = float(np.dot(raw_grad, direction))
        if gtd >= 0.0:  # non-descent (can happen if clipping zeros the grad)
            status = "armijo_nondescent"
            stats = make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0,
                                    time.perf_counter() - iter_t0, calls)
            state.record(stats)
            break

        # Backtracking line search
        alpha = float(cur_alpha0)
        accepted = False
        calls_this_iter = calls
        bt_used = 0
        # print(f"Iter {iteration}: backtracking line search starting at alpha={alpha}")
        for bt in range(max_backtracks + 1):
            candidate_coeffs = coeffs + alpha * direction
            cand_cost, cand_grad, cand_extras = evaluate_problem(
                problem, candidate_coeffs, neg_kappa=neg_kappa
            )
            calls_this_iter += int(cand_extras.get("oracle_calls", 1))
            lhs = float(cand_cost.get("total", 0.0))
            rhs = total_cost + sigma * alpha * gtd

            if np.isfinite(lhs) and (lhs <= rhs):
                accepted = True
                bt_used = bt
                break
            alpha *= beta  # shrink and try again
            # print(f"  backtrack {bt+1}: alpha -> {alpha}")

        total_backtracks += bt_used

        if not accepted:
            status = "armijo_failed"
            stats = make_step_stats(iteration, cost_dict, grad_norm, 0.0, 0.0,
                                    time.perf_counter() - iter_t0, calls_this_iter)
            state.record(stats)
            break

        # Accept step and log POST-step metrics (more informative for histories)
        coeffs = candidate_coeffs
        step_norm = safe_norm(alpha * direction)
        stats = make_step_stats(iteration, cand_cost, safe_norm(cand_grad), step_norm, alpha,
                                time.perf_counter() - iter_t0, calls_this_iter)
        state.record(stats)

        # Prepare next iteration
        prev_total = lhs
        cached_triplet = (cand_cost, cand_grad, cand_extras)  # reuse next loop
        cur_alpha0 = alpha/beta  # warm-start next line search with last accepted step
        state.runtime_s = time.perf_counter() - start_time

        if (max_time_s is not None) and (state.runtime_s >= max_time_s):
            status = "time_limit"
            break
    else:
        status = "max_iters"

    # --- finalize -------------------------------------------------------------
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
        "alpha0": alpha0,           # original config value, preserved
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
