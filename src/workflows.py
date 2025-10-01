"""Workflow orchestration for optimization experiments (us/MHz units)."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping

import numpy as np

from .artifacts import default_root, format_run_name, prepare_run_directory
from .config import ExperimentConfig
from .optimizers import available_optimizers, get_optimizer, register_optimizer
from .optimizers.base import OptimizationOutput, build_grape_problem
from .result import Result
from .utils import json_ready

__all__ = ["run_experiment", "register_optimizer", "available_optimizers"]


def _coerce_config(config: ExperimentConfig | Mapping[str, Any]) -> ExperimentConfig:
    """Normalize user input into an :class:`ExperimentConfig` instance."""

    if isinstance(config, ExperimentConfig):
        return config
    if isinstance(config, Mapping):
        return ExperimentConfig.from_dict(config)
    raise TypeError("config must be ExperimentConfig or mapping")


def _write_json(path, payload) -> None:
    """Persist ``payload`` as indented JSON after converting NumPy types."""

    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def _write_history(path, history: Mapping[str, np.ndarray]) -> None:
    """Store optimizer history arrays in a NumPy ``.npz`` archive."""

    np.savez(path, **{k: np.asarray(v) for k, v in history.items()})


def _write_pulses(path, problem, outcome: OptimizationOutput) -> None:
    """Serialize time grid, controls, and coefficients for later analysis."""

    data = {
        "t_us": problem.t_us,
        "omega": outcome.omega,
        "omega_base": problem.omega_base,
        "basis_omega": problem.basis_omega,
        "coeffs_omega": outcome.coeffs[problem.omega_slice],
    }
    if outcome.delta is not None and problem.basis_delta is not None:
        data["delta"] = outcome.delta
        data["delta_base"] = problem.delta_base if problem.delta_base is not None else np.zeros_like(outcome.delta)
        data["basis_delta"] = problem.basis_delta
        data["coeffs_delta"] = outcome.coeffs[problem.delta_slice]
    np.savez(path, **data)


def _build_result(
    run_name: str,
    experiment_config: ExperimentConfig,
    problem,
    outcome: OptimizationOutput,
    artifacts_dir,
) -> Result:
    """Assemble a :class:`Result` instance from optimizer outputs."""

    pulses_dict = {
        "t_us": problem.t_us,
        "omega": outcome.omega,
        "omega_base": problem.omega_base,
        "delta": outcome.delta,
        "delta_base": problem.delta_base,
        "coeffs": outcome.coeffs,
        "basis_meta": {
            "omega_shape": problem.basis_omega.shape,
            "delta_shape": None if problem.basis_delta is None else problem.basis_delta.shape,
            "baseline_name": problem.metadata.get("baseline_name"),
        },
    }

    optimizer_state = dict(outcome.optimizer_state)
    if outcome.extras:
        optimizer_state.setdefault('extras', outcome.extras)

    return Result(
        run_name=run_name,
        artifacts_dir=artifacts_dir,
        config=experiment_config,
        history={k: np.asarray(v) for k, v in outcome.history.items()},
        final_metrics=outcome.cost_terms,
        pulses=pulses_dict,
        optimizer_state=optimizer_state,
    )


def run_experiment(
    config: ExperimentConfig | Mapping[str, Any],
    *,
    method: str = "adam",
    run_name: str | None = None,
    timestamp: datetime | None = None,
    exist_ok: bool = False,
) -> Result:
    """Run an experiment using a registered optimizer and persist artifacts.

    Parameters
    ----------
    config : ExperimentConfig or Mapping[str, Any]
        Base configuration or overrides describing the experiment.
    method : str, optional
        Optimizer registry key; overrides ``config.optimizer_options['method']`` when provided.
    run_name : str, optional
        Custom run directory name; defaults to ``config.run_name`` or a timestamped slug.
    timestamp : datetime, optional
        Timestamp injected into the run name; ``datetime.utcnow`` when omitted.
    exist_ok : bool, optional
        Allow reuse of an existing run directory (helpful for reruns).

    Returns
    -------
    Result
        Structured record containing metrics, histories, and saved controls.
    """

    experiment_config = _coerce_config(config)
    method_name = method or experiment_config.optimizer_options.get("method", "")
    if not method_name:
        raise ValueError("Optimizer method not specified.")

    root = default_root(experiment_config)
    final_run_name = run_name or experiment_config.run_name or format_run_name(
        experiment_config,
        method_name,
        timestamp=timestamp,
    )

    paths = prepare_run_directory(final_run_name, root=root, exist_ok=exist_ok)

    try:
        optimizer = get_optimizer(method_name)
    except KeyError as exc:
        available = ", ".join(available_optimizers()) or "(none registered)"
        raise ValueError(f"Optimizer '{method_name}' not registered. Available: {available}") from exc

    problem, coeffs0, _ = build_grape_problem(experiment_config)
    outcome = optimizer(experiment_config, paths, problem, coeffs0=coeffs0)
    if not isinstance(outcome, OptimizationOutput):
        raise TypeError("Optimizer must return OptimizationOutput.")

    metrics_payload = dict(outcome.cost_terms)
    metrics_payload["status"] = outcome.optimizer_state.get("status", "unknown")
    metrics_payload.setdefault('terminal', 0.0)
    metrics_payload.setdefault('path', 0.0)
    metrics_payload.setdefault('ensemble', 0.0)
    metrics_payload.setdefault('power_penalty', 0.0)
    metrics_payload.setdefault('neg_penalty', 0.0)
    metrics_payload['objective'] = problem.objective
    metrics_payload['terminal_infidelity'] = metrics_payload['terminal']
    metrics_payload['path_infidelity'] = metrics_payload['path']
    metrics_payload['ensemble_infidelity'] = metrics_payload['ensemble']
    metrics_payload['terminal_eval'] = float(outcome.cost_terms.get('terminal_eval', metrics_payload['terminal']))
    # Populate defaults so downstream plotting/reporting code can rely on common keys.

    _write_json(paths.config_json, experiment_config.to_dict())
    _write_json(paths.metrics_json, metrics_payload)
    _write_history(paths.history_npz, outcome.history)
    _write_pulses(paths.pulses_npz, problem, outcome)
    paths.log_path.write_text(
        f"status={metrics_payload['status']} total={metrics_payload.get('total', float('nan')):.6f}\n",
        encoding="utf-8",
    )

    result = _build_result(final_run_name, experiment_config, problem, outcome, paths.run_dir)
    return result


