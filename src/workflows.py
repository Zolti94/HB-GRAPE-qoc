"""Workflow orchestration for optimization experiments (us/MHz units)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .artifacts import default_root, format_run_name, prepare_run_directory
from .config import ExperimentConfig
from .result import Result
from .optimizers import available_optimizers, get_optimizer, register_optimizer

__all__ = ["run_experiment", "register_optimizer", "available_optimizers"]


def _coerce_config(config: ExperimentConfig | Mapping[str, Any]) -> ExperimentConfig:
    if isinstance(config, ExperimentConfig):
        return config
    if isinstance(config, Mapping):
        return ExperimentConfig.from_dict(config)
    raise TypeError("config must be ExperimentConfig or mapping")


def run_experiment(
    config: ExperimentConfig | Mapping[str, Any],
    *,
    method: str = "adam",
    run_name: str | None = None,
    timestamp: datetime | None = None,
    exist_ok: bool = False,
) -> Result:
    """Run an experiment using a registered optimizer."""

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

    result = optimizer(experiment_config, paths)
    if not isinstance(result, Result):
        raise TypeError("Optimizer must return Result instance.")

    result.run_name = final_run_name
    result.artifacts_dir = paths.run_dir
    result.config = experiment_config
    return result
