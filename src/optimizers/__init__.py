"""Optimizer registry supporting custom methods."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from ..artifacts import ArtifactPaths
from ..config import ExperimentConfig
from .base import CrabProblem, NDArrayFloat, OptimizationOutput

OptimizerCallable = Callable[[ExperimentConfig, ArtifactPaths, CrabProblem, NDArrayFloat | None], OptimizationOutput]

__all__ = [
    "OptimizerCallable",
    "register_optimizer",
    "get_optimizer",
    "available_optimizers",
]

_REGISTRY: Dict[str, OptimizerCallable] = {}


def register_optimizer(name: str, factory: OptimizerCallable, *, overwrite: bool = False) -> None:
    key = name.lower()
    if not overwrite and key in _REGISTRY:
        raise ValueError(f"Optimizer '{name}' already registered.")
    _REGISTRY[key] = factory


def get_optimizer(name: str) -> OptimizerCallable:
    key = name.lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Optimizer '{name}' is not registered.") from exc


def available_optimizers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY.keys()))


# Register built-in optimizers.
from .crab_adam import optimize_adam  # noqa: E402
from .crab_const import optimize_const  # noqa: E402
from .crab_linesearch import optimize_linesearch  # noqa: E402

register_optimizer("adam", optimize_adam, overwrite=True)
register_optimizer("const", optimize_const, overwrite=True)
register_optimizer("linesearch", optimize_linesearch, overwrite=True)
