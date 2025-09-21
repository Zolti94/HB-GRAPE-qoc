"""Optimizer registry supporting custom methods."""
from __future__ import annotations

from typing import Callable, Dict

from ..config import ExperimentConfig
from ..artifacts import ArtifactPaths
from ..result import Result

OptimizerCallable = Callable[[ExperimentConfig, ArtifactPaths], Result]

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
