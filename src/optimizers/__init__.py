"""Optimizer registry supporting custom methods."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from ..artifacts import ArtifactPaths
from ..config import ExperimentConfig
from .base import GrapeControlProblem, NDArrayFloat, OptimizationOutput

OptimizerCallable = Callable[[ExperimentConfig, ArtifactPaths, GrapeControlProblem, NDArrayFloat | None], OptimizationOutput]

__all__ = [
    "OptimizerCallable",
    "register_optimizer",
    "get_optimizer",
    "available_optimizers",
]

_REGISTRY: Dict[str, OptimizerCallable] = {}


def register_optimizer(name: str, factory: OptimizerCallable, *, overwrite: bool = False) -> None:
    """Register an optimizer implementation under ``name``.

    Parameters
    ----------
    name : str
        Registry key used to retrieve the optimizer.
    factory : OptimizerCallable
        Callable that constructs or executes the optimizer.
    overwrite : bool, optional
        Allow replacing an existing registration when ``True``.
    """

    key = name.lower()
    if not overwrite and key in _REGISTRY:
        raise ValueError(f"Optimizer '{name}' already registered.")
    _REGISTRY[key] = factory


def get_optimizer(name: str) -> OptimizerCallable:
    """Return the optimizer factory previously registered under ``name``.

    Parameters
    ----------
    name : str
        Registry key corresponding to a registered optimizer.

    Returns
    -------
    OptimizerCallable
        Callable capable of running the chosen optimizer.

    Raises
    ------
    KeyError
        If ``name`` is not present in the registry.
    """

    key = name.lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Optimizer '{name}' is not registered.") from exc


def available_optimizers() -> tuple[str, ...]:
    """Return registered optimizer names sorted alphabetically.

    Returns
    -------
    tuple[str, ...]
        Sorted tuple of registry keys.
    """

    return tuple(sorted(_REGISTRY.keys()))


# Register built-in optimizers.
from .crab_adam import optimize_adam  # noqa: E402
from .crab_const import optimize_const  # noqa: E402
from .crab_linesearch import optimize_linesearch  # noqa: E402

register_optimizer("adam", optimize_adam, overwrite=True)
register_optimizer("const", optimize_const, overwrite=True)
register_optimizer("linesearch", optimize_linesearch, overwrite=True)

