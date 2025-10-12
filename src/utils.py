"""Shared utilities for the GRAPE/CRAB workflows."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator
from datetime import datetime

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]

__all__ = [
    "ensure_dir",
    "json_ready",
    "require_real_finite",
    "set_random_seed",
    "time_block",
]


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return it as a Path object."""

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def json_ready(value: Any) -> Any:
    """Recursively convert numpy, pathlib, and datetime types to JSON primitives."""

    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.complexfloating, complex)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def require_real_finite(name: str, array: npt.NDArray[Any]) -> NDArrayFloat:
    """Return the array as float64 after verifying it is real and finite."""

    arr = np.asarray(array)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values.")
    if np.iscomplexobj(arr):
        if np.any(np.abs(arr.imag) > 0.0):
            raise ValueError(f"{name} must be real-valued.")
        arr = arr.real
    return arr.astype(np.float64, copy=False)


def set_random_seed(seed: int | None) -> np.random.Generator:
    """Return a numpy Generator after seeding the global RNG when seed provided."""

    if seed is None:
        return np.random.default_rng()
    np.random.seed(seed)
    return np.random.default_rng(seed)


@contextmanager
def time_block(registry: Dict[str, float] | None = None, key: str | None = None) -> Generator[Dict[str, float], None, None]:
    """Context manager yielding a dict containing the elapsed time in seconds."""

    start = time.perf_counter()
    stats: Dict[str, float] = {"elapsed": 0.0}
    try:
        yield stats
    finally:
        elapsed = time.perf_counter() - start
        stats["elapsed"] = elapsed
        if registry is not None and key is not None:
            registry[key] = elapsed

