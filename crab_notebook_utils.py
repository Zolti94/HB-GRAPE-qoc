"""Utility helpers shared by CRAB notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

I2 = np.eye(2, dtype=np.complex128)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def json_ready(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays for JSON serialization."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value

def collect_versions(modules: Mapping[str, Any]) -> Dict[str, str]:
    """Return module versions when available, else 'n/a'."""
    versions: Dict[str, str] = {}
    for name, module in modules.items():
        version = getattr(module, "__version__", None)
        versions[name] = str(version) if version is not None else "n/a"
    return versions

def validate_time_grid(t: np.ndarray, dt: float, T: float, *, atol: float = 1e-12) -> None:
    """Validate uniform grid consistency."""
    t = np.asarray(t, dtype=float)
    dt = float(dt)
    T = float(T)
    if t.ndim != 1:
        raise ValueError("Time grid must be one-dimensional.")
    if t.size < 2:
        raise ValueError("Time grid must contain at least two samples.")
    if not np.isclose(t[0], 0.0, atol=atol):
        raise ValueError("Time grid should start at 0.")
    expected_T = t[0] + dt * (t.size - 1)
    if not np.isclose(expected_T, T, atol=atol, rtol=0.0):
        raise ValueError("T does not align with dt * (N-1).")
    diffs = np.diff(t)
    if not np.allclose(diffs, dt, atol=atol, rtol=0.0):
        raise ValueError("Time grid spacing is inconsistent with dt.")

def assert_finite(name: str, array: np.ndarray) -> None:
    """Raise ValueError if array contains non-finite entries."""
    arr = np.asarray(array)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")

def penalty_terms(
    omega: np.ndarray,
    dt: float,
    *,
    power_weight: float = 0.0,
    neg_weight: float = 0.0,
    neg_kappa: float = 10.0,
) -> tuple[float, float, np.ndarray]:
    """Return (power_penalty, neg_penalty, grad_omega) for CRAB controls."""
    omega = np.asarray(omega, dtype=float)
    dt = float(dt)
    grad = np.zeros_like(omega, dtype=float)

    power_penalty = 0.0
    if power_weight != 0.0:
        power_penalty = 0.5 * float(power_weight) * float(np.sum(omega * omega) * dt)
        grad += float(power_weight) * omega * dt

    neg_penalty = 0.0
    if neg_weight != 0.0:
        k = float(neg_kappa)
        if k <= 0.0:
            raise ValueError("neg_kappa must be positive.")
        neg = np.logaddexp(0.0, -k * omega) / k
        neg_penalty = 0.5 * float(neg_weight) * float(np.sum(neg * neg) * dt)
        sigmoid = 0.5 * (np.tanh(-0.5 * k * omega) + 1.0)
        grad += -(float(neg_weight) * neg * sigmoid) * dt

    return power_penalty, neg_penalty, grad

def ground_state_projectors(omega: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Compute instantaneous ground-state projectors for the two-level Hamiltonian."""
    omega = np.asarray(omega, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if omega.shape != delta.shape:
        raise ValueError("Omega and Delta must share the same shape.")
    n = omega.size
    projectors = np.empty((n, 2, 2), dtype=np.complex128)
    for idx, (om, de) in enumerate(zip(omega, delta)):
        norm = float(np.hypot(om, de))
        if norm < 1e-14:
            projectors[idx] = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        else:
            nx = om / norm
            nz = de / norm
            projectors[idx] = 0.5 * (I2 - nx * SIGMA_X - nz * SIGMA_Z)
    return projectors

def bloch_coordinates(rhos: np.ndarray) -> np.ndarray:
    """Convert density matrices to Bloch-vector coordinates."""
    rhos = np.asarray(rhos, dtype=np.complex128)
    if rhos.ndim == 2:
        rhos = rhos[None, ...]
    coords = np.empty((rhos.shape[0], 3), dtype=float)
    for idx, rho in enumerate(rhos):
        coords[idx, 0] = 2.0 * np.real(rho[0, 1])
        coords[idx, 1] = 2.0 * np.imag(rho[1, 0])
        coords[idx, 2] = float(np.real(rho[0, 0] - rho[1, 1]))
    return coords

def population_excited(rhos: np.ndarray) -> np.ndarray:
    """Return excited-state population trajectory (rho_11)."""
    rhos = np.asarray(rhos, dtype=np.complex128)
    if rhos.ndim == 2:
        rhos = rhos[None, ...]
    return np.real(rhos[:, 1, 1])

__all__ = [
    "I2",
    "SIGMA_X",
    "SIGMA_Y",
    "SIGMA_Z",
    "ensure_dir",
    "json_ready",
    "collect_versions",
    "validate_time_grid",
    "assert_finite",
    "penalty_terms",
    "ground_state_projectors",
    "bloch_coordinates",
    "population_excited",
]
