
"""Reusable helpers for notebook-based GRAPE runners."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from ..baselines import (
    BasisSpec,
    GrapeBaselineConfig,
    PulseShapeSpec,
    TimeGridSpec,
    build_grape_baseline,
)
from ..config import BaselineSpec, ExperimentConfig, PenaltyConfig


DEFAULT_TIME_GRID = {"duration_us": 0.1, "num_points": 2001, "start_us": 0.0}
_DEFAULT_RHO0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
_DEFAULT_TARGET = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)


@dataclass(frozen=True)
class BaselineArrays:
    """Structured payload returned by :func:`prepare_baseline`."""

    config: GrapeBaselineConfig
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    psi0: np.ndarray
    target: np.ndarray
    t_us: np.ndarray
    dt_us: float


def coerce_vector(name: str, values: Sequence[float] | np.ndarray, expected_len: int) -> np.ndarray:
    """Return ``values`` as a float vector of ``expected_len`` or raise ValueError."""

    vec = np.asarray(values, dtype=np.float64).reshape(-1)
    if vec.size != expected_len:
        raise ValueError(f"{name} expected length {expected_len}, got {vec.size}")
    return vec


def _rho_to_state(rho: np.ndarray | Sequence[Sequence[complex]]) -> np.ndarray:
    """Convert a state vector or 2x2 density matrix to a normalized vector."""

    arr = np.asarray(rho, dtype=np.complex128)
    if arr.shape == (2,):
        vec = arr
    elif arr.shape == (2, 2):
        vals, vecs = np.linalg.eigh(arr)
        vec = vecs[:, int(np.argmax(vals))]
    else:
        raise ValueError("rho must have shape (2,) or (2, 2)")
    phase = np.exp(-1j * np.angle(vec[0])) if abs(vec[0]) > 1e-12 else 1.0
    return (vec * phase).astype(np.complex128)


def prepare_baseline(
    *,
    time_grid: Mapping[str, Any] | None = None,
    omega_shape: Mapping[str, Any],
    delta_shape: Mapping[str, Any] | None,
    K_omega: int,
    K_delta: int,
    rho0: np.ndarray | Sequence[Sequence[complex]] | None = None,
    target: np.ndarray | Sequence[Sequence[complex]] | None = None,
    initial_omega: Sequence[float] | np.ndarray | None = None,
    initial_delta: Sequence[float] | np.ndarray | None = None,
) -> BaselineArrays:
    """Construct baseline arrays using repository defaults unless overridden."""

    time_payload: MutableMapping[str, Any] = dict(DEFAULT_TIME_GRID)
    if time_grid is not None:
        time_payload.update(time_grid)
    time_spec = TimeGridSpec(**time_payload)
    basis_spec = BasisSpec(num_omega=int(K_omega), num_delta=int(K_delta))
    omega_spec = PulseShapeSpec(**omega_shape)
    delta_spec = PulseShapeSpec(**delta_shape) if delta_shape is not None else None

    rho_seed = _rho_to_state(rho0) if rho0 is not None else _rho_to_state(_DEFAULT_RHO0)
    target_seed = _rho_to_state(target) if target is not None else _rho_to_state(_DEFAULT_TARGET)

    baseline_cfg = GrapeBaselineConfig(
        time_grid=time_spec,
        omega=omega_spec,
        delta=delta_spec,
        basis=basis_spec,
        rho0=np.outer(rho_seed, rho_seed.conj()),
        target=np.outer(target_seed, target_seed.conj()),
    )
    arrays_raw, metadata = build_grape_baseline(baseline_cfg)
    arrays = {k: np.asarray(v).copy() for k, v in arrays_raw.items()}

    if initial_omega is not None:
        arrays["Omega0"] = coerce_vector("initial_omega", initial_omega, arrays["Omega0"].size)
    if initial_delta is not None and arrays.get("Delta0") is not None:
        arrays["Delta0"] = coerce_vector("initial_delta", initial_delta, arrays["Delta0"].size)

    t_us = np.asarray(arrays["t_us"], dtype=np.float64)
    dt_us = float(np.asarray(arrays["dt_us"], dtype=np.float64))

    return BaselineArrays(
        config=baseline_cfg,
        arrays=arrays,
        metadata=dict(metadata),
        psi0=_rho_to_state(arrays.get("rho0", rho_seed)),
        target=_rho_to_state(arrays.get("target", target_seed)),
        t_us=t_us,
        dt_us=dt_us,
    )


def build_base_config(
    baseline_cfg: GrapeBaselineConfig,
    *,
    run_name: str | None,
    artifact_root: str | Path | None,
    penalties: Mapping[str, Any] | PenaltyConfig | None,
    objective: str,
    base_optimizer_options: Mapping[str, Any],
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Create an :class:`ExperimentConfig` suitable for optimizer sweeps."""

    if isinstance(penalties, PenaltyConfig):
        penalty_cfg = penalties
    else:
        payload = dict(penalties or {})
        penalty_cfg = PenaltyConfig(
            power_weight=float(payload.get("power_weight", 0.0)),
            neg_weight=float(payload.get("neg_weight", 0.0)),
            neg_kappa=float(payload.get("neg_kappa", 10.0)),
        )

    baseline_spec = BaselineSpec(name=run_name or "baseline", params=baseline_cfg.to_dict())
    base_opts = dict(base_optimizer_options)
    config = ExperimentConfig(
        baseline=baseline_spec,
        run_name=run_name,
        artifacts_root=Path(artifact_root) if artifact_root is not None else None,
        optimizer_options=base_opts,
        penalties=penalty_cfg,
        metadata={"objective": str(objective)},
    )
    return config, base_opts


def method_options(
    method: str,
    base_options: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge base optimizer options with method-specific overrides."""

    options = dict(base_options)
    if overrides:
        options.update(overrides)
    options["method"] = method
    return options
