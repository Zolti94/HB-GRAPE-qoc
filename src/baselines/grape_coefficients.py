"""Construct baseline arrays for GRAPE coefficient optimizations.

This module centralises the deterministic pulse construction that used to live in
notebook cells or ad-hoc scripts.  All functions operate in microseconds (time)
and radians per microsecond (angular frequencies), mirroring the conventions used
throughout the GRAPE workflows in this repository.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

__all__ = [
    "BasisSpec",
    "GrapeBaselineConfig",
    "PulseShapeSpec",
    "TimeGridSpec",
    "build_grape_baseline",
    "write_baseline",
]


@dataclass(frozen=True)
class TimeGridSpec:
    """Uniform time grid specification expressed in microseconds.

    Parameters
    ----------
    duration_us : float
        Total control window length in microseconds.
    num_points : int
        Number of discretisation points (must be >= 2).
    start_us : float, optional
        Start time in microseconds.  Defaults to ``0.0``.
    """

    duration_us: float
    num_points: int
    start_us: float = 0.0

    def to_dict(self) -> Dict[str, float | int]:
        """Return a JSON-serialisable representation."""

        return {
            "duration_us": float(self.duration_us),
            "num_points": int(self.num_points),
            "start_us": float(self.start_us),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TimeGridSpec":
        """Instantiate from a mapping produced by :meth:`to_dict`."""

        return cls(
            duration_us=float(payload["duration_us"]),
            num_points=int(payload["num_points"]),
            start_us=float(payload.get("start_us", 0.0)),
        )

    def samples(self) -> Tuple[np.ndarray, float]:
        """Return ``(t_us, dt_us)`` for the uniform grid.

        Raises
        ------
        ValueError
            If ``num_points`` is smaller than 2 or ``duration_us`` is non-positive.
        """

        if self.num_points < 2:
            raise ValueError("Time grid requires at least two samples.")
        if self.duration_us <= 0.0:
            raise ValueError("duration_us must be positive.")
        stop = self.start_us + float(self.duration_us)
        t = np.linspace(self.start_us, stop, self.num_points, dtype=np.float64)
        dt = float(self.duration_us) / float(self.num_points - 1)
        return t, dt


@dataclass(frozen=True)
class PulseShapeSpec:
    """Deterministic pulse envelope expressed through a named shape.

    Parameters
    ----------
    kind : str
        Identifier of the pulse family. Supported values: ``"polynomial"``,
        ``"gaussian"``, ``"sech"``, ``"blackman"``, ``"sine_bump"``,
        ``"linear_chirp"``, ``"tanh_chirp"``, ``"zero"``.
    area_pi : float, optional
        Desired pulse area in units of ``pi``. If provided, the pulse is scaled so
        that ``sum_t pulse(t) * dt == area_pi * pi``.  Leave ``None`` to skip
        area-based normalisation.  For inherently zero-area shapes (e.g. odd
        detuning ramps) set ``area_pi`` to ``0.0`` so a well-defined scale exists.
    amplitude_scale : float, optional
        Additional multiplicative factor applied after area normalisation. This is
        useful for detuning envelopes where the magnitude is specified directly
        rather than via an integral constraint.
    options : Mapping[str, float], optional
        Extra numeric parameters forwarded to the shape generator (e.g. custom
        width for the Gaussian profile).
    """

    kind: str
    area_pi: float | None = None
    amplitude_scale: float | None = None
    options: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the specification."""

        return {
            "kind": str(self.kind),
            "area_pi": None if self.area_pi is None else float(self.area_pi),
            "amplitude_scale": None if self.amplitude_scale is None else float(self.amplitude_scale),
            "options": {str(k): float(v) for k, v in self.options.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PulseShapeSpec":
        """Instantiate from a mapping produced by :meth:`to_dict`."""

        return cls(
            kind=str(payload["kind"]),
            area_pi=None if payload.get("area_pi") is None else float(payload["area_pi"]),
            amplitude_scale=None if payload.get("amplitude_scale") is None else float(payload["amplitude_scale"]),
            options={str(k): float(v) for k, v in dict(payload.get("options", {})).items()},
        )


@dataclass(frozen=True)
class BasisSpec:
    """Harmonic basis description for GRAPE coefficient parameterisations.

    Parameters
    ----------
    num_omega : int
        Number of sine harmonics for the drive amplitude control. Set to ``0`` to
        disable optimisation of this channel.
    num_delta : int, optional
        Number of sine harmonics for the detuning control. Defaults to ``0``.
    omega_harmonics : Sequence[int], optional
        Explicit harmonic indices to use for the amplitude channel. When provided
        this overrides ``num_omega``.
    delta_harmonics : Sequence[int], optional
        Explicit harmonic indices to use for the detuning channel. When provided
        this overrides ``num_delta``.
    """

    num_omega: int
    num_delta: int = 0
    omega_harmonics: Sequence[int] | None = None
    delta_harmonics: Sequence[int] | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return {
            "num_omega": int(self.num_omega),
            "num_delta": int(self.num_delta),
            "omega_harmonics": None if self.omega_harmonics is None else [int(m) for m in self.omega_harmonics],
            "delta_harmonics": None if self.delta_harmonics is None else [int(m) for m in self.delta_harmonics],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BasisSpec":
        """Instantiate from a mapping produced by :meth:`to_dict`."""

        return cls(
            num_omega=int(payload.get("num_omega", 0)),
            num_delta=int(payload.get("num_delta", 0)),
            omega_harmonics=None
            if payload.get("omega_harmonics") is None
            else [int(v) for v in payload["omega_harmonics"]],
            delta_harmonics=None
            if payload.get("delta_harmonics") is None
            else [int(v) for v in payload["delta_harmonics"]],
        )

    def omega_modes(self) -> np.ndarray:
        """Return the harmonic indices for the amplitude channel."""

        if self.omega_harmonics is not None:
            modes = np.asarray(self.omega_harmonics, dtype=int)
        else:
            modes = np.arange(1, int(self.num_omega) + 1, dtype=int)
        return np.asarray(modes, dtype=int)

    def delta_modes(self) -> np.ndarray:
        """Return the harmonic indices for the detuning channel."""

        if self.delta_harmonics is not None:
            modes = np.asarray(self.delta_harmonics, dtype=int)
        else:
            modes = np.arange(1, int(self.num_delta) + 1, dtype=int)
        return np.asarray(modes, dtype=int)


def _default_rho0() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)


def _default_target() -> np.ndarray:
    return np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)


@dataclass(frozen=True)
class GrapeBaselineConfig:
    """Aggregate configuration for constructing deterministic GRAPE baselines.

    Parameters
    ----------
    time_grid : TimeGridSpec
        Uniform sampling grid used for both propagation and basis functions.
    omega : PulseShapeSpec
        Baseline envelope for the drive amplitude control.
    basis : BasisSpec
        Harmonic basis specification for coefficient optimisation.
    delta : PulseShapeSpec, optional
        Baseline envelope for the detuning control. If omitted, the detuning
        baseline is identically zero and its basis may still be configured via
        ``basis``.
    rho0 : numpy.ndarray, optional
        Initial density matrix. Defaults to ``|0><0|``.
    target : numpy.ndarray, optional
        Target density matrix. Defaults to ``|1><1|``.
    extra_metadata : Mapping[str, Any], optional
        Arbitrary metadata copied into the builder output.
    """

    time_grid: TimeGridSpec
    omega: PulseShapeSpec
    basis: BasisSpec
    delta: PulseShapeSpec | None = None
    rho0: np.ndarray = field(default_factory=_default_rho0)
    target: np.ndarray = field(default_factory=_default_target)
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the full configuration."""

        payload: Dict[str, Any] = {
            "time_grid": self.time_grid.to_dict(),
            "omega": self.omega.to_dict(),
            "basis": self.basis.to_dict(),
            "rho0": np.asarray(self.rho0, dtype=np.complex128).tolist(),
            "target": np.asarray(self.target, dtype=np.complex128).tolist(),
            "extra_metadata": _json_ready(self.extra_metadata),
        }
        if self.delta is not None:
            payload["delta"] = self.delta.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GrapeBaselineConfig":
        """Instantiate from a mapping produced by :meth:`to_dict`."""

        time_grid = TimeGridSpec.from_dict(payload["time_grid"])
        omega = PulseShapeSpec.from_dict(payload["omega"])
        basis = BasisSpec.from_dict(payload["basis"])
        delta_payload = payload.get("delta")
        delta = PulseShapeSpec.from_dict(delta_payload) if delta_payload is not None else None
        rho0 = np.asarray(payload.get("rho0", _default_rho0()), dtype=np.complex128)
        target = np.asarray(payload.get("target", _default_target()), dtype=np.complex128)
        extra_metadata = dict(payload.get("extra_metadata", {}))
        return cls(
            time_grid=time_grid,
            omega=omega,
            basis=basis,
            delta=delta,
            rho0=rho0,
            target=target,
            extra_metadata=extra_metadata,
        )


def _shape_values(kind: str, tau: np.ndarray, duration_us: float, options: Mapping[str, float]) -> np.ndarray:
    """Return baseline values for the requested shape.

    Parameters
    ----------
    kind : str
        Shape identifier (case-insensitive).
    tau : numpy.ndarray
        Time coordinates shifted so ``tau[0] == 0``.
    duration_us : float
        Total duration in microseconds.
    options : Mapping[str, float]
        Optional numerical modifiers (shape specific).

    Returns
    -------
    numpy.ndarray
        Unnormalised pulse samples with ``dtype=float``.
    """

    s = 2.0 * tau / float(duration_us) - 1.0
    kind_lc = kind.lower()
    if kind_lc == "polynomial":
        values = 1.0 - 3.0 * s**4 + 2.0 * s**6
    elif kind_lc == "gaussian":
        width = float(options.get("fwhm_fraction", 1.0))
        scale = max(width, 1e-12)
        values = np.exp(-4.0 * np.log(2.0) * (s / scale) ** 2)
    elif kind_lc == "sech":
        sharpness = float(options.get("sharpness", 4.0 * np.arccosh(2.0)))
        values = 1.0 / np.cosh(sharpness * s)
    elif kind_lc == "blackman":
        values = np.blackman(tau.size)
    elif kind_lc == "sine_bump":
        values = np.sin(np.pi * tau / float(duration_us)) ** 2
    elif kind_lc == "linear_chirp":
        values = s
    elif kind_lc == "tanh_chirp":
        beta = float(options.get("beta", 0.9))
        beta = np.clip(beta, -0.999999, 0.999999)
        values = np.tanh(2.0 * np.arctanh(beta) * s)
    elif kind_lc == "zero":
        values = np.zeros_like(tau)
    else:
        raise ValueError(f"Unknown pulse shape '{kind}'.")
    return values.astype(np.float64, copy=False)


def _scale_pulse(values: np.ndarray, dt_us: float, spec: PulseShapeSpec) -> np.ndarray:
    area_target = None if spec.area_pi is None else float(spec.area_pi) * np.pi
    scaled = np.asarray(values, dtype=np.float64)
    if area_target is not None:
        area_current = float(np.sum(scaled) * dt_us)
        if abs(area_current) < 1e-12:
            if abs(area_target) > 1e-12:
                raise ValueError(
                    "Cannot match non-zero area for a pulse with zero integral."
                )
            scale = 1.0
        else:
            scale = area_target / area_current
        scaled = scale * scaled
    if spec.amplitude_scale is not None:
        scaled = float(spec.amplitude_scale) * scaled
    return scaled.astype(np.float64)


def _build_pulse(
    spec: PulseShapeSpec,
    t_us: np.ndarray,
    duration_us: float,
) -> np.ndarray:
    tau = t_us - float(t_us[0])
    raw = _shape_values(spec.kind, tau, duration_us, spec.options)
    return _scale_pulse(raw, float(duration_us) / float(t_us.size - 1), spec)


def _harmonic_frequencies(modes: np.ndarray, duration_us: float) -> np.ndarray:
    if modes.size == 0:
        return np.empty((0,), dtype=np.float64)
    return 2.0 * np.pi * np.asarray(modes, dtype=np.float64) / float(duration_us)


def _sine_basis(tau: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    if omegas.size == 0:
        return np.zeros((tau.size, 0), dtype=np.float64)
    phase = np.outer(tau, np.asarray(omegas, dtype=np.float64))
    return np.sin(phase)


def _normalize_columns(matrix: np.ndarray, dt_us: float) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float64)
    working = np.asarray(matrix, dtype=np.float64)
    norms = np.sqrt(np.sum(working * working, axis=0) * float(dt_us))
    norms[norms == 0.0] = 1.0
    return working / norms


def _pulse_summary(pulse: np.ndarray, dt_us: float) -> Dict[str, float]:
    return {
        "area_pi": float(np.sum(pulse) * dt_us / np.pi),
        "max_rad_per_us": float(np.max(np.abs(pulse))) if pulse.size else 0.0,
        "rms_rad_per_us": float(np.sqrt(np.mean(pulse * pulse))) if pulse.size else 0.0,
    }


def build_grape_baseline(config: GrapeBaselineConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Construct deterministic GRAPE baseline arrays.

    Parameters
    ----------
    config : GrapeBaselineConfig
        High-level specification covering the time grid, baseline envelopes,
        harmonic basis, and initial/target states.

    Returns
    -------
    arrays : dict[str, numpy.ndarray]
        Dictionary containing the baseline waveforms and basis tensors. Keys:
        ``t_us``, ``dt_us`` (scalar stored as ``np.float64``), ``duration_us``,
        ``num_points``, ``omega_baseline``, ``delta_baseline``, ``omega_basis``,
        ``delta_basis``, ``omega_frequencies_rad_per_us``,
        ``delta_frequencies_rad_per_us``, ``rho0``, ``target``.
    metadata : dict[str, Any]
        Summary statistics (areas, maxima) merged with
        ``config.extra_metadata``.
    """

    t_us, dt_us = config.time_grid.samples()
    duration_us = config.time_grid.duration_us
    tau = t_us - float(t_us[0])

    omega0 = _build_pulse(config.omega, t_us, duration_us)
    if config.delta is not None:
        delta0 = _build_pulse(config.delta, t_us, duration_us)
    else:
        delta0 = np.zeros_like(omega0)

    omega_modes = config.basis.omega_modes()
    delta_modes = config.basis.delta_modes()

    omega_freqs = _harmonic_frequencies(omega_modes, duration_us)
    delta_freqs = _harmonic_frequencies(delta_modes, duration_us)

    omega_basis = _normalize_columns(_sine_basis(tau, omega_freqs), dt_us)
    delta_basis = _normalize_columns(_sine_basis(tau, delta_freqs), dt_us)

    arrays: Dict[str, np.ndarray] = {
        "t_us": np.asarray(t_us, dtype=np.float64),
        "dt_us": np.array(dt_us, dtype=np.float64),
        "duration_us": np.array(duration_us, dtype=np.float64),
        "num_points": np.array(t_us.size, dtype=np.int32),
        "omega_baseline": omega0.astype(np.float64),
        "delta_baseline": delta0.astype(np.float64),
        "omega_basis": omega_basis.astype(np.float64),
        "delta_basis": delta_basis.astype(np.float64),
        "omega_frequencies_rad_per_us": omega_freqs.astype(np.float64),
        "delta_frequencies_rad_per_us": delta_freqs.astype(np.float64),
        "rho0": np.asarray(config.rho0, dtype=np.complex128),
        "target": np.asarray(config.target, dtype=np.complex128),
    }

    metadata: Dict[str, Any] = {
        "time_grid": config.time_grid.to_dict(),
        "omega": config.omega.to_dict(),
        "basis": config.basis.to_dict(),
        "omega_summary": _pulse_summary(omega0, dt_us),
        "delta_summary": _pulse_summary(delta0, dt_us),
    }
    if config.delta is not None:
        metadata["delta"] = config.delta.to_dict()
    metadata.update(_json_ready(config.extra_metadata))
    return arrays, metadata


def write_baseline(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    out_dir: Path | str | None,
) -> None:
    """Persist baseline arrays and metadata to disk.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Arrays produced by :func:`build_grape_baseline`.
    metadata : Mapping[str, Any]
        JSON-serialisable metadata dictionary.
    out_dir : Path-like or None
        Target directory. If ``None`` the function is a no-op.
    """

    if out_dir is None:
        return
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "arrays.npz", **arrays)
    with (destination / "metadata.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(_json_ready(metadata), f, indent=2)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return value.tolist()
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.complexfloating):
        return complex(value)
    return value
