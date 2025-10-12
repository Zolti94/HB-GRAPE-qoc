"""CRAB control utilities operating in microseconds (us) and rad/us."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]

__all__ = [
    "crab_linear_basis",
    "coeffs_to_control",
    "grad_control_wrt_coeffs",
    "ensure_time_grid_match",
    "normalize_basis_columns",
    "harmonic_sine_basis",
    "build_normalized_harmonic_bases",
]


def crab_linear_basis(
    t: NDArrayFloat,
    K: int,
    omegas: NDArrayFloat,
    phases: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """Return a sine-only CRAB basis matrix evaluated on time grid ``t``."""

    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional.")
    omegas = np.asarray(omegas, dtype=np.float64)
    if omegas.ndim != 1:
        raise ValueError("omegas must be one-dimensional.")
    if omegas.size != int(K):
        raise ValueError("omegas length must equal K.")
    if phases is None:
        phases_arr = np.zeros_like(omegas)
    else:
        phases_arr = np.asarray(phases, dtype=np.float64)
        if phases_arr.shape != omegas.shape:
            raise ValueError("phases must match omegas shape.")

    basis = np.sin(np.outer(t, omegas) + phases_arr)
    return basis.astype(np.float64, copy=False)


def normalize_basis_columns(basis: NDArrayFloat, dt_us: float) -> NDArrayFloat:
    """Column-normalise ``basis`` under the discrete L2 inner product."""

    basis = np.asarray(basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("Basis must be a two-dimensional array.")
    scales = np.sqrt(np.sum(basis * basis, axis=0) * float(dt_us))
    scales[scales == 0.0] = 1.0
    return basis / scales


def harmonic_sine_basis(
    t: Sequence[float] | NDArrayFloat,
    modes: Sequence[int] | NDArrayFloat,
    duration_us: float,
) -> NDArrayFloat:
    """Return sine basis matrix for integer harmonic ``modes`` over ``duration_us``."""

    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.ndim != 1:
        raise ValueError("t must be one-dimensional.")
    modes_arr = np.asarray(modes, dtype=np.float64)
    if modes_arr.ndim != 1:
        raise ValueError("modes must be one-dimensional.")
    if modes_arr.size == 0:
        return np.zeros((t_arr.size, 0), dtype=np.float64)
    duration = float(duration_us)
    if duration <= 0.0:
        raise ValueError("duration_us must be positive.")
    tau = t_arr - float(t_arr[0])
    phase = np.pi * np.outer(tau / duration, modes_arr)
    return np.sin(phase).astype(np.float64, copy=False)


def build_normalized_harmonic_bases(
    t: Sequence[float] | NDArrayFloat,
    dt_us: float,
    duration_us: float,
    modes_omega: Sequence[int] | NDArrayFloat,
    modes_delta: Sequence[int] | NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Construct orthonormal sine bases aligned with the supplied time grid."""

    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.ndim != 1:
        raise ValueError("t must be one-dimensional.")
    if t_arr.size < 2:
        raise ValueError("Time grid must contain at least two samples.")
    dt = float(dt_us)
    duration = float(duration_us)
    grid_span = t_arr[0] + dt * (t_arr.size - 1)
    if not (np.isclose(t_arr[-1], duration) or np.isclose(grid_span, duration)):
        raise ValueError("Harmonic bases require grid to span the declared duration.")

    basis_omega = harmonic_sine_basis(t_arr, modes_omega, duration)
    basis_delta = harmonic_sine_basis(t_arr, modes_delta, duration)
    basis_omega = normalize_basis_columns(basis_omega, dt)
    basis_delta = normalize_basis_columns(basis_delta, dt)
    if basis_omega.shape[0] != t_arr.shape[0] or basis_delta.shape[0] != t_arr.shape[0]:
        raise ValueError("Basis rows must equal the number of time samples.")
    return basis_omega, basis_delta


def coeffs_to_control(
    B: NDArrayFloat,
    coeffs: Iterable[float],
    base: NDArrayFloat | float = 0.0,
) -> NDArrayFloat:
    """Assemble a control envelope from CRAB coefficients."""

    B = np.asarray(B, dtype=np.float64)
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError("Basis matrix B must be two-dimensional.")
    if coeffs_arr.ndim != 1 or coeffs_arr.size != B.shape[1]:
        raise ValueError("Coefficient vector size must match number of basis columns.")
    base_arr = np.asarray(base, dtype=np.float64)
    control = B @ coeffs_arr
    if base_arr.ndim == 0:
        control = control + float(base_arr)
    else:
        if base_arr.shape != (B.shape[0],):
            raise ValueError("Baseline envelope must have length equal to time samples.")
        control = control + base_arr
    return control


def grad_control_wrt_coeffs(B: NDArrayFloat) -> NDArrayFloat:
    """Return d(control)/d(coeffs) which equals the basis matrix for linear CRAB."""

    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError("Basis matrix must be two-dimensional.")
    return B


def ensure_time_grid_match(t_basis: NDArrayFloat, t_dyn: NDArrayFloat) -> None:
    """Raise if the CRAB basis grid differs from the dynamics grid (element-wise)."""

    t_basis = np.asarray(t_basis, dtype=np.float64)
    t_dyn = np.asarray(t_dyn, dtype=np.float64)
    if t_basis.shape != t_dyn.shape or not np.array_equal(t_basis, t_dyn):
        raise ValueError("CRAB basis and dynamics grids must match exactly.")
