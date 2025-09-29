"""CRAB control utilities operating in microseconds (us) and rad/us."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]

__all__ = [
    "crab_linear_basis",
    "coeffs_to_control",
    "grad_control_wrt_coeffs",
    "ensure_time_grid_match",
]


def crab_linear_basis(
    t: NDArrayFloat,
    K: int,
    omegas: NDArrayFloat,
    phases: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """Return a sine-only CRAB basis matrix evaluated on time grid ``t``.

    Parameters
    ----------
    t : array_like
        Time samples in us with shape (T,).
    K : int
        Number of basis functions.
    omegas : array_like
        Angular frequencies in rad/us with shape (K,).
    phases : array_like, optional
        Phase offsets in radians with shape (K,). Defaults to zeros.

    Returns
    -------
    numpy.ndarray
        Basis matrix ``B`` with shape (T, K) and dtype float64 where
        ``B[:, j] = sin(omegas[j] * t + phases[j])``.
    """

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


def coeffs_to_control(
    B: NDArrayFloat,
    coeffs: Iterable[float],
    base: NDArrayFloat | float = 0.0,
) -> NDArrayFloat:
    """Assemble a control envelope from CRAB coefficients.

    Parameters
    ----------
    B : np.ndarray
        Basis matrix of shape (T, K).
    coeffs : iterable
        Coefficient vector length K.
    base : np.ndarray or float, optional
        Baseline envelope broadcast against the assembled pulses.
    """

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
