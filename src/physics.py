"""Two-level system physics helpers using microseconds (us) and radians per microsecond (rad/us).

All time values are expressed in microseconds and angular frequencies in rad/us.
State vectors are complex-valued arrays normalized in the computational basis
with |0> = [1, 0]."""
from __future__ import annotations

from typing import Callable, Mapping, MutableMapping, Sequence

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]
NDArrayComplex = npt.NDArray[np.complex128]

__all__ = [
    "make_time_grid",
    "hamiltonian_t",
    "propagate_piecewise_const",
    "propagate_with_controls",
    "bloch_components",
    "fidelity_pure",
    "adjoint_steps",
    "validate_units_and_shapes",
    "rho_to_state",
]

SIGMA_X: NDArrayComplex = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y: NDArrayComplex = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z: NDArrayComplex = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
I2: NDArrayComplex = np.eye(2, dtype=np.complex128)

_GROUND: NDArrayComplex = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)


def make_time_grid(t0: float, t1: float, samples: int) -> NDArrayFloat:
    """Return an inclusive, strictly increasing time grid in microseconds.

    Parameters
    ----------
    t0 : float
        Starting time in us.
    t1 : float
        Final time in us; must satisfy t1 > t0.
    samples : int
        Number of sample points (>= 2). The grid is inclusive of both endpoints.
    """

    if samples < 2:
        raise ValueError("Time grid must contain at least two samples.")
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0 for a forward grid.")
    grid = np.linspace(float(t0), float(t1), int(samples), dtype=np.float64)
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("Generated time grid is not strictly increasing.")
    return grid


def hamiltonian_t(omega: NDArrayFloat, delta: NDArrayFloat) -> Callable[[int], NDArrayComplex]:
    """Return H_k = 0.5 * (omega[k] * sigma_x + delta[k] * sigma_z) for index k."""

    omega = np.asarray(omega, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if omega.shape != delta.shape:
        raise ValueError("omega and delta must share the same shape.")

    def _H(k: int) -> NDArrayComplex:
        return 0.5 * (omega[k] * SIGMA_X + delta[k] * SIGMA_Z)

    return _H


def _su2_step(omega_k: float, delta_k: float, dt_us: float) -> NDArrayComplex:
    """Analytic SU(2) propagator for a single time slice (rad/us, us)."""

    dt = float(dt_us)
    omega = float(omega_k)
    delta = float(delta_k)
    H = 0.5 * (delta * SIGMA_Z + omega * SIGMA_X)
    mu = 0.5 * float(np.hypot(delta, omega))
    if mu < 1e-14:
        return I2 - 1j * H * dt - 0.5 * (H @ H) * (dt * dt)
    cos_term = np.cos(mu * dt)
    sin_term = np.sin(mu * dt) / mu
    return cos_term * I2 - 1j * sin_term * H


def _psi_to_rho(psi: NDArrayComplex) -> NDArrayComplex:
    """Return the pure-state density matrix ``|psi><psi|``.

    Parameters
    ----------
    psi : numpy.ndarray
        State vector in the computational basis with shape ``(2,)``.

    Returns
    -------
    numpy.ndarray
        Rank-1 density matrix compatible with Bloch-vector utilities.
    """

    vec = np.asarray(psi, dtype=np.complex128)
    if vec.shape != (2,):
        raise ValueError("State vector must have shape (2,).")
    return np.outer(vec, vec.conj())


def rho_to_state(rho: NDArrayComplex | Sequence[Sequence[complex]] | Sequence[complex]) -> NDArrayComplex:
    """Return the dominant eigenvector of ``rho`` with a real leading component.

    Accepts either a state vector shaped ``(2,)`` or a ``(2, 2)`` density matrix.
    """

    arr = np.asarray(rho, dtype=np.complex128)
    if arr.shape == (2,):
        vec = arr
    elif arr.shape == (2, 2):
        vals, vecs = np.linalg.eigh(arr)
        idx = int(np.argmax(vals))
        vec = vecs[:, idx]
    else:
        raise ValueError("rho must be a state vector (2,) or density matrix (2, 2).")
    phase = np.exp(-1j * np.angle(vec[0])) if abs(vec[0]) > 1e-12 else 1.0
    return (vec * phase).astype(np.complex128)


def propagate_piecewise_const(
    omega: NDArrayFloat,
    delta: NDArrayFloat,
    dt_us: float,
    *,
    psi0: NDArrayComplex | None = None,
    U0: NDArrayComplex | None = None,
) -> dict[str, NDArrayComplex | NDArrayFloat]:
    """Propagate a two-level system with piecewise-constant controls.

    Parameters
    ----------
    omega, delta : array_like
        Control sequences in rad/us with shape (T,).
    dt_us : float
        Time-step duration in us.
    psi0 : np.ndarray, optional
        Initial state vector (2,) complex128. Defaults to |0>.
    U0 : np.ndarray, optional
        Initial unitary propagator (2, 2). Defaults to identity.

    Returns
    -------
    dict
        Keys:
          - "U_T": accumulated unitary at final time (2, 2)
          - "U_hist": per-step propagators shaped (T, 2, 2)
          - "psi_path": state vectors shaped (T + 1, 2)
          - "psi_T": final state vector (2,)
          - "rho_path": density matrices shaped (T + 1, 2, 2)
    """

    omega = np.asarray(omega, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if omega.shape != delta.shape:
        raise ValueError("omega and delta must share shape.")
    steps = omega.size
    dt = float(dt_us)

    psi_init = np.asarray(psi0, dtype=np.complex128) if psi0 is not None else _GROUND.copy()
    if psi_init.shape != (2,):
        raise ValueError("psi0 must have shape (2,).")
    U_accum = np.asarray(U0, dtype=np.complex128) if U0 is not None else I2.copy()
    if U_accum.shape != (2, 2):
        raise ValueError("U0 must have shape (2, 2).")

    U_hist = np.empty((steps, 2, 2), dtype=np.complex128)
    psi_path = np.empty((steps + 1, 2), dtype=np.complex128)
    rho_path = np.empty((steps + 1, 2, 2), dtype=np.complex128)
    psi_path[0] = psi_init
    rho_path[0] = _psi_to_rho(psi_init)

    for k in range(steps):
        Uk = _su2_step(omega[k], delta[k], dt)
        U_hist[k] = Uk
        psi_path[k + 1] = Uk @ psi_path[k]
        rho_path[k + 1] = _psi_to_rho(psi_path[k + 1])

    for Uk in reversed(U_hist):
        U_accum = Uk @ U_accum

    return {
        "U_T": U_accum,
        "U_hist": U_hist,
        "psi_path": psi_path,
        "psi_T": psi_path[-1],
        "rho_path": rho_path,
    }


def propagate_with_controls(
    controls: Mapping[str, NDArrayFloat],
    dt_us: float,
    *,
    psi0: NDArrayComplex | None = None,
    U0: NDArrayComplex | None = None,
) -> dict[str, NDArrayComplex | NDArrayFloat]:
    """Wrapper around :func:`propagate_piecewise_const` using a control mapping."""

    omega = np.asarray(controls["omega"], dtype=np.float64)
    delta = np.asarray(controls["delta"], dtype=np.float64)
    return propagate_piecewise_const(omega, delta, dt_us, psi0=psi0, U0=U0)


def bloch_components(state: NDArrayComplex) -> NDArrayFloat:
    """Convert a state vector or density matrix to Bloch-vector components."""

    arr = np.asarray(state)
    if arr.shape == (2,):
        rho = _psi_to_rho(arr)
    elif arr.shape == (2, 2):
        rho = np.asarray(arr, dtype=np.complex128)
    else:
        raise ValueError("Input must be a state vector (2,) or density matrix (2, 2).")

    x = float(np.real(np.trace(rho @ SIGMA_X)))
    y = float(np.real(np.trace(rho @ SIGMA_Y)))
    z = float(np.real(np.trace(rho @ SIGMA_Z)))
    return np.array([x, y, z], dtype=np.float64)


def fidelity_pure(psi_T: NDArrayComplex, psi_target: NDArrayComplex) -> float:
    """Return |<psi_target|psi_T>|^2 for normalized state vectors."""

    psi_T = np.asarray(psi_T, dtype=np.complex128)
    target = np.asarray(psi_target, dtype=np.complex128)
    overlap = np.vdot(target, psi_T)
    return float(np.abs(overlap) ** 2)


def adjoint_steps(
    omega: NDArrayFloat,
    delta: NDArrayFloat,
    dt_us: float,
    *,
    psi0: NDArrayComplex | None = None,
    psi_target: NDArrayComplex | None = None,
) -> MutableMapping[str, NDArrayComplex | NDArrayFloat]:
    """Compute forward and adjoint trajectories required for GRAPE gradients.

    Returns
    -------
    dict
        Keys include forward densities ('forward_rhos': (T + 1, 2, 2)), adjoint
        costates ('adjoint_lams': (T + 1, 2, 2)), per-step propagators, and the
        time-step size in us.
    """

    omega = np.asarray(omega, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if omega.shape != delta.shape:
        raise ValueError("omega and delta must share shape.")
    steps = omega.size
    dt = float(dt_us)

    psi_init = np.asarray(psi0, dtype=np.complex128) if psi0 is not None else _GROUND.copy()
    rho0 = _psi_to_rho(psi_init)
    target_state = (
        np.asarray(psi_target, dtype=np.complex128)
        if psi_target is not None
        else np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    )
    target_rho = _psi_to_rho(target_state)

    U_hist = np.empty((steps, 2, 2), dtype=np.complex128)
    forward_rhos = np.empty((steps + 1, 2, 2), dtype=np.complex128)
    forward_rhos[0] = rho0

    for k in range(steps):
        Uk = _su2_step(omega[k], delta[k], dt)
        U_hist[k] = Uk
        forward_rhos[k + 1] = Uk @ forward_rhos[k] @ Uk.conj().T

    adjoint_lams = np.empty((steps + 1, 2, 2), dtype=np.complex128)
    adjoint_lams[-1] = target_rho
    for k in range(steps - 1, -1, -1):
        Uk = U_hist[k]
        adjoint_lams[k] = Uk.conj().T @ adjoint_lams[k + 1] @ Uk

    return {
        "dt_us": dt,
        "forward_rhos": forward_rhos,
        "adjoint_lams": adjoint_lams,
        "U_hist": U_hist,
        "psi0": psi_init,
        "psi_target": target_state,
    }


def validate_units_and_shapes(
    omega: NDArrayFloat,
    delta: NDArrayFloat,
    t: NDArrayFloat,
) -> None:
    """Validate that controls (rad/us) align with a strictly increasing time grid (us)."""

    omega = np.asarray(omega)
    delta = np.asarray(delta)
    t = np.asarray(t)
    if omega.ndim != 1 or delta.ndim != 1 or t.ndim != 1:
        raise ValueError("omega, delta, and t must be one-dimensional arrays.")
    if omega.shape != delta.shape:
        raise ValueError("omega and delta must share shape.")
    if t.size != omega.size + 1 and t.size != omega.size:
        raise ValueError("time grid length must match control length or control length + 1.")
    if not np.isfinite(omega).all() or not np.isfinite(delta).all() or not np.isfinite(t).all():
        raise ValueError("Controls and time grid must contain finite values.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("Time grid must be strictly increasing.")
