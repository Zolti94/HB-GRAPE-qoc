"""
qoc_common.py - Shared utilities for GRAPE/CRAB notebooks.
- Physics: SU(2) step, forward/adjoint propagation
- Terminal cost & GRAPE gradient with fluence/negativity penalties
- Constraints: optional amplitude bounds helper
- Viz helpers: rad/s <-> MHz conversion and quick_plot

All physics arrays are expected in microsecond / rad-per-microsecond units.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import numpy as np

from .penalties import compute_penalties, penalty_terms as penalty_terms_shared

# ---------- Physics ----------
I2 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

def U_step(omega: float, delta: float, dt: float) -> np.ndarray:
    """Analytic SU(2) step for H = 0.5 * (delta * sigma_z + omega * sigma_x)."""
    H = 0.5 * (delta * sigma_z + omega * sigma_x)
    mu = 0.5 * np.sqrt(delta * delta + omega * omega)
    if mu < 1e-14:
        return I2 - 1j * H * dt - 0.5 * (H @ H) * (dt * dt)
    c = np.cos(mu * dt)
    s = np.sin(mu * dt) / (mu + 0.0)
    return c * I2 - 1j * s * H

def forward_rhos(Omega: np.ndarray, Delta: np.ndarray, rho0: np.ndarray, dt: float) -> np.ndarray:
    """Propagate density matrices along the time grid."""
    Nt = Omega.shape[0]
    rhos = np.empty((Nt, 2, 2), dtype=np.complex128)
    rhos[0] = rho0
    for k in range(Nt - 1):
        U = U_step(float(Omega[k]), float(Delta[k]), dt)
        rhos[k + 1] = U @ rhos[k] @ U.conj().T
    return rhos

def adjoint_lams(Omega: np.ndarray, Delta: np.ndarray, Lambda_T: np.ndarray, dt: float) -> np.ndarray:
    """Backward propagate adjoint costates."""
    Nt = Omega.shape[0]
    lams = np.empty((Nt, 2, 2), dtype=np.complex128)
    lams[-1] = Lambda_T
    for k in range(Nt - 2, -1, -1):
        U = U_step(float(Omega[k]), float(Delta[k]), dt)
        lams[k] = U.conj().T @ lams[k + 1] @ U
    return lams

# ---------- Cost & Gradient (Terminal) ----------
_PULSE_AREA_WARNING_EMITTED = False

def _consume_deprecated_pulse_kwargs(kwargs: Dict[str, Any]) -> None:
    """Handle legacy pulse-area kwargs without affecting the new API."""
    global _PULSE_AREA_WARNING_EMITTED
    if not kwargs:
        return
    target = kwargs.pop("pulse_area_target", None)
    weight = kwargs.pop("pulse_area_weight", 0.0)
    try:
        weight_value = float(weight)
    except (TypeError, ValueError):
        weight_value = 0.0
    if target is not None or weight_value != 0.0:
        if not _PULSE_AREA_WARNING_EMITTED:
            warnings.warn(
                "pulse_area_target/pulse_area_weight are deprecated; use power_weight instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            _PULSE_AREA_WARNING_EMITTED = True
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {tuple(sorted(kwargs))}")




def terminal_cost(
    Omega: np.ndarray,
    Delta: np.ndarray,
    rho0: np.ndarray,
    dt: float,
    target: np.ndarray,
    *,
    power_weight: float = 0.0,
    neg_kappa: float = 10.0,
    neg_weight: float = 0.0,
    **kwargs: Any,
) -> float:
    """Terminal infidelity plus optional fluence and smooth negativity penalties."""
    kwargs = dict(kwargs)
    _consume_deprecated_pulse_kwargs(kwargs)
    Omega = np.asarray(Omega, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    dt = float(np.asarray(dt, dtype=float))
    rhos = forward_rhos(Omega, Delta, rho0, dt)
    cost = 1.0 - float(np.real(np.trace(rhos[-1] @ target)))
    power_penalty, neg_penalty, _, _ = compute_penalties(
        Omega,
        Delta,
        dt,
        power_weight=power_weight,
        neg_weight=neg_weight,
        neg_kappa=neg_kappa,
    )
    return cost + power_penalty + neg_penalty

def terminal_cost_and_grad(
    Omega: np.ndarray,
    Delta: np.ndarray,
    rho0: np.ndarray,
    dt: float,
    target: np.ndarray,
    *,
    power_weight: float = 0.0,
    neg_kappa: float = 10.0,
    neg_weight: float = 0.0,
    **kwargs: Any,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Return (cost, g_Omega, g_Delta) with optional fluence and negativity penalties.
    Gradients follow the GRAPE commutator form; the last slice remains zero
    for the piecewise-constant controls propagated here.
    """
    kwargs = dict(kwargs)
    _consume_deprecated_pulse_kwargs(kwargs)
    Omega = np.asarray(Omega, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    dt = float(np.asarray(dt, dtype=float))
    Nt = Omega.shape[0]
    rhos = forward_rhos(Omega, Delta, rho0, dt)
    cost = 1.0 - float(np.real(np.trace(rhos[-1] @ target)))
    lams = adjoint_lams(Omega, Delta, target, dt)
    dHx = 0.5 * sigma_x
    dHz = 0.5 * sigma_z
    gO = np.zeros_like(Omega, dtype=float)
    gD = np.zeros_like(Delta, dtype=float)
    for k in range(Nt - 1):
        rho_k = rhos[k]
        lam_next = lams[k + 1]
        gO[k] = -np.imag(np.trace(lam_next @ (dHx @ rho_k - rho_k @ dHx))) * dt
        gD[k] = -np.imag(np.trace(lam_next @ (dHz @ rho_k - rho_k @ dHz))) * dt
    power_penalty, neg_penalty, pen_grad_omega, pen_grad_delta = compute_penalties(
        Omega,
        Delta,
        dt,
        power_weight=power_weight,
        neg_weight=neg_weight,
        neg_kappa=neg_kappa,
    )
    cost += power_penalty + neg_penalty
    gO += pen_grad_omega
    gD += pen_grad_delta
    return cost, gO, gD


# Backwards-compatible export
penalty_terms = penalty_terms_shared
# ---------- Constraints ----------
def apply_bounds(x: np.ndarray, limit: float | None) -> np.ndarray:
    """Symmetric box bounds: clip to [-limit, limit] if limit provided."""
    return np.clip(x, -limit, limit) if (limit is not None) else x

# ---------- Viz helpers ----------
def radus_to_MHz(x: np.ndarray | float) -> np.ndarray | float:
    """Convert rad/us to MHz for display."""
    import numpy as _np
    array_x = _np.asarray(x)
    converted = array_x / (2 * _np.pi)
    return converted if isinstance(x, _np.ndarray) else float(converted)

def quick_plot(
    t: np.ndarray,
    Omega0: np.ndarray,
    Delta0: np.ndarray,
    Omega: np.ndarray,
    Delta: np.ndarray,
    cost_hist: np.ndarray,
    times_s: np.ndarray,
) -> None:
    """Quick three-panel plot in microseconds and MHz with baseline overlays."""
    import matplotlib.pyplot as plt

    t_us = np.asarray(t, dtype=float)  # Already in microseconds.

    def _to_MHz(arr):
        return np.asarray(arr, dtype=float) / (2 * np.pi)

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), layout="constrained")
    axs[0].plot(t_us, _to_MHz(Omega), label=r"$\\Omega(t)$")
    axs[0].plot(t_us, _to_MHz(Delta), label=r"$\\Delta(t)$")
    axs[0].plot(t_us, _to_MHz(Omega0), "--", label=r"$\\Omega_0(t)$")
    axs[0].plot(t_us, _to_MHz(Delta0), "--", label=r"$\\Delta_0(t)$")
    axs[0].set_xlabel("Time (us)")
    axs[0].set_ylabel("Control amplitudes (MHz)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].semilogy(np.arange(1, len(cost_hist) + 1), cost_hist, label="Cost")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Cost")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(times_s, cost_hist, label="Cost")
    axs[2].set_xlabel("Wall time (s)")
    axs[2].set_ylabel("Cost")
    axs[2].grid(True)
    axs[2].legend()

    plt.show()

__all__ = [
    "U_step",
    "forward_rhos",
    "adjoint_lams",
    "terminal_cost",
    "terminal_cost_and_grad",
    "penalty_terms",
    "apply_bounds",
    "radus_to_MHz",
    "quick_plot",
]
