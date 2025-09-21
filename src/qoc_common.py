"""
qoc_common.py - Shared utilities for GRAPE/CRAB notebooks.
- Baseline loading (arrays + policy)
- Physics: SU(2) step, forward/adjoint propagation
- Terminal cost & GRAPE gradient with fluence/negativity penalties
- Constraints: optional amplitude bounds helper
- Viz helpers: rad/s <-> MHz conversion and quick_plot

All physics arrays are expected in SI units (t in seconds, amplitudes in rad/s).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# ---------- Baseline I/O ----------
def load_baseline(base_dir: str | Path = "outputs/_baseline") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load baseline arrays and policy metadata.
    Returns: (arrays, policy) where
      arrays: dict with t, dt, T, Nt, Omega0, Delta0, MASK, rho0, target, NORM, SEED
      policy: dict loaded from metadata.json
    """
    base = Path(base_dir)
    with np.load(base / "arrays.npz", allow_pickle=True) as arrs_file:
        arrays = {k: np.asarray(arrs_file[k]) for k in arrs_file.files}
    with open(base / "metadata.json", "r", encoding="utf-8") as f:
        policy = json.load(f)
    arrays["t"] = arrays["t"].astype(float)
    arrays["dt"] = np.asarray(arrays["dt"], dtype=float)
    arrays["Omega0"] = arrays["Omega0"].astype(float)
    arrays["Delta0"] = arrays["Delta0"].astype(float)
    mask = arrays.get("MASK")
    arrays["MASK"] = np.asarray(mask, dtype=float) if mask is not None else None
    arrays["rho0"] = arrays["rho0"].astype(np.complex128)
    arrays["target"] = arrays["target"].astype(np.complex128)
    arrays["Nt"] = np.asarray(arrays["Nt"]).astype(int)
    arrays["SEED"] = np.asarray(arrays["SEED"]).astype(int)
    return arrays, policy

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
    if power_weight > 0.0:
        fluence = float(np.sum(Omega * Omega, dtype=float) * dt)
        cost += 0.5 * power_weight * fluence
    if neg_weight > 0.0:
        k = float(neg_kappa)
        if k <= 0.0:
            raise ValueError("neg_kappa must be positive.")
        neg = np.logaddexp(0.0, -k * Omega) / k  # softplus(-Omega)
        cost += 0.5 * neg_weight * float(np.sum(neg * neg, dtype=float) * dt)
    return cost

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
    if power_weight > 0.0:
        fluence = float(np.sum(Omega * Omega, dtype=float) * dt)
        cost += 0.5 * power_weight * fluence
        gO += power_weight * Omega * dt
    if neg_weight > 0.0:
        k = float(neg_kappa)
        if k <= 0.0:
            raise ValueError("neg_kappa must be positive.")
        neg = np.logaddexp(0.0, -k * Omega) / k
        sigmoid = 0.5 * (np.tanh(-0.5 * k * Omega) + 1.0)  # sigmoid(-k * Omega)
        cost += 0.5 * neg_weight * float(np.sum(neg * neg, dtype=float) * dt)
        gO += -(neg_weight * neg * sigmoid) * dt
    return cost, gO, gD

# ---------- Constraints ----------
def apply_bounds(x: np.ndarray, limit: float | None) -> np.ndarray:
    """Symmetric box bounds: clip to [-limit, limit] if limit provided."""
    return np.clip(x, -limit, limit) if (limit is not None) else x

# ---------- Viz helpers ----------
def radps_to_MHz(x: np.ndarray | float) -> np.ndarray | float:
    """Convert rad/s to MHz for display."""
    import numpy as _np
    return (_np.asarray(x) / (2 * _np.pi * 1e6)) if isinstance(x, _np.ndarray) else (x / (2 * np.pi * 1e6))

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

    t_us = t * 1e6

    def _to_MHz(arr):
        import numpy as _np
        return _np.asarray(arr) / (2 * np.pi * 1e6)

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
    "load_baseline",
    "U_step",
    "forward_rhos",
    "adjoint_lams",
    "terminal_cost",
    "terminal_cost_and_grad",
    "apply_bounds",
    "radps_to_MHz",
    "quick_plot",
]
