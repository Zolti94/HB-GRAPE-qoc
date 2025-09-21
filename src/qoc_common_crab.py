"""
qoc_common_crab.py - Utilities for CRAB-parameterized controls with gradient-based optimizers.
This module mirrors the shared physics helpers from qoc_common.py but exposes
helpers to assemble controls from sine bases, enforce grid consistency, and
differentiate the terminal cost with respect to CRAB coefficients.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from src.qoc_common import (
    terminal_cost,
    terminal_cost_and_grad,
)


def load_baseline_crab(base_dir: str | Path = "outputs/_baseline_crab") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load CRAB baseline arrays (controls, bases, policy metadata)."""
    base = Path(base_dir)
    with np.load(base / "arrays.npz", allow_pickle=True) as arrs_file:
        arrays = {k: np.asarray(arrs_file[k]) for k in arrs_file.files}
    with open(base / "metadata.json", "r", encoding="utf-8") as f:
        policy = json.load(f)

    arrays["t"] = arrays["t"].astype(float)
    arrays["dt"] = np.asarray(arrays["dt"], dtype=float)
    arrays["T"] = np.asarray(arrays["T"], dtype=float)
    arrays["Omega0"] = arrays["Omega0"].astype(float)
    arrays["Delta0"] = arrays["Delta0"].astype(float)
    arrays["CRAB_BASIS_OMEGA"] = arrays["CRAB_BASIS_OMEGA"].astype(float)
    arrays["CRAB_BASIS_DELTA"] = arrays["CRAB_BASIS_DELTA"].astype(float)
    arrays["CRAB_MODES_OMEGA"] = arrays.get(
        "CRAB_MODES_OMEGA",
        np.arange(arrays["CRAB_BASIS_OMEGA"].shape[1], dtype=int),
    ).astype(int)
    arrays["CRAB_MODES_DELTA"] = arrays.get(
        "CRAB_MODES_DELTA",
        np.arange(arrays["CRAB_BASIS_DELTA"].shape[1], dtype=int),
    ).astype(int)
    arrays["rho0"] = arrays["rho0"].astype(np.complex128)
    arrays["target"] = arrays["target"].astype(np.complex128)
    arrays["Nt"] = np.asarray(arrays["Nt"], dtype=int)
    arrays["SEED"] = np.asarray(arrays["SEED"], dtype=int)

    t = arrays["t"]
    nt = t.shape[0]
    if arrays["Omega0"].shape[0] != nt or arrays["Delta0"].shape[0] != nt:
        raise ValueError("Baseline controls must share the same time grid length as t.")

    return arrays, policy


def sine_basis(t: np.ndarray, modes: np.ndarray, T: float) -> np.ndarray:
    """Return sine basis matrix with zero-valued endpoints for CRAB."""
    t = np.asarray(t, dtype=float)
    modes = np.asarray(modes, dtype=float)
    phase = np.pi * np.outer(t / T, modes)
    return np.sin(phase)


def normalize_basis_columns(basis: np.ndarray, dt: float) -> np.ndarray:
    """Normalize columns to unit L2 norm under the discrete time grid."""
    basis = np.asarray(basis, dtype=float)
    scales = np.sqrt(np.sum(basis * basis, axis=0) * dt)
    scales[scales == 0.0] = 1.0
    return basis / scales


def build_crab_bases(
    t: np.ndarray,
    dt: float,
    T: float,
    modes_Omega: np.ndarray,
    modes_Delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct normalized CRAB bases using the dynamics time grid.

    Ensures the same sampling grid (t, dt, T) is used for both dynamics and
    basis construction. Raises ValueError if inconsistencies are detected.
    """
    t = np.asarray(t, dtype=float)
    dt = float(np.asarray(dt, dtype=float))
    T = float(np.asarray(T, dtype=float))
    nt = t.shape[0]
    if nt < 2:
        raise ValueError("Time grid must contain at least two samples.")
    grid_span = t[0] + dt * (nt - 1)
    if np.isclose(t[0], 0.0) and np.isclose(t[-1], T):
        T_basis = T
    elif np.isclose(grid_span, T):
        T_basis = T
    else:
        raise ValueError("CRAB basis requires the same sampling grid as dynamics.")

    basis_Omega = normalize_basis_columns(sine_basis(t, modes_Omega, T_basis), dt)
    basis_Delta = normalize_basis_columns(sine_basis(t, modes_Delta, T_basis), dt)
    if basis_Omega.shape[0] != nt or basis_Delta.shape[0] != nt:
        raise ValueError("CRAB bases must have one row per time sample.")
    return basis_Omega, basis_Delta


def controls_from_coeffs(
    coeffs_Omega: np.ndarray,
    coeffs_Delta: np.ndarray,
    Omega0: np.ndarray,
    Delta0: np.ndarray,
    basis_Omega: np.ndarray,
    basis_Delta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble CRAB controls from baseline envelopes and coefficient vectors."""
    Omega = Omega0 + basis_Omega @ coeffs_Omega
    Delta = Delta0 + basis_Delta @ coeffs_Delta
    return Omega, Delta


def terminal_cost_and_grad_crab(
    coeffs_Omega: np.ndarray,
    coeffs_Delta: np.ndarray,
    Omega0: np.ndarray,
    Delta0: np.ndarray,
    basis_Omega: np.ndarray,
    basis_Delta: np.ndarray,
    rho0: np.ndarray,
    dt: float,
    target: np.ndarray,
    *,
    power_weight: float = 0.0,
    neg_kappa: float = 10.0,
    neg_weight: float = 0.0,
    **kwargs: Any,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return terminal cost, gradients w.r.t. CRAB coefficients, and assembled controls."""
    if basis_Omega.shape[0] != Omega0.shape[0] or basis_Delta.shape[0] != Delta0.shape[0]:
        raise ValueError("CRAB bases must share the same number of rows as the baseline envelopes.")

    Omega, Delta = controls_from_coeffs(
        coeffs_Omega,
        coeffs_Delta,
        Omega0,
        Delta0,
        basis_Omega,
        basis_Delta,
    )
    cost, gO_time, gD_time = terminal_cost_and_grad(
        Omega,
        Delta,
        rho0,
        dt,
        target,
        power_weight=power_weight,
        neg_kappa=neg_kappa,
        neg_weight=neg_weight,
        **kwargs,
    )
    gO_coeff = basis_Omega.T @ gO_time
    gD_coeff = basis_Delta.T @ gD_time
    return cost, gO_coeff, gD_coeff, Omega, Delta


__all__ = [
    "load_baseline_crab",
    "sine_basis",
    "normalize_basis_columns",
    "build_crab_bases",
    "controls_from_coeffs",
    "terminal_cost_and_grad_crab",
    "terminal_cost",
]
