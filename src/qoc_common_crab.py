"""Coefficient-based GRAPE helpers (harmonic bases, gradients, and assembly)."""
from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

from .baselines import GrapeBaselineConfig, build_grape_baseline
from .controls import (
    harmonic_sine_basis,
    normalize_basis_columns,
    build_normalized_harmonic_bases,
    coeffs_to_control,
)
from .qoc_common import terminal_cost, terminal_cost_and_grad

__all__ = [
    "build_grape_arrays",
    "harmonic_sine_basis",
    "normalize_basis_columns",
    "build_normalized_harmonic_bases",
    "assemble_controls_from_coeffs",
    "terminal_cost_and_grad_coeffs",
    "terminal_cost",
]


def build_grape_arrays(
    config: GrapeBaselineConfig | Mapping[str, Any],
) -> Tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return baseline arrays/metadata by delegating to :func:`build_grape_baseline`.

    Parameters
    ----------
    config : GrapeBaselineConfig or Mapping[str, Any]
        Baseline specification. ``Mapping`` inputs are converted via
        :meth:`GrapeBaselineConfig.from_dict`.
    """

    if not isinstance(config, GrapeBaselineConfig):
        config = GrapeBaselineConfig.from_dict(dict(config))
    return build_grape_baseline(config)
def assemble_controls_from_coeffs(
    coeffs_omega: np.ndarray,
    coeffs_delta: np.ndarray,
    omega_base: np.ndarray,
    delta_base: np.ndarray,
    basis_omega: np.ndarray,
    basis_delta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble controls from baseline envelopes and harmonic coefficients."""

    omega = coeffs_to_control(basis_omega, coeffs_omega, base=omega_base)
    delta = coeffs_to_control(basis_delta, coeffs_delta, base=delta_base)
    return omega, delta


def terminal_cost_and_grad_coeffs(
    coeffs_omega: np.ndarray,
    coeffs_delta: np.ndarray,
    omega_base: np.ndarray,
    delta_base: np.ndarray,
    basis_omega: np.ndarray,
    basis_delta: np.ndarray,
    rho0: np.ndarray,
    dt_us: float,
    target: np.ndarray,
    *,
    power_weight: float = 0.0,
    neg_kappa: float = 10.0,
    neg_weight: float = 0.0,
    **kwargs: Any,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cost, gradients (coefficients), and assembled controls."""

    if basis_omega.shape[0] != omega_base.shape[0] or basis_delta.shape[0] != delta_base.shape[0]:
        raise ValueError("Baseline and basis matrices must share the same number of rows.")

    omega, delta = assemble_controls_from_coeffs(
        coeffs_omega,
        coeffs_delta,
        omega_base,
        delta_base,
        basis_omega,
        basis_delta,
    )
    cost, grad_omega_time, grad_delta_time = terminal_cost_and_grad(
        omega,
        delta,
        rho0,
        dt_us,
        target,
        power_weight=power_weight,
        neg_kappa=neg_kappa,
        neg_weight=neg_weight,
        **kwargs,
    )
    grad_omega_coeff = basis_omega.T @ grad_omega_time
    grad_delta_coeff = basis_delta.T @ grad_delta_time
    return cost, grad_omega_coeff, grad_delta_coeff, omega, delta
