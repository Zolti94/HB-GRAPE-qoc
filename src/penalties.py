"""Shared penalty helpers for control optimizers."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]

__all__ = [
    "compute_penalties",
    "penalty_terms",
]


_DEF_CLIP = 60.0


def compute_penalties(
    omega: NDArrayFloat,
    delta: NDArrayFloat | None,
    dt_us: float,
    *,
    power_weight: float = 0.0,
    neg_weight: float = 0.0,
    neg_kappa: float = 10.0,
) -> Tuple[float, float, NDArrayFloat, NDArrayFloat]:
    """Return penalty values and gradients for omega/delta controls.

    The fluence penalty acts on both control channels; the negativity penalty
    applies only to ``omega``. Both terms share the same definition across all
    optimizers so their initial values match regardless of the active
    objective.
    """

    omega_arr = np.asarray(omega, dtype=np.float64)
    if omega_arr.ndim != 1:
        raise ValueError("omega must be a one-dimensional array")
    if delta is None:
        delta_arr = np.zeros_like(omega_arr)
    else:
        delta_arr = np.asarray(delta, dtype=np.float64)
        if delta_arr.shape != omega_arr.shape:
            raise ValueError("delta must match omega shape")
    dt = float(dt_us)

    grad_omega = np.zeros_like(omega_arr)
    grad_delta = np.zeros_like(delta_arr)

    pen_power = 0.0
    if power_weight != 0.0:
        weight = float(power_weight)
        fluence = dt * float(np.sum(omega_arr * omega_arr + delta_arr * delta_arr, dtype=np.float64))
        pen_power = weight * fluence
        scale = 2.0 * weight * dt
        grad_omega += scale * omega_arr
        grad_delta += scale * delta_arr

    pen_neg = 0.0
    if neg_weight != 0.0:
        kappa = float(neg_kappa)
        if not np.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("neg_kappa must be positive and finite")
        weight = float(neg_weight)
        z = np.clip(omega_arr / kappa, -_DEF_CLIP, _DEF_CLIP)
        soft = kappa * np.logaddexp(0.0, -z)
        pen_neg = weight * float(np.sum(soft * soft, dtype=np.float64))
        sigma = 1.0 / (1.0 + np.exp(z))
        grad_omega += -2.0 * weight * soft * sigma

    return pen_power, pen_neg, grad_omega, grad_delta


def penalty_terms(
    omega: NDArrayFloat,
    dt_us: float,
    *,
    power_weight: float = 0.0,
    neg_weight: float = 0.0,
    neg_kappa: float = 10.0,
) -> Tuple[float, float, NDArrayFloat]:
    """Compatibility wrapper returning only omega gradients.

    Matches the historical interface used by notebook utilities and tests.
    """

    pen_power, pen_neg, grad_omega, _ = compute_penalties(
        omega,
        None,
        dt_us,
        power_weight=power_weight,
        neg_weight=neg_weight,
        neg_kappa=neg_kappa,
    )
    return pen_power, pen_neg, grad_omega
