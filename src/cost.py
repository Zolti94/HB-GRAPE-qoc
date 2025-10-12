"""Cost functionals and gradients in microseconds (us) and rad/us."""
from __future__ import annotations

from typing import Callable, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .penalties import compute_penalties, penalty_terms
from .physics import adjoint_steps, fidelity_pure, SIGMA_X, SIGMA_Z

NDArrayFloat = npt.NDArray[np.float64]
NDArrayComplex = npt.NDArray[np.complex128]
CostDict = dict[str, float]
GradDict = dict[str, NDArrayFloat]

__all__ = [
    "terminal_infidelity",
    "penalty_terms",
    "total_cost",
    "grad_terminal_wrt_controls",
    "accumulate_cost_and_grads",
]

def terminal_infidelity(psi_T: NDArrayComplex, psi_target: NDArrayComplex) -> CostDict:
    """Return terminal infidelity cost = 1 - |<psi_target|psi_T>|^2."""

    fidelity = fidelity_pure(psi_T, psi_target)
    return {"terminal": float(1.0 - fidelity)}


def total_cost(terms: Iterable[CostDict]) -> CostDict:
    """Combine individual term dictionaries and add a "total" entry."""

    combined: dict[str, float] = {}
    for term in terms:
        for key, value in term.items():
            if key == "total":
                continue
            combined[key] = combined.get(key, 0.0) + float(value)
    combined["total"] = float(sum(combined.values()))
    return combined


def grad_terminal_wrt_controls(
    omega: NDArrayFloat,
    delta: NDArrayFloat,
    dt_us: float,
    psi0: NDArrayComplex,
    psi_target: NDArrayComplex,
    caches: MutableMapping[str, np.ndarray],
) -> GradDict:
    """Return gradients of terminal infidelity with respect to omega and delta."""

    forward = np.asarray(caches["forward_rhos"], dtype=np.complex128)
    adjoint = np.asarray(caches["adjoint_lams"], dtype=np.complex128)
    steps = forward.shape[0] - 1
    dt = float(caches.get("dt_us", dt_us))

    dHx = 0.5 * SIGMA_X
    dHz = 0.5 * SIGMA_Z
    g_omega = np.zeros(steps, dtype=np.float64)
    g_delta = np.zeros(steps, dtype=np.float64)

    for k in range(steps):
        rho_k = forward[k]
        lam_next = adjoint[k + 1]
        comm_x = dHx @ rho_k - rho_k @ dHx
        comm_z = dHz @ rho_k - rho_k @ dHz
        g_omega[k] = -np.imag(np.trace(lam_next @ comm_x)) * dt
        g_delta[k] = -np.imag(np.trace(lam_next @ comm_z)) * dt

    return {"dJ/dOmega": g_omega, "dJ/dDelta": g_delta}




def accumulate_cost_and_grads(
    omega: NDArrayFloat,
    delta: NDArrayFloat,
    dt_us: float,
    *,
    psi0: NDArrayComplex,
    psi_target: NDArrayComplex,
    w_power: float = 0.0,
    w_neg: float = 0.0,
    neg_kappa: float = 10.0,
    caches: MutableMapping[str, np.ndarray] | None = None,
) -> Tuple[CostDict, GradDict]:
    """Compute terminal cost plus optional penalties and their gradients."""

    omega = np.asarray(omega, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if caches is None:
        caches = adjoint_steps(omega, delta, dt_us, psi0=psi0, psi_target=psi_target)
    else:
        caches = dict(caches)
        caches.setdefault("dt_us", dt_us)

    forward_rhos = np.asarray(caches["forward_rhos"], dtype=np.complex128)
    rho_T = forward_rhos[-1]
    target_rho = np.outer(psi_target, psi_target.conj())
    term_cost = 1.0 - float(np.real(np.trace(rho_T @ target_rho)))

    cost_terms = [{"terminal": term_cost}]
    grad_time = grad_terminal_wrt_controls(omega, delta, dt_us, psi0, psi_target, caches)

    pen_power, pen_neg, grad_pen_omega, grad_pen_delta = compute_penalties(
        omega,
        delta,
        dt_us,
        power_weight=w_power,
        neg_weight=w_neg,
        neg_kappa=neg_kappa,
    )

    if w_power != 0.0:
        cost_terms.append({"power_penalty": pen_power})
    if w_neg != 0.0:
        cost_terms.append({"neg_penalty": pen_neg})

    combined_cost = total_cost(cost_terms)

    total_grad_omega = np.asarray(grad_time["dJ/dOmega"], dtype=np.float64) + grad_pen_omega
    total_grad_delta = np.asarray(grad_time["dJ/dDelta"], dtype=np.float64) + grad_pen_delta

    grad_dict = {"dJ/dOmega": total_grad_omega, "dJ/dDelta": total_grad_delta}
    return combined_cost, grad_dict




