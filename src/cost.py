"""Cost functionals and gradients in microseconds (us) and rad/us."""
from __future__ import annotations

from typing import Callable, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .physics import adjoint_steps, fidelity_pure

NDArrayFloat = npt.NDArray[np.float64]
NDArrayComplex = npt.NDArray[np.complex128]
CostDict = dict[str, float]
GradDict = dict[str, NDArrayFloat]

__all__ = [
    "terminal_infidelity",
    "path_infidelity",
    "ensemble_expectation",
    "power_fluence_penalty",
    "negativity_smooth_penalty",
    "penalty_terms",
    "total_cost",
    "grad_terminal_wrt_controls",
    "grad_power_fluence",
    "grad_negativity_smooth",
    "accumulate_cost_and_grads",
    "check_time_consistency",
    "finite_and_real",
]

SIGMA_X: NDArrayComplex = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Z: NDArrayComplex = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def terminal_infidelity(psi_T: NDArrayComplex, psi_target: NDArrayComplex) -> CostDict:
    """Return terminal infidelity cost = 1 - |<psi_target|psi_T>|^2."""

    fidelity = fidelity_pure(psi_T, psi_target)
    return {"terminal": float(1.0 - fidelity)}


def path_infidelity(
    psi_path: NDArrayComplex,
    psi_ref_path: NDArrayComplex | None = None,
    weights: NDArrayFloat | None = None,
) -> CostDict:
    """Return path infidelity along a trajectory (defaults to zero without reference)."""

    psi_path = np.asarray(psi_path, dtype=np.complex128)
    if psi_path.ndim != 2 or psi_path.shape[1] != 2:
        raise ValueError("psi_path must have shape (T, 2) or (T + 1, 2).")
    if psi_ref_path is None:
        return {"path": 0.0}
    psi_ref = np.asarray(psi_ref_path, dtype=np.complex128)
    if psi_ref.shape != psi_path.shape:
        raise ValueError("psi_ref_path must match psi_path shape.")
    overlaps = np.einsum("ti,ti->t", psi_ref.conj(), psi_path)
    fidelities = np.abs(overlaps) ** 2
    weights_arr = (
        np.asarray(weights, dtype=np.float64)
        if weights is not None
        else np.ones_like(fidelities, dtype=np.float64)
    )
    if weights_arr.shape != fidelities.shape:
        raise ValueError("weights must match the number of time samples.")
    total_weight = float(np.sum(weights_arr))
    if total_weight <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    weights_arr = weights_arr / total_weight
    cost = float(np.sum(weights_arr * (1.0 - fidelities)))
    return {"path": cost}


def ensemble_expectation(
    cost_fn: Callable[[Mapping[str, np.ndarray]], CostDict],
    samples: Sequence[Mapping[str, np.ndarray]],
) -> CostDict:
    """Return the average cost dictionary across an ensemble of samples."""

    if not samples:
        raise ValueError("samples must not be empty.")
    accum: dict[str, float] = {}
    for sample in samples:
        term = cost_fn(sample)
        for key, value in term.items():
            accum[key] = accum.get(key, 0.0) + float(value)
    scale = float(len(samples))
    for key in accum:
        accum[key] /= scale
    return accum


def power_fluence_penalty(
    omega: NDArrayFloat,
    delta: NDArrayFloat,
    dt_us: float,
    w_power: float,
) -> CostDict:
    """Return fluence penalty w_power * dt * sum(omega^2 + delta^2)."""

    omega = np.asarray(omega, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    dt = float(dt_us)
    fluence = dt * float(np.sum(omega * omega + delta * delta, dtype=np.float64))
    return {"power_penalty": float(w_power) * fluence}


def negativity_smooth_penalty(
    omega: NDArrayFloat,
    w_neg: float,
    epsilon: float = 1e-6,
) -> CostDict:
    """Return smooth softplus-based penalty enforcing omega >= 0."""

    omega = np.asarray(omega, dtype=np.float64)
    eps = float(epsilon)
    z = omega / eps
    soft = eps * np.logaddexp(0.0, -z)
    penalty = float(w_neg) * float(np.sum(soft * soft, dtype=np.float64))
    return {"neg_penalty": penalty}



def penalty_terms(
    omega: NDArrayFloat,
    dt_us: float,
    *,
    power_weight: float,
    neg_weight: float,
    neg_kappa: float,
) -> tuple[float, float, NDArrayFloat]:
    """Return penalty contributions and gradient with respect to omega."""

    omega = np.asarray(omega, dtype=np.float64)
    dt = float(dt_us)
    grad = np.zeros_like(omega, dtype=np.float64)
    power_penalty = 0.0
    if power_weight != 0.0:
        scale = float(power_weight)
        power_penalty = 0.5 * scale * float(np.sum(omega * omega, dtype=np.float64) * dt)
        grad += scale * omega * dt
    neg_penalty = 0.0
    if neg_weight != 0.0:
        kappa = float(neg_kappa)
        if kappa <= 0.0:
            raise ValueError("neg_kappa must be positive.")
        z = np.clip(omega / kappa, -60.0, 60.0)
        soft = kappa * np.logaddexp(0.0, -z)
        neg_penalty = float(neg_weight) * float(np.sum(soft * soft, dtype=np.float64))
        sigma = 1.0 / (1.0 + np.exp(z))
        grad += -2.0 * float(neg_weight) * soft * sigma
    return power_penalty, neg_penalty, grad

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


def grad_power_fluence(
    omega: NDArrayFloat,
    delta: NDArrayFloat,
    dt_us: float,
    w_power: float,
) -> GradDict:
    """Return gradients of the fluence penalty with respect to omega and delta."""

    dt = float(dt_us)
    scale = 2.0 * float(w_power) * dt
    return {
        "dJ/dOmega": scale * np.asarray(omega, dtype=np.float64),
        "dJ/dDelta": scale * np.asarray(delta, dtype=np.float64),
    }


def grad_negativity_smooth(
    omega: NDArrayFloat,
    w_neg: float,
    epsilon: float = 1e-6,
) -> GradDict:
    """Return gradients of the smooth non-negativity penalty."""

    omega = np.asarray(omega, dtype=np.float64)
    eps = float(epsilon)
    z = omega / eps
    soft = eps * np.logaddexp(0.0, -z)
    sigma = 1.0 / (1.0 + np.exp(z))
    grad = -2.0 * float(w_neg) * soft * sigma
    return {"dJ/dOmega": grad, "dJ/dDelta": np.zeros_like(omega)}


def accumulate_cost_and_grads(
    omega: NDArrayFloat,
    delta: NDArrayFloat,
    dt_us: float,
    *,
    psi0: NDArrayComplex,
    psi_target: NDArrayComplex,
    w_power: float = 0.0,
    w_neg: float = 0.0,
    neg_epsilon: float = 1e-6,
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
    grad_terms = [grad_terminal_wrt_controls(omega, delta, dt_us, psi0, psi_target, caches)]

    if w_power != 0.0:
        cost_terms.append(power_fluence_penalty(omega, delta, dt_us, w_power))
        grad_terms.append(grad_power_fluence(omega, delta, dt_us, w_power))
    if w_neg != 0.0:
        cost_terms.append(negativity_smooth_penalty(omega, w_neg, epsilon=neg_epsilon))
        grad_terms.append(grad_negativity_smooth(omega, w_neg, epsilon=neg_epsilon))

    combined_cost = total_cost(cost_terms)

    total_grad_omega = np.zeros_like(omega)
    total_grad_delta = np.zeros_like(delta)
    for grad in grad_terms:
        if "dJ/dOmega" in grad:
            total_grad_omega += np.asarray(grad["dJ/dOmega"], dtype=np.float64)
        if "dJ/dDelta" in grad:
            total_grad_delta += np.asarray(grad["dJ/dDelta"], dtype=np.float64)

    grad_dict = {"dJ/dOmega": total_grad_omega, "dJ/dDelta": total_grad_delta}
    return combined_cost, grad_dict


def check_time_consistency(t: NDArrayFloat, omega: NDArrayFloat, delta: NDArrayFloat) -> None:
    """Ensure time grid and controls share compatible lengths."""

    t = np.asarray(t, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if t.size not in {omega.size, omega.size + 1}:
        raise ValueError("time grid must have length equal to controls or controls + 1.")
    if omega.shape != delta.shape:
        raise ValueError("omega and delta must share shape.")


def finite_and_real(*arrays: np.ndarray) -> None:
    """Validate that all provided arrays contain finite real entries."""

    for arr in arrays:
        arr_np = np.asarray(arr)
        if not np.isfinite(arr_np).all():
            raise ValueError("Arrays must contain only finite values.")
        if np.iscomplexobj(arr_np) and np.any(np.abs(arr_np.imag) > 0.0):
            raise ValueError("Arrays must be real-valued.")
