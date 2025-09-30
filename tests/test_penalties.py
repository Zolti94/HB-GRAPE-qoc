from __future__ import annotations

import json
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

from src.baselines import (
    BasisSpec,
    GrapeBaselineConfig,
    PulseShapeSpec,
    TimeGridSpec,
    build_grape_baseline,
)
from src.qoc_common import penalty_terms, terminal_cost, terminal_cost_and_grad
from src.qoc_common_crab import (
    build_normalized_harmonic_bases,
    terminal_cost_and_grad_coeffs,
)


def default_baseline_config() -> GrapeBaselineConfig:
    return GrapeBaselineConfig(
        time_grid=TimeGridSpec(duration_us=0.1, num_points=201),
        omega=PulseShapeSpec(kind="polynomial", area_pi=1.0),
        delta=PulseShapeSpec(kind="linear_chirp", area_pi=0.0, amplitude_scale=0.5),
        basis=BasisSpec(num_omega=4, num_delta=4),
        extra_metadata={"label": "unit_test"},
    )


def build_default_baseline() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    return build_grape_baseline(default_baseline_config())


def test_terminal_grad_fd() -> None:
    rng = np.random.default_rng(123)
    Nt = 64
    T = 0.01
    t = np.linspace(0.0, T, Nt)
    dt = float(t[1] - t[0])
    omega = rng.normal(size=Nt)
    delta = rng.normal(size=Nt)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    target = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    weights = {"power_weight": 2e-3, "neg_weight": 1e-3, "neg_kappa": 9.0}
    cost, gO, gD = terminal_cost_and_grad(
        omega,
        delta,
        rho0,
        dt,
        target,
        **weights,
    )
    eye = np.eye(Nt)
    eps = 1e-7
    max_err = 0.0
    for idx in range(Nt):
        direction = eye[idx]
        cost_plus = terminal_cost(omega + eps * direction, delta, rho0, dt, target, **weights)
        cost_minus = terminal_cost(omega - eps * direction, delta, rho0, dt, target, **weights)
        fd = (cost_plus - cost_minus) / (2 * eps)
        max_err = max(max_err, abs(fd - gO[idx]))
    for idx in range(Nt):
        direction = eye[idx]
        cost_plus = terminal_cost(omega, delta + eps * direction, rho0, dt, target, **weights)
        cost_minus = terminal_cost(omega, delta - eps * direction, rho0, dt, target, **weights)
        fd = (cost_plus - cost_minus) / (2 * eps)
        max_err = max(max_err, abs(fd - gD[idx]))
    assert max_err < 1e-6


def test_projection_gradients() -> None:
    rng = np.random.default_rng(456)
    Nt = 64
    T = 0.01
    t = np.linspace(0.0, T, Nt)
    dt = float(t[1] - t[0])
    modes_omega = np.arange(1, 5)
    modes_delta = np.arange(1, 4)
    basis_omega, basis_delta = build_normalized_harmonic_bases(t, dt, T, modes_omega, modes_delta)
    omega0 = 0.05 * rng.normal(size=Nt)
    delta0 = 0.05 * rng.normal(size=Nt)
    coeffs_omega = 0.1 * rng.normal(size=basis_omega.shape[1])
    coeffs_delta = 0.1 * rng.normal(size=basis_delta.shape[1])
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    target = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    cost, gO_coeff, gD_coeff, omega, delta = terminal_cost_and_grad_coeffs(
        coeffs_omega,
        coeffs_delta,
        omega0,
        delta0,
        basis_omega,
        basis_delta,
        rho0,
        dt,
        target,
    )
    assert np.isfinite(cost)
    assert omega.shape == (Nt,)
    assert delta.shape == (Nt,)
    assert np.isfinite(gO_coeff).all()
    assert np.isfinite(gD_coeff).all()


def test_penalty_terms_consistency() -> None:
    rng = np.random.default_rng(654)
    Nt = 128
    T = 0.02
    t = np.linspace(0.0, T, Nt)
    dt = float(t[1] - t[0])
    omega = rng.normal(size=Nt)
    delta = rng.normal(size=Nt)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    target = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    weights = {"power_weight": 1e-3, "neg_weight": 2e-3, "neg_kappa": 7.5}
    cost_plain, gO_plain, _ = terminal_cost_and_grad(omega, delta, rho0, dt, target)
    cost_pen, gO_pen, _ = terminal_cost_and_grad(omega, delta, rho0, dt, target, **weights)
    power_penalty, neg_penalty, grad_penalty = penalty_terms(omega, dt, **weights)
    assert np.isclose(cost_pen - cost_plain, power_penalty + neg_penalty)
    assert np.allclose(gO_pen - gO_plain, grad_penalty)


def test_negativity_penalty_behavior() -> None:
    Nt = 32
    dt = 0.05
    rng = np.random.default_rng(789)
    omega = rng.normal(size=Nt) - 1.0
    delta = np.zeros_like(omega)
    rho0 = 0.5 * np.eye(2, dtype=np.complex128)
    target = rho0.copy()
    neg_weight = 5e-2
    neg_kappa = 10.0
    cost_base = terminal_cost(omega, delta, rho0, dt, target, neg_weight=neg_weight, neg_kappa=neg_kappa)
    omega_more_negative = omega.copy()
    omega_more_negative[omega_more_negative < 0.0] *= 2.0
    cost_harder = terminal_cost(omega_more_negative, delta, rho0, dt, target, neg_weight=neg_weight, neg_kappa=neg_kappa)
    assert cost_harder > cost_base
    _, gO, _ = terminal_cost_and_grad(omega, delta, rho0, dt, target, neg_weight=neg_weight, neg_kappa=neg_kappa)
    omega_next = omega - 0.2 * gO
    assert float(np.quantile(omega_next, 0.05)) >= float(np.quantile(omega, 0.05))


def test_no_legacy_update_symbols() -> None:
    banned = ("lowpass_update", "apply_shared_update", "pin_endpoints", "cutoff_MHz")
    search_roots = (Path("src"), Path("notebooks"))
    this_file = Path(__file__).resolve()
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.rglob('*'):
            if candidate.resolve() == this_file:
                continue
            if not candidate.is_file() or candidate.suffix not in {'.py', '.ipynb'}:
                continue
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            for token in banned:
                assert token not in text, f"Found legacy token '{token}' in {candidate}"


def test_baseline_grid_consistency() -> None:
    arrays, _ = build_default_baseline()
    t = arrays["t"]
    dt = float(arrays["dt"])
    duration = float(arrays["T"])
    modes_omega = arrays["CRAB_MODES_OMEGA"]
    modes_delta = arrays["CRAB_MODES_DELTA"]
    basis_omega = arrays["CRAB_BASIS_OMEGA"]
    basis_delta = arrays["CRAB_BASIS_DELTA"]
    B_omega, B_delta = build_normalized_harmonic_bases(t, dt, duration, modes_omega, modes_delta)
    assert np.allclose(basis_omega, B_omega)
    assert np.allclose(basis_delta, B_delta)


if __name__ == "__main__":
    results = {}
    for name, func in globals().items():
        if name.startswith("test_") and callable(func):
            func()
            results[name] = "ok"
    print(json.dumps(results, indent=2))
