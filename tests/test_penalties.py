from __future__ import annotations

import json
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
CRAB_DIR = THIS_DIR.parent
sys.path.insert(0, str(CRAB_DIR))

import numpy as np

from qoc_common import terminal_cost, terminal_cost_and_grad
from qoc_common_crab import build_crab_bases, terminal_cost_and_grad_crab


def test_terminal_grad_fd() -> None:
    rng = np.random.default_rng(123)
    Nt = 64
    T = 0.01
    t = np.linspace(0.0, T, Nt)
    dt = float(t[1] - t[0])
    Omega = rng.normal(size=Nt)
    Delta = rng.normal(size=Nt)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    target = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    power_weight = 2e-3
    neg_weight = 1e-3
    neg_kappa = 9.0
    cost, gO, gD = terminal_cost_and_grad(
        Omega,
        Delta,
        rho0,
        dt,
        target,
        power_weight=power_weight,
        neg_weight=neg_weight,
        neg_kappa=neg_kappa,
    )
    eps = 1e-7
    max_err = 0.0
    for idx in range(Nt):
        direction = np.zeros_like(Omega)
        direction[idx] = 1.0
        cost_plus = terminal_cost(
            Omega + eps * direction,
            Delta,
            rho0,
            dt,
            target,
            power_weight=power_weight,
            neg_weight=neg_weight,
            neg_kappa=neg_kappa,
        )
        cost_minus = terminal_cost(
            Omega - eps * direction,
            Delta,
            rho0,
            dt,
            target,
            power_weight=power_weight,
            neg_weight=neg_weight,
            neg_kappa=neg_kappa,
        )
        fd = (cost_plus - cost_minus) / (2 * eps)
        max_err = max(max_err, abs(fd - gO[idx]))
    for idx in range(Nt):
        direction = np.zeros_like(Delta)
        direction[idx] = 1.0
        cost_plus = terminal_cost(
            Omega,
            Delta + eps * direction,
            rho0,
            dt,
            target,
            power_weight=power_weight,
            neg_weight=neg_weight,
            neg_kappa=neg_kappa,
        )
        cost_minus = terminal_cost(
            Omega,
            Delta - eps * direction,
            rho0,
            dt,
            target,
            power_weight=power_weight,
            neg_weight=neg_weight,
            neg_kappa=neg_kappa,
        )
        fd = (cost_plus - cost_minus) / (2 * eps)
        max_err = max(max_err, abs(fd - gD[idx]))
    assert max_err < 1e-6, f"Max gradient error {max_err:.3e} exceeds tolerance"


def test_crab_projection_gradients() -> None:
    rng = np.random.default_rng(456)
    Nt = 64
    T = 0.01
    t = np.linspace(0.0, T, Nt)
    dt = float(t[1] - t[0])
    modes_omega = np.arange(1, 5)
    modes_delta = np.arange(1, 4)
    basis_omega, basis_delta = build_crab_bases(t, dt, T, modes_omega, modes_delta)
    Omega0 = 0.05 * rng.normal(size=Nt)
    Delta0 = 0.05 * rng.normal(size=Nt)
    coeffs_Omega = 0.1 * rng.normal(size=basis_omega.shape[1])
    coeffs_Delta = 0.1 * rng.normal(size=basis_delta.shape[1])
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    target = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    cost, gO_coeff, gD_coeff, Omega, Delta = terminal_cost_and_grad_crab(
        coeffs_Omega,
        coeffs_Delta,
        Omega0,
        Delta0,
        basis_omega,
        basis_delta,
        rho0,
        dt,
        target,
        power_weight=2e-3,
        neg_weight=1e-3,
    )
    _, gO_time, gD_time = terminal_cost_and_grad(
        Omega,
        Delta,
        rho0,
        dt,
        target,
        power_weight=2e-3,
        neg_weight=1e-3,
    )
    proj_Omega = basis_omega.T @ gO_time
    proj_Delta = basis_delta.T @ gD_time
    assert np.allclose(gO_coeff, proj_Omega, atol=1e-12)
    assert np.allclose(gD_coeff, proj_Delta, atol=1e-12)

def test_negativity_penalty_behavior() -> None:
    Nt = 32
    dt = 0.05
    rng = np.random.default_rng(789)
    Omega = rng.normal(size=Nt) - 1.0
    Delta = np.zeros_like(Omega)
    rho0 = 0.5 * np.eye(2, dtype=np.complex128)
    target = rho0.copy()
    neg_weight = 5e-2
    neg_kappa = 10.0
    cost_base = terminal_cost(
        Omega,
        Delta,
        rho0,
        dt,
        target,
        neg_weight=neg_weight,
        neg_kappa=neg_kappa,
    )
    Omega_more_negative = Omega.copy()
    Omega_more_negative[Omega_more_negative < 0.0] *= 2.0
    cost_harder = terminal_cost(
        Omega_more_negative,
        Delta,
        rho0,
        dt,
        target,
        neg_weight=neg_weight,
        neg_kappa=neg_kappa,
    )
    assert cost_harder > cost_base, "Negativity penalty should increase with larger negative amplitudes"
    _, gO, _ = terminal_cost_and_grad(
        Omega,
        Delta,
        rho0,
        dt,
        target,
        neg_weight=neg_weight,
        neg_kappa=neg_kappa,
    )
    eta = 0.2
    Omega_next = Omega - eta * gO
    neg_quantile_before = float(np.quantile(Omega, 0.05))
    neg_quantile_after = float(np.quantile(Omega_next, 0.05))
    assert neg_quantile_after >= neg_quantile_before, "Gradient step should reduce negative tail"


def test_no_legacy_update_symbols() -> None:
    banned = ("lowpass_update", "apply_shared_update", "pin_endpoints", "cutoff_MHz")
    root = Path("crab")
    this_file = Path(__file__).resolve()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".ipynb"}:
            if path.resolve() == this_file:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in banned:
                assert token not in text, f"Found legacy token '{token}' in {path}"


def test_baseline_grid_consistency() -> None:
    base_dir = Path("crab/outputs/_baseline_crab")
    with np.load(base_dir / "arrays.npz", allow_pickle=True) as data:
        t = data["t"]
        dt = float(data["dt"])
        T = float(data["T"])
        Omega0 = data["Omega0"]
        Delta0 = data["Delta0"]
        basis_omega = data["CRAB_BASIS_OMEGA"]
        basis_delta = data["CRAB_BASIS_DELTA"]
        modes_omega = data["CRAB_MODES_OMEGA"]
        modes_delta = data["CRAB_MODES_DELTA"]
    Nt = t.shape[0]
    assert Omega0.shape[0] == Nt == basis_omega.shape[0] == basis_delta.shape[0]
    assert np.isclose(dt * (Nt - 1), T) or np.isclose(t[-1], T)
    B_omega, B_delta = build_crab_bases(t, dt, T, modes_omega, modes_delta)
    assert np.allclose(basis_omega, B_omega)
    assert np.allclose(basis_delta, B_delta)


def main() -> None:
    tests = [
        ("terminal gradients", test_terminal_grad_fd),
        ("crab projection gradients", test_crab_projection_gradients),
        ("negativity penalty", test_negativity_penalty_behavior),
        ("no legacy symbols", test_no_legacy_update_symbols),
        ("grid consistency", test_baseline_grid_consistency),
    ]
    for name, func in tests:
        func()
        print(f"[ok] {name}")


if __name__ == "__main__":
    main()



