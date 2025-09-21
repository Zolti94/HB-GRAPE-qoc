"""Regenerate the tracked CRAB baseline dataset."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.qoc_common_crab import DEFAULT_CRAB_BASELINE_DIR, build_crab_bases


def rap_seed(t: np.ndarray, T: float, *, area: float = 4 * np.pi, delta_scale: float = 0.55) -> tuple[np.ndarray, np.ndarray, float]:
    """Return baseline envelopes used throughout the CRAB workflows."""
    s = 2 * (t / T) - 1.0
    shape = 1 - 3 * s**4 + 2 * s**6
    norm = area / (np.trapz(shape, t) + 1e-15)
    omega = norm * shape
    delta = norm * delta_scale * (1.25 * s - 0.25 * s**3)
    return omega.astype(float), delta.astype(float), float(norm)


def generate_baseline(*, Nt: int = 201, T: float = 0.1, seed: int = 42) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Assemble the reference baseline arrays and metadata."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, Nt)
    dt = float(t[1] - t[0])
    omega0, delta0, norm = rap_seed(t, T)
    modes_omega = np.arange(1, 4, dtype=int)
    modes_delta = np.arange(1, 4, dtype=int)
    basis_omega, basis_delta = build_crab_bases(t, dt, T, modes_omega, modes_delta)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    target = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    arrays = {
        "t": t,
        "dt": np.asarray(dt),
        "T": np.asarray(T),
        "Nt": np.asarray(Nt),
        "Omega0": omega0,
        "Delta0": delta0,
        "CRAB_BASIS_OMEGA": basis_omega,
        "CRAB_BASIS_DELTA": basis_delta,
        "CRAB_MODES_OMEGA": modes_omega,
        "CRAB_MODES_DELTA": modes_delta,
        "rho0": rho0,
        "target": target,
        "NORM": np.asarray(norm),
        "SEED": np.asarray(seed),
    }
    fluence_omega = float(np.trapz(omega0**2, t))
    fluence_delta = float(np.trapz(delta0**2, t))
    policy = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "units_stored": {"time": "us", "amplitude": "rad/us"},
        "summary": {
            "Nt": Nt,
            "dt_ns": dt * 1e3,
            "T_us": T,
            "omega_fluence": fluence_omega,
            "delta_fluence": fluence_delta,
            "num_modes_omega": int(len(modes_omega)),
            "num_modes_delta": int(len(modes_delta)),
        },
        "penalties": {
            "power_weight": 0.00005,
            "neg_weight": 1.0,
            "neg_kappa": 1.0,
        },
        "seed": seed,
        "note": "CRAB baseline with sine basis (zero endpoints); arrays saved in SI units.",
    }
    return arrays, policy


def write_baseline(arrays: dict[str, np.ndarray], policy: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "arrays.npz", **arrays)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_CRAB_BASELINE_DIR, help="Output directory for baseline data")
    parser.add_argument("--Nt", type=int, default=201, help="Number of time samples")
    parser.add_argument("--T", type=float, default=0.1, help="Total duration in microseconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    arrays, policy = generate_baseline(Nt=args.Nt, T=args.T, seed=args.seed)
    write_baseline(arrays, policy, args.out)
    print(f"[baseline saved] {args.out / 'arrays.npz'}")


if __name__ == "__main__":
    main()
