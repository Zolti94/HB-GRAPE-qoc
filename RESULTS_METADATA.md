# Results Chapter Metadata

## Simulation

| Field | Value |
| --- | --- |
| Solver | custom SU(2) propagator `propagate_piecewise_const` (analytic SU(2) steps)【F:src/physics.py†L127-L194】 |
| Total duration | 0.1 µs over 2000 samples (`dt` ≈ 5.00×10⁻⁵ µs)【F:notebooks/12_costfunction_types.ipynb†L76-L101】【F:src/baselines/grape_coefficients.py†L66-L82】 |
| Initial / target states | ρ₀ = |0⟩⟨0|, target = |1⟩⟨1| (Hilbert dimension 2)【F:notebooks/12_costfunction_types.ipynb†L87-L103】 |
| Frequency units | All controls expressed in radians per microsecond (repository convention)【F:src/baselines/grape_coefficients.py†L1-L6】 |
| Differentiation | Adjoint-state gradient with commutator evaluation for Ω/Δ【F:src/cost.py†L45-L127】 |

## Controls and Basis

| Field | Value |
| --- | --- |
| Harmonic basis sizes | KΩ = 8, KΔ = 4【F:notebooks/12_costfunction_types.ipynb†L76-L113】 |
| Pulse formulas | s = 2·(t−t₀)/duration − 1; Ω(s) = 1 − 3s⁴ + 2s⁶; Δ(s) = 1.25s − 0.25s³【F:src/baselines/grape_coefficients.py†L302-L327】 |
| Scaling | Ω area normalised to 4π; Δ uses zero-area spec with amplitude_scale = 150 rad/µs (≈23.87 MHz)【F:notebooks/12_costfunction_types.ipynb†L80-L109】【F:src/baselines/grape_coefficients.py†L332-L348】 |

## Objectives and Penalties

| Field | Value |
| --- | --- |
| Terminal cost (C₁) | `accumulate_cost_and_grads` terminal infidelity with penalties【F:src/cost.py†L78-L127】 |
| Path cost (C₂) | Ground-state projector tracking objective with adjoint back-propagation【F:src/optimizers/objectives.py†L101-L208】 |
| Ensemble cost (C₃) | Mean terminal infidelity over β and detuning grids【F:src/optimizers/objectives.py†L210-L352】 |
| Penalty weights | power_weight = 1e⁻⁵, neg_weight = 1e⁻⁸, κ = 10【F:notebooks/12_costfunction_types.ipynb†L105-L111】 |
| Optimizer budget | max_iters = 2000, grad_tol = 1e⁻¹⁰, rtol = 1e⁻¹⁰ (Stage 1 Adam)【F:notebooks/12_costfunction_types.ipynb†L94-L109】 |

## HB-GRAPE Optimizer

| Field | Value |
| --- | --- |
| Method | Adam with learning_rate = 0.02, β₁ = 0.9, β₂ = 0.999, ε = 1e⁻⁸, lr_decay = 1.0【F:notebooks/12_costfunction_types.ipynb†L94-L109】 |
| Projection | Time-domain gradients projected with harmonic basis transposes (Bᵀg)【F:src/optimizers/problem.py†L96-L117】 |

## CMA-ES Baseline Sweep

| Field | Value |
| --- | --- |
| Budget / seed | 2000 evaluations per objective, SEED = 42 (global RNG + per-optimizer RNG)【F:notebooks/13_optimize_baseline_from_grape.ipynb†L98-L113】【F:notebooks/13_optimize_baseline_from_grape.ipynb†L279-L288】 |
| Penalties & constraints | AREA_TARGET = 4π with quadratic overflow penalty; ensemble objective reuses β/Δ grids【F:notebooks/13_optimize_baseline_from_grape.ipynb†L98-L109】【F:notebooks/13_optimize_baseline_from_grape.ipynb†L197-L266】 |
| Basis | Shared sine basis matrices with KΩ = 8, KΔ = 4 over 0.1 µs window【F:notebooks/13_optimize_baseline_from_grape.ipynb†L53-L109】 |

## Ensemble & Robustness Grids

| Field | Value |
| --- | --- |
| HB ensemble grids | β ∈ linspace(0.9, 1.1, 5); detuning shifts ±0.1·max|Δ| (5 samples)【F:src/optimizers/problem.py†L204-L214】【F:src/optimizers/objectives.py†L210-L352】 |
| Robustness heatmap | Area grid linspace(0.1, 8.0, 40); detuning shifts linspace(−140, 140, 41); metric = log₁₀(max(1−F, 1e−12))【F:notebooks/12_costfunction_types.ipynb†L722-L776】 |
| Canonical plotting grid | 0.1 µs duration, 2000-point canonical timeline for figure resampling and adiabaticity metric【F:notebooks/figures_for_results.ipynb†L54-L56】【F:notebooks/figures_for_results.ipynb†L1442-L1446】 |

## Reproducibility Notes

- Global random seed 42 used for CMA-ES experiments (`np.random.seed` and `RandomState`).【F:notebooks/13_optimize_baseline_from_grape.ipynb†L111-L288】
- Stage 1 HB-GRAPE notebooks reuse deterministic baseline preparation via `prepare_baseline` (same time grid / shapes).【F:notebooks/12_costfunction_types.ipynb†L161-L199】
- See `RESULTS_METADATA.yaml` / `.json` for structured data and citation index.

### HOWTO: capture library versions

```python
import qutip, numpy, scipy, matplotlib, qiskit_ibm_runtime as qir
print("qutip", qutip.__version__)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("matplotlib", matplotlib.__version__)
try:
    import qiskit
    print("qiskit", qiskit.__version__)
    print("qiskit-ibm-runtime", qir.__version__)
except Exception as exc:
    print("qiskit not available:", exc)
```
