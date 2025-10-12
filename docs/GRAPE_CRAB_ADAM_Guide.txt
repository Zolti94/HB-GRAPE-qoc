# GRAPE + CRAB + Adam Guide

This note summarizes the settings used across the GRAPE/CRAB notebooks and mirrors the PDF version shipped with the project. It collects the core equations, penalty terms, and optimisation settings so they can be viewed without a PDF reader.

## 1. System model and propagation
- Two-level Hamiltonian
  
  \[ H_k = \frac{1}{2}\big(\Delta_k\,\sigma_z + \Omega_k\,\sigma_x\big). \]
- Controls are sampled on a uniform grid of \(N_t\) points with spacing \(\Delta t\). The propagator for slice \(k\) is
  
  \[ U_k = \exp\left(-i H_k\,\Delta t\right),\qquad \rho_{k+1} = U_k \rho_k U_k^\dagger. \]
- Sensitivities used by GRAPE:
  
  \[ \partial H / \partial \Omega = \tfrac{1}{2}\sigma_x,\qquad \partial H / \partial \Delta = \tfrac{1}{2}\sigma_z. \]

## 2. Optimisation objectives
- **Terminal fidelity**: maximise \(F = \operatorname{Re}\operatorname{Tr}(\rho_T\,\rho_{\text{target}})\); the loss function is \(1 - F\).
- **Ensemble fidelity**: average the terminal loss over Doppler and amplitude grids.
- **Adiabatic tracking**: encourage the state to remain in the instantaneous ground state by minimising the projector distance along the trajectory.

## 3. Regularisation penalties
Let \(\Omega\) denote the discretised control and \(\Delta t\) the time step.
- **Fluence penalty** (weight `power_weight`):
  
  \[ J_{\text{power}} = \frac{w_p}{2} \sum_k \Omega_k^2\,\Delta t. \]
- **Negativity penalty** (weight `neg_weight`, softplus scale `neg_kappa`):
  
  \[ J_{\text{neg}} = \frac{w_n}{2} \sum_k \text{softplus}(-\kappa\,\Omega_k)^2\,\Delta t. \]
- The shared helper `src.penalties.penalty_terms` returns these penalty values together with the gradient contribution \(\partial J/\partial \Omega\). Both `terminal_cost[_and_grad]` and the CRAB notebooks call this helper.

## 4. CRAB parameterisation
- Express each control as a baseline envelope plus a truncated sine basis:
  
  \[ \Omega(t) = \Omega_0(t) + B_{\Omega}(t) c_\Omega,\qquad \Delta(t) = \Delta_0(t) + B_{\Delta}(t) c_\Delta. \]
- `build_crab_bases` normalises basis columns with respect to \(\Delta t\) and checks grid consistency.
- Baseline envelopes and bases are constructed from `GrapeBaselineConfig` (see `src/baselines/grape_coefficients.py`).

## 5. Adam optimiser settings
- Default hyperparameters: learning rate `8e-2`, `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`.
- Stopping criteria:
  - `max_iters = 1000`
  - `target_cost = 1e-5`
  - `max_oracle_calls = 400000`
- Every mode (terminal / ensemble / adiabatic) stores iterations, wall-clock timing, and gradients in `results/<run_name>/` for reproducibility.

## 6. Reproducibility checklist
1. Configure the baseline parameters directly inside `notebooks/01_CRAB_Quickstart.ipynb` (the notebook calls `src.baselines.build_grape_baseline`).
2. Run `notebooks/20_adam_crab_comparison.ipynb` to synthesise new pulse histories with Adam.
3. Use `notebooks/21_optimization_figures.ipynb` to regenerate publication figures.

For additional detail (plots and derivations) refer to `GRAPE_CRAB_ADAM_Guide.pdf`.
