# AGENTS

## Overview
This repository collects optimization notebooks and supporting modules for Harmonic-Basis GRAPE (HB-GRAPE) control experiments. The agents work together to:

- Configure baseline pulse shapes and harmonic bases (`prepare_baseline`, `build_base_config`).
- Launch optimizers (constant-step, line-search, Adam) via `run_experiment` with per-experiment overrides.
- Post-process results (pulse plots, Bloch dynamics, robustness heatmaps) for comparison and reporting.

## Notebook Agents

| Notebook | Purpose | Key Actions |
| --- | --- | --- |
| `10_StepSize_Methods.ipynb` | Step-size comparisons (constant vs line-search vs Adam) | Runs three optimizers on a shared baseline, capturing histories, runtimes, and pulse trajectories. |
| `11_Basis_Capacity_Sweep.ipynb` | Basis-size sweep for Adam | Iterates over user-defined `(K_omega, K_delta)` grids, aggregates metrics, saves pulse snapshots, and produces summary plots. |
| `12_costfunction_types.ipynb` | Objective comparison (terminal, ensemble, path) | Reuses the baseline configuration, runs Adam for each objective, and generates cost-history plots, Bloch dynamics, and robustness heatmaps with per-objective overrides. |

## Configuration Agents

- `LR_DECAY_BY_OBJECTIVE`: optional per-objective Adam decay factors (default `1.0`).
- `POWER_WEIGHT_BY_OBJECTIVE`, `NEG_WEIGHT_BY_OBJECTIVE`: per-objective penalty scalars applied during Stage 1 in notebook 12.
- `runner_ctx` (from `prepare_baseline`): exposes baseline arrays (`Omega0`, `Delta0`, `t_us`, etc.) and initial/target states reused across notebooks.

## Execution Flow

1. **Baseline Preparation** - `prepare_baseline` builds deterministic controls using the shared pulse shapes (`omega_shape`, `delta_shape`) and harmonic counts (`K_OMEGA`, `K_DELTA`).
2. **Experiment Configuration** - `build_base_config` packages the baseline plus penalties and optimizer defaults into an `ExperimentConfig` that notebooks can override.
3. **Optimizer Invocation** - `run_experiment` dispatches to the registered optimizer (e.g., `optimize_const`, `optimize_linesearch`, `optimize_adam`) and persists artifacts.
4. **Post-Processing** - Notebooks extract histories, produce plots, and serialize summaries (CSV/JSON/NPZ) for subsequent analysis.

## Artifact Outputs

Optimizers automatically persist:

- `artifacts/<run-name>/config.json`, `metrics.json`, `history.npz`, `pulses.npz`.
- Notebook-specific exports (e.g., `artifacts/basis_capacity_sweep/summary.csv`).

Users can modify per-agent parameters (learning rate, penalties, baseline shapes), rerun the Stage 1 setup, and regenerate downstream plots and heatmaps to compare HB-GRAPE strategies quickly.
