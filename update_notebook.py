import nbformat
from pathlib import Path

nb_path = Path('notebooks/10_StepSize_Methods.ipynb')
nb = nbformat.read(nb_path.open('r', encoding='utf-8'), as_version=4)

cell_4_source = '''# Analysis helpers
from typing import Dict

import numpy as np

from src.analysis import (
    ResultSummary,
    TrajectoryRequest,
    compute_bloch_bundle,
    render_step_size_dashboard,
    summarize_result,
)
from src.analysis.plots import plot_cost_components, plot_population_traces, save_figure
'''

cell_6_source = '''# Run all (const, linesearch, adam)
time_grid_cfg = globals().get("time_grid_params")
runner_ctx = prepare_baseline(
    time_grid=time_grid_cfg,
    omega_shape=omega_shape,
    delta_shape=delta_shape,
    K_omega=K_omega,
    K_delta=K_delta,
    rho0=globals().get("rho0"),
    target=globals().get("target"),
    initial_omega=globals().get("initial_omega"),
    initial_delta=globals().get("initial_delta"),
)

penalties = {
    "power_weight": float(power_weight),
    "neg_weight": float(neg_weight),
    "neg_kappa": float(neg_kappa),
}
base_config, base_opts = build_base_config(
    runner_ctx.config,
    run_name=run_name,
    artifact_root=artifact_root,
    penalties=penalties,
    objective=objective,
    base_optimizer_options={
        "max_iters": int(max_iters),
        "grad_tol": float(grad_tol),
        "rtol": float(rtol),
        "max_time_s": float(max_time_min) * 60.0,
        "optimize_delta": bool(K_delta > 0),
    },
)

method_overrides = {
    "const": {"learning_rate": float(const_learning_rate)},
    "linesearch": {
        "alpha0": float(alpha0),
        "ls_beta": float(ls_beta),
        "ls_sigma": float(ls_sigma),
        "ls_max_backtracks": int(ls_max_backtracks),
    },
    "adam": {
        "learning_rate": float(adam_learning_rate),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "epsilon": float(epsilon),
    },
}

results: Dict[str, ResultSummary] = {}
bloch_data: Dict[str, Dict[str, np.ndarray]] = {}

trajectory_base = TrajectoryRequest(
    psi0=runner_ctx.psi0,
    dt_us=runner_ctx.dt_us,
    omega_base=runner_ctx.arrays["Omega0"],
    delta_base=runner_ctx.arrays.get("Delta0"),
)

for method in RUN_METHODS:
    print(f"[{method}] starting optimization")
    try:
        overrides = method_overrides.get(method, {})
        opts = method_options(method, base_opts, overrides)
        config = override_from_dict(base_config, {"optimizer_options": opts})
        result = run_experiment(
            config,
            method=method,
            run_name=f"{run_name}-{method}",
            exist_ok=True,
        )
        summary = summarize_result(result)
        results[method] = summary
        bloch_payload = compute_bloch_bundle(summary, trajectory_base, include_baseline=True)
        bloch_data[method] = {
            "optimized": bloch_payload["optimized"].pop_excited,
            "baseline": bloch_payload["baseline"].pop_excited if "baseline" in bloch_payload else None,
            "time_us": bloch_payload["optimized"].t_us,
        }
        print(f"[{method}] finished optimization (status={summary.status})")
    except Exception as exc:  # noqa: BLE001
        print(f"[{method}] error: {exc}")

if not results:
    raise RuntimeError("All optimizations failed; inspect logs above.")
'''

cell_8_source = '''# Results summary (no plots)
if not results:
    raise RuntimeError("Run the optimizer cell first.")

header = (
    f"{'method':>10}  {'total':>12}  {'terminal':>12}  {'power':>10}  "
    f"{'neg':>10}  {'iters':>8}  {'runtime_s':>10}  {'oracle':>8}  "
    f"{'max|Omega|':>10}  {'area/pi':>10}  {'neg_frac':>10}"
)
print(header)
rows = []
for method in RUN_METHODS:
    summary = results.get(method)
    if summary is None:
        continue
    metrics = summary.metrics
    row = {
        "method": method,
        "total_final": float(metrics.get("total", np.nan)),
        "terminal_final": float(metrics.get("terminal", np.nan)),
        "power_final": float(metrics.get("power_penalty", 0.0)),
        "neg_final": float(metrics.get("neg_penalty", 0.0)),
        "iterations": int(summary.iterations),
        "runtime_s": float(summary.runtime_s),
        "oracle_calls": int(summary.oracle_calls),
        "max_abs_omega": float(summary.max_abs_omega),
        "area_omega_over_pi": float(summary.area_omega_over_pi),
        "negativity_fraction": float(summary.negativity_fraction),
    }
    rows.append(row)
    line = (
        f"{method:>10}  {row['total_final']:12.5e}  {row['terminal_final']:12.5e}  "
        f"{row['power_final']:10.3e}  {row['neg_final']:10.3e}  {row['iterations']:8d}  "
        f"{row['runtime_s']:10.3f}  {row['oracle_calls']:8d}  {row['max_abs_omega']:10.3f}  "
        f"{row['area_omega_over_pi']:10.3f}  {row['negativity_fraction']:10.3f}"
    )
    print(line)

summary_fields = [
    "method",
    "total_final",
    "terminal_final",
    "power_final",
    "neg_final",
    "iterations",
    "runtime_s",
    "oracle_calls",
    "max_abs_omega",
    "area_omega_over_pi",
    "negativity_fraction",
]
summary_dir = (Path(artifact_root) / run_name).resolve()
summary_dir.mkdir(parents=True, exist_ok=True)
csv_path = summary_dir / "summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=summary_fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Summary written to {csv_path}")
'''

cell_10_source = '''# Figures
if not results:
    raise RuntimeError("Run the optimizer cell first.")

labels = [method for method in RUN_METHODS if method in results]
summary_list = [results[label] for label in labels]

fig_dir = (Path(artifact_root) / run_name / "figures").resolve()
fig_dir.mkdir(parents=True, exist_ok=True)

fig, _ = render_step_size_dashboard(summary_list, labels=labels, out_dir=fig_dir, save_svg=True)

fig_cost_components, ax_cost_components = plt.subplots(figsize=(7, 4))
plot_cost_components(summary_list, ax=ax_cost_components, labels=labels)
save_figure(fig_cost_components, fig_dir, "cost_components", save_svg=True)

population_payload = {label: bloch_data[label] for label in labels if label in bloch_data}
fig_population, ax_population = plt.subplots(figsize=(7, 4))
plot_population_traces(
    {
        label: TrajectoryBundle(
            t_us=payload["time_us"],
            psi_path=np.empty((0, 2), dtype=np.complex128),
            rho_path=np.empty((0, 2, 2), dtype=np.complex128),
            pop_excited=payload["optimized"],
        )
        for label, payload in population_payload.items()
    },
    ax=ax_population,
)
save_figure(fig_population, fig_dir, "population_vs_time", save_svg=True)

print("Saved figures:")
for path in sorted(fig_dir.glob("*.png")):
    print(f" - {path}")
'''

nb.cells[4].source = cell_4_source
nb.cells[6].source = cell_6_source
nb.cells[8].source = cell_8_source
nb.cells[10].source = cell_10_source

nbformat.write(nb, nb_path.open('w', encoding='utf-8'))
