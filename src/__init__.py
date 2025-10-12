"""Public API for GRAPE coefficient workflows using microseconds and megahertz."""

from .baselines import (
    BasisSpec,
    GrapeBaselineConfig,
    PulseShapeSpec,
    TimeGridSpec,
    build_grape_baseline,
    write_baseline,
)
from .config import BaselineSpec, ExperimentConfig, PenaltyConfig, override_from_dict
from .result import Result
from .workflows import available_optimizers, register_optimizer, run_experiment
from .physics import (
    make_time_grid,
    hamiltonian_t,
    propagate_piecewise_const,
    propagate_with_controls,
    bloch_components,
    fidelity_pure,
    adjoint_steps,
)
from .cost import (
    terminal_infidelity,
    path_infidelity,
    ensemble_expectation,
    power_fluence_penalty,
    total_cost,
    grad_terminal_wrt_controls,
    grad_power_fluence,
    accumulate_cost_and_grads,
)
from .penalties import compute_penalties, penalty_terms
from .controls import (
    crab_linear_basis,
    coeffs_to_control,
    grad_control_wrt_coeffs,
    ensure_time_grid_match,
)
from .plotting import (
    plot_cost_history,
    plot_pulses,
    plot_summary,
    plot_penalties_history,
    plot_robustness_heatmap,
)
from .utils import ensure_dir, json_ready, require_real_finite, set_random_seed, time_block

from .notebook_runners import (
    BaselineArrays,
    prepare_baseline,
    coerce_vector,
    build_base_config,
    method_options,
)

__all__ = [
    "BaselineSpec",
    "BasisSpec",
    "ExperimentConfig",
    "GrapeBaselineConfig",
    "PenaltyConfig",
    "PulseShapeSpec",
    "TimeGridSpec",
    "accumulate_cost_and_grads",
    "adjoint_steps",
    "available_optimizers",
    "bloch_components",
    "build_grape_baseline",
    "coeffs_to_control",
    "compute_penalties",
    "crab_linear_basis",
    "ensure_dir",
    "ensure_time_grid_match",
    "ensemble_expectation",
    "fidelity_pure",
    "grad_control_wrt_coeffs",
    "grad_power_fluence",
    "grad_terminal_wrt_controls",
    "hamiltonian_t",
    "json_ready",
    "make_time_grid",
    "override_from_dict",
    "path_infidelity",
    "penalty_terms",
    "plot_cost_history",
    "plot_penalties_history",
    "plot_pulses",
    "plot_robustness_heatmap",
    "plot_summary",
    "power_fluence_penalty",
    "propagate_piecewise_const",
    "propagate_with_controls",
    "register_optimizer",
    "require_real_finite",
    "Result",
    "run_experiment",
    "set_random_seed",
    "terminal_infidelity",
    "time_block",
    "total_cost",
    "write_baseline",
    "BaselineArrays",
    "build_base_config",
    "coerce_vector",
    "method_options",
    "prepare_baseline",
]
