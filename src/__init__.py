"""Public API for GRAPE/CRAB workflows using microseconds and megahertz."""

from .config import BaselineSpec, ExperimentConfig, PenaltyConfig
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
    negativity_smooth_penalty,
    total_cost,
    grad_terminal_wrt_controls,
    grad_power_fluence,
    grad_negativity_smooth,
    accumulate_cost_and_grads,
)
from .controls import (
    crab_linear_basis,
    coeffs_to_control,
    grad_control_wrt_coeffs,
    ensure_time_grid_match,
)
from .utils import ensure_dir, json_ready, require_real_finite, set_random_seed, time_block

__all__ = [
    "BaselineSpec",
    "ExperimentConfig",
    "PenaltyConfig",
    "Result",
    "run_experiment",
    "register_optimizer",
    "available_optimizers",
    "make_time_grid",
    "hamiltonian_t",
    "propagate_piecewise_const",
    "propagate_with_controls",
    "bloch_components",
    "fidelity_pure",
    "adjoint_steps",
    "terminal_infidelity",
    "path_infidelity",
    "ensemble_expectation",
    "power_fluence_penalty",
    "negativity_smooth_penalty",
    "total_cost",
    "grad_terminal_wrt_controls",
    "grad_power_fluence",
    "grad_negativity_smooth",
    "accumulate_cost_and_grads",
    "crab_linear_basis",
    "coeffs_to_control",
    "grad_control_wrt_coeffs",
    "ensure_time_grid_match",
    "ensure_dir",
    "json_ready",
    "require_real_finite",
    "set_random_seed",
    "time_block",
]
