from src.config import ExperimentConfig
from src.optimizers.base import load_crab_problem, evaluate_problem

cfg = ExperimentConfig()
problem, coeffs0, meta = load_crab_problem(cfg)
print('Loaded problem:', problem.basis_omega.shape, None if problem.basis_delta is None else problem.basis_delta.shape)

problem.objective = 'terminal'
cost, g, extras = evaluate_problem(problem, problem.coeffs_init)
print('Terminal total=', cost['total'])

problem.objective = 'path'
cost2, g2, extras2 = evaluate_problem(problem, problem.coeffs_init)
print('Path total=', cost2['total'])

problem.objective = 'ensemble'
cost3, g3, extras3 = evaluate_problem(problem, problem.coeffs_init)
print('Ensemble total=', cost3['total'], 'oracle_calls=', extras3.get('oracle_calls'))
