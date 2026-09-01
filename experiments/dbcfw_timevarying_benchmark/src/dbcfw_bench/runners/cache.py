from __future__ import annotations

from dbcfw_bench.config import RunConfig
from dbcfw_bench.data import make_problem

_CACHE: dict[tuple, tuple[object, object]] = {}


def problem_and_reference(config: RunConfig):
    key = _key(config)
    if key not in _CACHE:
        problem = make_problem(config)
        config.dim = problem.dim
        if hasattr(problem, "block_count"):
            config.blocks = problem.block_count
            config.samples_per_agent = problem.block_count
        _CACHE[key] = (problem, problem.solve_reference(config.opt_maxiter))
    problem, reference = _CACHE[key]
    config.dim = problem.dim
    if hasattr(problem, "block_count"):
        config.blocks = problem.block_count
        config.samples_per_agent = problem.block_count
    return problem, reference


def _key(config: RunConfig) -> tuple:
    return (
        config.objective,
        config.agents,
        config.dim,
        config.blocks,
        config.samples_per_agent,
        config.reg,
        config.box_radius,
        config.seed,
        config.opt_maxiter,
        config.data_dir,
        config.hidden_dim,
        config.sequence_length,
        config.label_count,
    )
