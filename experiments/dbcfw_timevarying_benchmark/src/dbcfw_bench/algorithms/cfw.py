from __future__ import annotations

from dataclasses import replace

import numpy as np

from dbcfw_bench.algorithms.trace import append_row, budget_hit
from dbcfw_bench.config import RunConfig
from dbcfw_bench.lmo import block_lmo, block_slices, feasible_start, full_lmo
from dbcfw_bench.metrics import MetricRow, arrays_nbytes
from dbcfw_bench.schedules import choose_blocks, fw_step_size
from dbcfw_bench.utils.arrays import stack_local_gradients
from dbcfw_bench.utils.timing import Timer


def run_fw(problem, config: RunConfig, f_star: float) -> list[MetricRow]:
    fw_config = replace(config, method="fw", batch=config.blocks, graph="centralized")
    return _run_central_fw(problem, fw_config, f_star)


def run_bcfw(problem, config: RunConfig, f_star: float) -> list[MetricRow]:
    bcfw_config = replace(config, method="bcfw", graph="centralized")
    return _run_central_fw(problem, bcfw_config, f_star)


def _run_central_fw(problem, config: RunConfig, f_star: float) -> list[MetricRow]:
    slices = block_slices(config.dim, config.blocks)
    rng = np.random.default_rng(config.seed + 2003)
    x = _initial_point(problem, config, slices)
    rows: list[MetricRow] = []
    total_coords = 0
    timer = Timer()
    gradient = _gradient(problem, config, x)
    append_row(
        rows,
        problem,
        config,
        _agent_points(x, config.agents),
        _agent_points(gradient, config.agents),
        0.0,
        0,
        np.nan,
        0,
        0,
        np.nan,
        f_star,
    )
    for iteration in range(config.iters):
        gradient = _gradient(problem, config, x)
        gamma = fw_step_size(iteration, config.gamma_offset)
        x, oracle_coords, oracle_bytes = _oracle_update(
            gradient, x, config, slices, rng, gamma
        )
        total_coords += oracle_coords
        elapsed = timer.elapsed()
        hit_budget = budget_hit(config, elapsed)
        should_log = (iteration + 1) % config.log_every == 0 or iteration + 1 == config.iters
        if should_log:
            points = _agent_points(x, config.agents)
            trackers = _agent_points(gradient, config.agents)
            state_bytes = arrays_nbytes(x, gradient, points, trackers) + oracle_bytes
            append_row(
                rows,
                problem,
                config,
                points,
                trackers,
                elapsed,
                iteration + 1,
                gamma,
                oracle_coords,
                total_coords,
                np.nan,
                f_star,
                state_bytes,
            )
        if hit_budget:
            if not should_log:
                append_row(
                    rows,
                    problem,
                    config,
                    _agent_points(x, config.agents),
                    _agent_points(gradient, config.agents),
                    elapsed,
                    iteration + 1,
                    gamma,
                    oracle_coords,
                    total_coords,
                    np.nan,
                    f_star,
                )
            break
    return rows


def _initial_point(problem, config: RunConfig, slices: list[slice]) -> np.ndarray:
    if hasattr(problem, "initial_point"):
        point = problem.initial_point(config.seed)
    else:
        point = np.zeros(config.dim, dtype=float)
    return feasible_start(point, config.box_radius, slices, config.lmo)


def _gradient(problem, config: RunConfig, x: np.ndarray) -> np.ndarray:
    if hasattr(problem, "grad"):
        return problem.grad(x)
    points = _agent_points(x, config.agents)
    return stack_local_gradients(problem, points).mean(axis=0)


def _oracle_update(
    gradient: np.ndarray,
    x: np.ndarray,
    config: RunConfig,
    slices: list[slice],
    rng: np.random.Generator,
    gamma: float,
) -> tuple[np.ndarray, int, int]:
    if config.batch >= config.blocks:
        target = full_lmo(gradient, config.box_radius, slices, config.lmo)
        return (1.0 - gamma) * x + gamma * target, config.dim, target.nbytes

    x_next = x.copy()
    coords = 0
    for idx in choose_blocks(rng, config.blocks, config.batch):
        sl = slices[int(idx)]
        target = block_lmo(gradient[sl], config.box_radius, config.lmo)
        x_next[sl] = (1.0 - gamma) * x[sl] + gamma * target
        coords += sl.stop - sl.start
    return x_next, coords, 0


def _agent_points(x: np.ndarray, agents: int) -> np.ndarray:
    return np.repeat(x[None, :], agents, axis=0)
