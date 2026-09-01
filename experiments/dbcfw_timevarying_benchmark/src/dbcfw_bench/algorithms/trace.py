from __future__ import annotations

import numpy as np

from dbcfw_bench.config import RunConfig, graph_name, run_id
from dbcfw_bench.metrics import (
    MetricRow,
    arrays_nbytes,
    boundary_activity,
    consensus_error,
    tracker_disagreement,
)
from dbcfw_bench.objective import QuadraticProblem


def budget_hit(config: RunConfig, elapsed: float) -> bool:
    budget = config.wall_time_budget_sec
    return budget is not None and elapsed >= budget


def append_row(
    rows: list[MetricRow],
    problem: QuadraticProblem,
    config: RunConfig,
    points: np.ndarray,
    trackers: np.ndarray,
    elapsed: float,
    iteration: int,
    gamma: float,
    oracle_coords: int,
    total_coords: int,
    lambda2: float,
    f_star: float,
    state_bytes: int | None = None,
) -> None:
    graph_seed = config.graph_seed if config.graph_seed is not None else config.seed
    x_avg = points.mean(axis=0)
    state = state_bytes if state_bytes is not None else arrays_nbytes(points, trackers)
    accuracy = _mean_agent_accuracy(problem, points)
    rows.append(MetricRow(
        run_id(config), config.method, graph_name(config.graph), config.agents,
        config.dim, config.blocks, config.batch, config.seed, graph_seed, iteration,
        gamma, elapsed, problem.objective(x_avg) - f_star, accuracy, consensus_error(points),
        tracker_disagreement(trackers), oracle_coords, total_coords, lambda2, state,
        boundary_activity(points, config.box_radius, config.lmo, config.blocks)
    ))


def _mean_agent_accuracy(problem: QuadraticProblem, points: np.ndarray) -> float:
    if hasattr(problem, "mean_agent_accuracy"):
        return float(problem.mean_agent_accuracy(points))
    return float("nan")
