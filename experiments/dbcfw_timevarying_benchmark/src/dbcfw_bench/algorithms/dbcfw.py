from __future__ import annotations

import numpy as np

from dbcfw_bench.comm_graphs import GraphConfig, GraphSequence
from dbcfw_bench.config import RunConfig
from dbcfw_bench.algorithms.oracle import fw_oracle_update
from dbcfw_bench.algorithms.trace import append_row, budget_hit
from dbcfw_bench.lmo import block_slices, feasible_start
from dbcfw_bench.metrics import MetricRow, arrays_nbytes
from dbcfw_bench.objective import QuadraticProblem
from dbcfw_bench.schedules import fw_step_size
from dbcfw_bench.utils.arrays import stack_local_gradients
from dbcfw_bench.utils.timing import Timer


def run_dbcfw(
    problem: QuadraticProblem,
    config: RunConfig,
    graph_config: GraphConfig,
    f_star: float,
) -> list[MetricRow]:
    n_agents, dim = config.agents, config.dim
    slices = block_slices(dim, config.blocks)
    graph_seq = GraphSequence(n_agents, graph_config)
    block_rng = np.random.default_rng(config.seed + 1009)
    if hasattr(problem, "initial_point"):
        start = problem.initial_point(config.seed)  # type: ignore[attr-defined]
        start = feasible_start(start, config.box_radius, slices, config.lmo)
        x = np.repeat(start[None, :], n_agents, axis=0)
    else:
        x = np.zeros((n_agents, dim), dtype=float)
    x_tilde_prev = x.copy()
    grad_prev = stack_local_gradients(problem, x_tilde_prev)
    g_tilde = grad_prev.copy()
    rows: list[MetricRow] = []
    total_coords = 0
    timer = Timer()
    append_row(
        rows, problem, config, x, g_tilde, 0.0, 0, np.nan, 0, 0, np.nan, f_star
    )
    for iteration in range(config.iters):
        weights, lambda2 = graph_seq.next()
        x_tilde = weights @ x
        grad_new = stack_local_gradients(problem, x_tilde)
        g = g_tilde + grad_new - grad_prev
        g_tilde = weights @ g
        gamma = fw_step_size(iteration, config.gamma_offset)
        x, oracle_coords, oracle_bytes = fw_oracle_update(
            g_tilde, x_tilde, config, slices, block_rng, gamma
        )
        x_tilde_prev = x_tilde
        grad_prev = grad_new
        total_coords += oracle_coords
        elapsed = timer.elapsed()
        hit_budget = budget_hit(config, elapsed)
        should_log = (iteration + 1) % config.log_every == 0 or iteration + 1 == config.iters
        if should_log:
            state_bytes = arrays_nbytes(x, x_tilde, x_tilde_prev, g, g_tilde)
            state_bytes += oracle_bytes
            append_row(
                rows, problem, config, x, g_tilde, elapsed, iteration + 1,
                gamma, oracle_coords, total_coords, lambda2, f_star, state_bytes
            )
        if hit_budget:
            if not should_log:
                append_row(
                    rows, problem, config, x, g_tilde, elapsed, iteration + 1,
                    gamma, oracle_coords, total_coords, lambda2, f_star
                )
            break
    return rows
