from __future__ import annotations

from dataclasses import replace

import numpy as np

from dbcfw_bench.comm_graphs import GraphConfig, GraphSequence
from dbcfw_bench.config import RunConfig, graph_name, run_id
from dbcfw_bench.metrics import MetricRow, arrays_nbytes, consensus_error
from dbcfw_bench.objective_structural_svm import StructuralSequenceSVMProblem
from dbcfw_bench.algorithms.trace import budget_hit
from dbcfw_bench.utils.timing import Timer


def run_structural_svm_fw(
    problem: StructuralSequenceSVMProblem,
    config: RunConfig,
    graph_config: GraphConfig,
    f_star: float,
) -> list[MetricRow]:
    del f_star
    if config.method == "dfw":
        config = replace(config, batch=problem.block_count)
    config.blocks = problem.block_count
    config.dim = problem.dim
    graph_seq = GraphSequence(config.agents, graph_config)
    rng = np.random.default_rng(config.seed + 1207)
    block_w = np.zeros((config.agents, problem.block_count, problem.dim), dtype=float)
    block_ell = np.zeros((config.agents, problem.block_count), dtype=float)
    local_w = np.zeros((config.agents, problem.dim), dtype=float)
    points = np.zeros((config.agents, problem.dim), dtype=float)
    rows: list[MetricRow] = []
    total_calls = 0
    timer = Timer()
    total_calls = _append_row(
        rows, problem, config, points, block_w, block_ell, local_w,
        0.0, 0, np.nan, 0, total_calls, np.nan,
    )
    for iteration in range(config.iters):
        weights, lambda2 = graph_seq.next()
        mixed = weights @ points
        next_points = mixed.copy()
        update_calls = 0
        gammas = []
        for agent in range(config.agents):
            selected = _selected_blocks(rng, problem.block_count, config.batch)
            delta, calls, gamma_values = _agent_update(
                problem, config, agent, selected, mixed[agent], block_w, block_ell
            )
            local_w[agent] += delta
            next_points[agent] += config.agents * delta
            update_calls += calls
            gammas.extend(gamma_values)
        points = next_points
        elapsed = timer.elapsed()
        hit_budget = budget_hit(config, elapsed)
        should_log = (iteration + 1) % config.log_every == 0 or iteration + 1 == config.iters
        total_calls += update_calls
        if should_log:
            total_calls = _append_row(
                rows, problem, config, points, block_w, block_ell, local_w,
                elapsed, iteration + 1, _mean_gamma(gammas), update_calls,
                total_calls, lambda2, mixed,
            )
        if hit_budget:
            if not should_log:
                total_calls = _append_row(
                    rows, problem, config, points, block_w, block_ell, local_w,
                    elapsed, iteration + 1, _mean_gamma(gammas), update_calls,
                    total_calls, lambda2, mixed,
                )
            break
    return rows


def _agent_update(
    problem: StructuralSequenceSVMProblem,
    config: RunConfig,
    agent: int,
    selected: np.ndarray,
    mixed_point: np.ndarray,
    block_w: np.ndarray,
    block_ell: np.ndarray,
) -> tuple[np.ndarray, int, list[float]]:
    old_w = block_w[agent, selected].sum(axis=0)
    old_ell = float(block_ell[agent, selected].sum())
    target_blocks = np.zeros((len(selected), problem.dim), dtype=float)
    target_ells = np.zeros(len(selected), dtype=float)
    calls = 0
    for pos, block in enumerate(selected):
        vertex, ell_s, _, _ = problem.oracle_vertex(agent, int(block), mixed_point)
        target_blocks[pos] = vertex
        target_ells[pos] = ell_s
        calls += 1
    target_w = target_blocks.sum(axis=0)
    target_ell = float(target_ells.sum())
    gamma = _line_search_gamma(problem.reg, old_w, old_ell, target_w, target_ell, mixed_point)
    block_w[agent, selected] = (1.0 - gamma) * block_w[agent, selected] + gamma * target_blocks
    block_ell[agent, selected] = (1.0 - gamma) * block_ell[agent, selected] + gamma * target_ells
    delta = gamma * (target_w - old_w)
    return delta, calls, [gamma]


def _line_search_gamma(
    reg: float,
    old_w: np.ndarray,
    old_ell: float,
    target_w: np.ndarray,
    target_ell: float,
    model: np.ndarray,
) -> float:
    diff = old_w - target_w
    denom = reg * float(diff @ diff)
    if denom <= 1e-18:
        return 0.0
    numerator = reg * float(diff @ model) - old_ell + target_ell
    return float(np.clip(numerator / denom, 0.0, 1.0))


def _append_row(
    rows: list[MetricRow],
    problem: StructuralSequenceSVMProblem,
    config: RunConfig,
    points: np.ndarray,
    block_w: np.ndarray,
    block_ell: np.ndarray,
    local_w: np.ndarray,
    elapsed: float,
    iteration: int,
    gamma: float,
    update_calls: int,
    total_calls: int,
    lambda2: float,
    mixed: np.ndarray | None = None,
) -> int:
    w_global = local_w.sum(axis=0)
    ell = float(block_ell.sum())
    gap, gap_calls = problem.duality_gap(w_global, ell)
    total_with_gap = total_calls + gap_calls
    state_arrays = [points, block_w, block_ell, local_w]
    if mixed is not None:
        state_arrays.append(mixed)
    graph_seed = config.graph_seed if config.graph_seed is not None else config.seed
    rows.append(MetricRow(
        run_id(config),
        config.method,
        graph_name(config.graph),
        config.agents,
        config.dim,
        config.blocks,
        config.batch,
        config.seed,
        graph_seed,
        iteration,
        gamma,
        elapsed,
        gap,
        problem.mean_agent_accuracy(points),
        consensus_error(points),
        float("nan"),
        update_calls + gap_calls,
        total_with_gap,
        lambda2,
        arrays_nbytes(*state_arrays),
        float("nan"),
        problem.test_error(points.mean(axis=0)),
    ))
    return total_with_gap


def _selected_blocks(rng: np.random.Generator, block_count: int, batch: int) -> np.ndarray:
    if batch >= block_count:
        return np.arange(block_count)
    return rng.choice(block_count, size=batch, replace=False)


def _mean_gamma(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")
