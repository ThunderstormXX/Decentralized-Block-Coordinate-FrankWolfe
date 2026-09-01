from __future__ import annotations

from dataclasses import replace

from dbcfw_bench.algorithms.dbcfw import run_dbcfw
from dbcfw_bench.comm_graphs import GraphConfig
from dbcfw_bench.config import RunConfig
from dbcfw_bench.metrics import MetricRow
from dbcfw_bench.objective import QuadraticProblem


def run_dfw(
    problem: QuadraticProblem,
    config: RunConfig,
    graph_config: GraphConfig,
    f_star: float,
) -> list[MetricRow]:
    dfw_config = replace(config, method="dfw", batch=config.blocks)
    return run_dbcfw(problem, dfw_config, graph_config, f_star)
