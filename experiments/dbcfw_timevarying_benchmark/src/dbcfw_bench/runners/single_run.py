from __future__ import annotations

from pathlib import Path
import tracemalloc

import pandas as pd

from dbcfw_bench.algorithms.dbcfw import run_dbcfw
from dbcfw_bench.algorithms.dfw import run_dfw
from dbcfw_bench.algorithms.cfw import run_bcfw, run_fw
from dbcfw_bench.algorithms.structural_svm import run_structural_svm_fw
from dbcfw_bench.comm_graphs import GraphConfig
from dbcfw_bench.config import RunConfig, dump_run_config, graph_name, is_structural_svm
from dbcfw_bench.runners.cache import problem_and_reference


def run_single(config: RunConfig, out_dir: str | Path) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    problem, reference = problem_and_reference(config)
    if is_structural_svm(config.objective) and config.lmo == "box":
        config.lmo = "simplex"
    if not is_structural_svm(config.objective) and config.dim % config.blocks != 0:
        raise ValueError("dim must be divisible by blocks for equal block sizes")
    graph_seed = config.graph_seed if config.graph_seed is not None else config.seed
    config.graph = graph_name(config.graph)
    config.batch = config.blocks if config.method in {"dfw", "fw"} else config.batch
    graph_config = GraphConfig(
        config.graph, config.edge_prob, config.geometric_radius, graph_seed
    )
    tracemalloc.start()
    rows = _run_algorithm(problem, config, graph_config, reference.value)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    frame = pd.DataFrame([row.to_dict() for row in rows])
    frame["peak_memory_bytes"] = int(peak)
    frame["wall_time_budget_sec"] = config.wall_time_budget_sec
    frame["objective"] = config.objective
    frame["lmo"] = config.lmo
    frame["box_radius"] = config.box_radius
    frame["reg"] = config.reg
    frame["samples_per_agent"] = config.samples_per_agent
    frame["edge_prob"] = config.edge_prob
    frame["gamma_offset"] = config.gamma_offset
    frame["sequence_length"] = config.sequence_length
    frame["label_count"] = config.label_count
    frame["f_star"] = reference.value
    frame["reference_success"] = reference.success
    frame["reference_message"] = str(reference.message)
    frame.to_csv(out / "results.csv", index=False)
    dump_run_config(config, out / "run_config.yaml")
    return frame


def _run_algorithm(problem, config: RunConfig, graph_config: GraphConfig, f_star: float):
    if is_structural_svm(config.objective):
        return run_structural_svm_fw(problem, config, graph_config, f_star)
    method = config.method.lower()
    if method == "dfw":
        return run_dfw(problem, config, graph_config, f_star)
    if method == "dbcfw":
        return run_dbcfw(problem, config, graph_config, f_star)
    if method == "fw":
        return run_fw(problem, config, f_star)
    if method == "bcfw":
        return run_bcfw(problem, config, f_star)
    raise ValueError(f"unknown method: {config.method}")
