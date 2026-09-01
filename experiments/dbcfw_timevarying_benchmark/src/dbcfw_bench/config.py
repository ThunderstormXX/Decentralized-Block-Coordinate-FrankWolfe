from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    objective: str = "quadratic"
    method: str = "dbcfw"
    agents: int = 20
    dim: int = 1000
    blocks: int = 50
    batch: int = 5
    iters: int = 500
    samples_per_agent: int = 20
    reg: float = 1e-3
    box_radius: float = 1.0
    lmo: str = "box"
    graph: str = "erdos"
    edge_prob: float = 0.25
    geometric_radius: float = 0.4
    seed: int = 42
    graph_seed: int | None = None
    log_every: int = 1
    opt_maxiter: int = 300
    wall_time_budget_sec: float | None = None
    data_dir: str | None = None
    hidden_dim: int = 16
    gamma_offset: float = 2.0
    sequence_length: int = 8
    label_count: int = 3


def is_structural_svm(objective: str) -> bool:
    return objective in {"structural_svm", "structural_sequence_svm", "ocr_structural_svm"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_run_config(config: RunConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=True)


def graph_name(name: str) -> str:
    aliases = {
        "erdos": "erdos_renyi_connected",
        "er": "erdos_renyi_connected",
        "geometric": "random_geometric_connected",
        "geo": "random_geometric_connected",
        "gossip": "pairwise_gossip",
        "matching": "pairwise_gossip",
    }
    return aliases.get(name, name)


def batch_value(value: Any, n_blocks: int) -> int:
    if isinstance(value, str) and value.lower() in {"all", "n_blocks"}:
        return n_blocks
    return int(value)


def run_id(config: RunConfig) -> str:
    parts = [
        config.objective,
        config.method,
        f"N{config.agents}",
        f"d{config.dim}",
        f"K{config.blocks}",
        f"B{config.batch}",
        f"L{config.lmo}",
        graph_name(config.graph).split("_")[0],
        f"s{config.seed}",
        f"gs{config.graph_seed if config.graph_seed is not None else config.seed}",
    ]
    return "_".join(parts)
