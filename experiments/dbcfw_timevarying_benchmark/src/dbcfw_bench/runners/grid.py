from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from dbcfw_bench.config import RunConfig, batch_value, is_structural_svm, load_yaml
from dbcfw_bench.runners.single_run import run_single


def run_grid(config_path: str | Path, out_dir: str | Path) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for run_cfg in expand_grid(cfg):
        run_out = out / run_cfg.method / _run_dir_name(run_cfg)
        frames.append(run_single(run_cfg, run_out))
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(out / "results.csv", index=False)
    with (out / "grid_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)
    return result


def expand_grid(cfg: dict) -> list[RunConfig]:
    base = RunConfig(
        objective=str(cfg.get("objective", "quadratic")),
        iters=int(cfg.get("iters", 500)),
        samples_per_agent=int(cfg.get("samples_per_agent", 20)),
        reg=float(cfg.get("reg", 1e-3)),
        box_radius=float(cfg.get("box_radius", 1.0)),
        lmo=str(cfg.get("lmo", "box")),
        edge_prob=float(cfg.get("edge_prob", 0.25)),
        geometric_radius=float(cfg.get("geometric_radius", 0.4)),
        seed=int(cfg.get("seed", 42)),
        graph_seed=cfg.get("graph_seed", cfg.get("seed", 42)),
        log_every=int(cfg.get("log_every", 1)),
        opt_maxiter=int(cfg.get("opt_maxiter", 300)),
        wall_time_budget_sec=cfg.get("wall_time_budget_sec"),
        data_dir=cfg.get("data_dir"),
        hidden_dim=int(cfg.get("hidden_dim", 16)),
        gamma_offset=float(cfg.get("gamma_offset", 2.0)),
        sequence_length=int(cfg.get("sequence_length", 8)),
        label_count=int(cfg.get("label_count", 3)),
    )
    axes = product(
        cfg.get("methods", ["dfw", "dbcfw"]),
        cfg.get("agents", [20]),
        cfg.get("dims", [1000]),
        cfg.get("blocks", [50]),
        cfg.get("batches", [5]),
        cfg.get("graphs", ["erdos"]),
        cfg.get("lmos", [cfg.get("lmo", "box")]),
    )
    runs: list[RunConfig] = []
    for method, agents, dim, blocks, batch, graph, lmo in axes:
        batch_int = batch_value(batch, int(blocks))
        if method in {"dfw", "fw"} and batch_int != int(blocks):
            continue
        if not is_structural_svm(base.objective) and int(dim) % int(blocks) != 0:
            continue
        runs.append(replace(
            base, method=str(method), agents=int(agents), dim=int(dim),
            blocks=int(blocks), batch=batch_int, graph=str(graph), lmo=str(lmo)
        ))
    return runs


def _run_dir_name(config: RunConfig) -> str:
    return (
        f"N{config.agents}_d{config.dim}_K{config.blocks}_B{config.batch}_"
        f"L{config.lmo}_{config.graph}_s{config.seed}_gs{config.graph_seed}"
    )
