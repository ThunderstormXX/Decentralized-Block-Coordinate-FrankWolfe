from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dbcfw_bench.algorithms.structural_svm import _line_search_gamma
from dbcfw_bench.comm_graphs import GraphConfig, GraphSequence
from dbcfw_bench.data_ocr import load_taskar_ocr
from dbcfw_bench.objective_structural_svm import StructuralSequenceSVMProblem


@dataclass
class TimedUpdate:
    delta: np.ndarray
    calls: int
    lmo_time_sec: float
    line_search_time_sec: float
    update_rest_time_sec: float
    update_time_sec: float
    total_time_sec: float


def main() -> None:
    args = _parser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    problem = _load_problem(args)
    batches = [int(value) for value in args.batches]
    if problem.block_count not in batches:
        batches.append(problem.block_count)
    batches = sorted(set(batches))

    rows: list[dict[str, float | int | str]] = []
    for batch in batches:
        method = "dfw" if batch >= problem.block_count else "dbcfw"
        rows.extend(_measure_batch(problem, args, method, batch))

    frame = pd.DataFrame(rows)
    round_path = out / "round_times.csv"
    frame.to_csv(round_path, index=False)

    summary = _summarize(frame, problem)
    summary_path = out / "round_time_summary.csv"
    summary.to_csv(summary_path, index=False)

    threshold = _communication_thresholds(summary, full_batch=problem.block_count)
    threshold_path = out / "communication_thresholds.csv"
    threshold.to_csv(threshold_path, index=False)

    plot_path = _plot_summary(summary, out / "round_time_vs_block_budget.png")
    md_path = _write_markdown(summary, threshold, problem, out / "round_time_report.md")

    print(f"wrote {round_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {threshold_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {md_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs_paper_ocr/ocr_round_timing"))
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=893)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1207)
    parser.add_argument("--graph-seed", type=int, default=2207)
    parser.add_argument("--graph", type=str, default="erdos")
    parser.add_argument("--edge-prob", type=float, default=0.35)
    parser.add_argument("--reg", type=str, default="1/n")
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 25, 50, 100, 200, 400],
        help="Local word-block count sampled per agent. Full DFW batch is added automatically.",
    )
    return parser


def _load_problem(args: argparse.Namespace) -> StructuralSequenceSVMProblem:
    train_x, train_y, test_x, test_y = load_taskar_ocr(args.data_dir, "ocr2")
    total = args.agents * args.blocks
    train_x = train_x[:total]
    train_y = train_y[:total]
    lambd = (1.0 / len(train_x)) if args.reg == "1/n" else float(args.reg)
    x_parts = [list(part) for part in np.array_split(np.asarray(train_x, dtype=object), args.agents)]
    y_parts = [list(part) for part in np.array_split(np.asarray(train_y, dtype=object), args.agents)]
    return StructuralSequenceSVMProblem(
        x_parts,
        y_parts,
        lambd,
        classes=26,
        position_bias=True,
        test_x=test_x,
        test_y=test_y,
    )


def _measure_batch(
    problem: StructuralSequenceSVMProblem,
    args: argparse.Namespace,
    method: str,
    batch: int,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(args.seed + 17 * batch)
    graph_seq = GraphSequence(args.agents, GraphConfig(args.graph, args.edge_prob, seed=args.graph_seed))
    block_w = np.zeros((args.agents, problem.block_count, problem.dim), dtype=float)
    block_ell = np.zeros((args.agents, problem.block_count), dtype=float)
    points = np.zeros((args.agents, problem.dim), dtype=float)
    local_w = np.zeros((args.agents, problem.dim), dtype=float)
    rows: list[dict[str, float | int | str]] = []

    total_rounds = args.warmup_rounds + args.rounds
    for round_id in range(total_rounds):
        graph_t0 = time.perf_counter()
        weights, lambda2 = graph_seq.next()
        graph_time = time.perf_counter() - graph_t0

        mix_t0 = time.perf_counter()
        mixed = weights @ points
        mix_time = time.perf_counter() - mix_t0

        next_points = mixed.copy()
        agent_lmo_times: list[float] = []
        agent_line_search_times: list[float] = []
        agent_update_rest_times: list[float] = []
        agent_update_times: list[float] = []
        agent_total_times: list[float] = []
        calls = 0

        for agent in range(args.agents):
            selected = _selected_blocks(rng, problem.block_count, batch)
            timed = _timed_agent_update(problem, agent, selected, mixed[agent], block_w, block_ell)
            local_w[agent] += timed.delta
            next_points[agent] += args.agents * timed.delta
            agent_lmo_times.append(timed.lmo_time_sec)
            agent_line_search_times.append(timed.line_search_time_sec)
            agent_update_rest_times.append(timed.update_rest_time_sec)
            agent_update_times.append(timed.update_time_sec)
            agent_total_times.append(timed.total_time_sec)
            calls += timed.calls

        points = next_points
        directed_messages = int(np.count_nonzero(weights - np.diag(np.diag(weights))))
        payload_bytes = directed_messages * problem.dim * 8

        if round_id >= args.warmup_rounds:
            rows.append(
                {
                    "method": method,
                    "batch": batch,
                    "round": round_id - args.warmup_rounds,
                    "agents": args.agents,
                    "blocks_per_agent": problem.block_count,
                    "dim": problem.dim,
                    "lambda2": lambda2,
                    "training_oracle_calls": calls,
                    "graph_sampling_sec": graph_time,
                    "communication_mixing_sec": mix_time,
                    "directed_messages": directed_messages,
                    "payload_bytes": payload_bytes,
                    "mean_agent_lmo_sec": float(np.mean(agent_lmo_times)),
                    "max_agent_lmo_sec": float(np.max(agent_lmo_times)),
                    "mean_agent_line_search_sec": float(np.mean(agent_line_search_times)),
                    "max_agent_line_search_sec": float(np.max(agent_line_search_times)),
                    "mean_agent_update_rest_sec": float(np.mean(agent_update_rest_times)),
                    "max_agent_update_rest_sec": float(np.max(agent_update_rest_times)),
                    "mean_agent_update_overhead_sec": float(np.mean(agent_update_times)),
                    "max_agent_update_overhead_sec": float(np.max(agent_update_times)),
                    "mean_agent_compute_sec": float(np.mean(agent_total_times)),
                    "max_agent_compute_sec": float(np.max(agent_total_times)),
                    "round_lmo_plus_comm_sec": float(np.max(agent_lmo_times) + mix_time),
                    "round_compute_plus_comm_sec": float(np.max(agent_total_times) + mix_time),
                }
            )
    return rows


def _timed_agent_update(
    problem: StructuralSequenceSVMProblem,
    agent: int,
    selected: np.ndarray,
    mixed_point: np.ndarray,
    block_w: np.ndarray,
    block_ell: np.ndarray,
) -> TimedUpdate:
    total_t0 = time.perf_counter()
    old_w = block_w[agent, selected].sum(axis=0)
    old_ell = float(block_ell[agent, selected].sum())
    target_blocks = np.zeros((len(selected), problem.dim), dtype=float)
    target_ells = np.zeros(len(selected), dtype=float)
    lmo_time = 0.0
    for pos, block in enumerate(selected):
        lmo_t0 = time.perf_counter()
        vertex, ell_s, _, _ = problem.oracle_vertex(agent, int(block), mixed_point)
        lmo_time += time.perf_counter() - lmo_t0
        target_blocks[pos] = vertex
        target_ells[pos] = ell_s
    update_t0 = time.perf_counter()
    target_w = target_blocks.sum(axis=0)
    target_ell = float(target_ells.sum())
    line_search_t0 = time.perf_counter()
    gamma = _line_search_gamma(problem.reg, old_w, old_ell, target_w, target_ell, mixed_point)
    line_search_time = time.perf_counter() - line_search_t0
    block_w[agent, selected] = (1.0 - gamma) * block_w[agent, selected] + gamma * target_blocks
    block_ell[agent, selected] = (1.0 - gamma) * block_ell[agent, selected] + gamma * target_ells
    delta = gamma * (target_w - old_w)
    update_time = time.perf_counter() - update_t0
    total_time = time.perf_counter() - total_t0
    update_rest_time = max(update_time - line_search_time, 0.0)
    return TimedUpdate(
        delta,
        int(len(selected)),
        lmo_time,
        line_search_time,
        update_rest_time,
        update_time,
        total_time,
    )


def _selected_blocks(rng: np.random.Generator, block_count: int, batch: int) -> np.ndarray:
    if batch >= block_count:
        return np.arange(block_count)
    return rng.choice(block_count, size=batch, replace=False)


def _summarize(frame: pd.DataFrame, problem: StructuralSequenceSVMProblem) -> pd.DataFrame:
    grouped = frame.groupby(["method", "batch"], as_index=False).agg(
        rounds=("round", "count"),
        calls_per_round=("training_oracle_calls", "mean"),
        mean_agent_lmo_ms=("mean_agent_lmo_sec", lambda s: 1000.0 * float(np.mean(s))),
        max_agent_lmo_ms=("max_agent_lmo_sec", lambda s: 1000.0 * float(np.mean(s))),
        max_agent_lmo_p90_ms=("max_agent_lmo_sec", lambda s: 1000.0 * float(np.percentile(s, 90))),
        mean_agent_line_search_ms=("mean_agent_line_search_sec", lambda s: 1000.0 * float(np.mean(s))),
        max_agent_line_search_ms=("max_agent_line_search_sec", lambda s: 1000.0 * float(np.mean(s))),
        mean_agent_update_rest_ms=("mean_agent_update_rest_sec", lambda s: 1000.0 * float(np.mean(s))),
        max_agent_update_rest_ms=("max_agent_update_rest_sec", lambda s: 1000.0 * float(np.mean(s))),
        max_agent_update_overhead_ms=("max_agent_update_overhead_sec", lambda s: 1000.0 * float(np.mean(s))),
        mean_agent_compute_ms=("mean_agent_compute_sec", lambda s: 1000.0 * float(np.mean(s))),
        max_agent_compute_ms=("max_agent_compute_sec", lambda s: 1000.0 * float(np.mean(s))),
        communication_mixing_us=("communication_mixing_sec", lambda s: 1e6 * float(np.mean(s))),
        graph_sampling_us=("graph_sampling_sec", lambda s: 1e6 * float(np.mean(s))),
        directed_messages=("directed_messages", "mean"),
        payload_kib=("payload_bytes", lambda s: float(np.mean(s)) / 1024.0),
        round_lmo_plus_comm_ms=("round_lmo_plus_comm_sec", lambda s: 1000.0 * float(np.mean(s))),
        round_compute_plus_comm_ms=("round_compute_plus_comm_sec", lambda s: 1000.0 * float(np.mean(s))),
    )
    full = grouped[grouped["batch"] == problem.block_count].iloc[0]
    grouped["speedup_vs_dfw_lmo_comm"] = (
        float(full["round_lmo_plus_comm_ms"]) / grouped["round_lmo_plus_comm_ms"]
    )
    grouped["speedup_vs_dfw_compute_comm"] = (
        float(full["round_compute_plus_comm_ms"]) / grouped["round_compute_plus_comm_ms"]
    )
    return grouped.sort_values("batch")


def _communication_thresholds(summary: pd.DataFrame, full_batch: int) -> pd.DataFrame:
    full = summary[summary["batch"] == full_batch].iloc[0]
    l_full = float(full["max_agent_compute_ms"]) / 1000.0
    rows = []
    for _, row in summary.iterrows():
        b = int(row["batch"])
        l_b = float(row["max_agent_compute_ms"]) / 1000.0
        for target_speedup in (10.0, 5.0, 2.0, 1.5, 1.1):
            # (L_full + C) / (L_b + C) >= target_speedup
            # C <= (L_full - target_speedup * L_b) / (target_speedup - 1)
            numerator = l_full - target_speedup * l_b
            threshold = numerator / (target_speedup - 1.0)
            rows.append(
                {
                    "batch": b,
                    "target_speedup": target_speedup,
                    "max_communication_sec_for_target": max(float(threshold), 0.0),
                }
            )
    return pd.DataFrame(rows)


def _plot_summary(summary: pd.DataFrame, path: Path) -> Path:
    df = summary.sort_values("batch")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4), dpi=180)
    axes[0].plot(df["batch"], df["max_agent_lmo_ms"], marker="o", label="max agent LMO")
    axes[0].plot(df["batch"], df["round_compute_plus_comm_ms"], marker="s", label="compute + mixing")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("local blocks sampled per agent, B")
    axes[0].set_ylabel("round time proxy (ms)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(df["batch"], df["speedup_vs_dfw_compute_comm"], marker="o", color="#008b84")
    axes[1].axhline(1.0, color="#667085", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("local blocks sampled per agent, B")
    axes[1].set_ylabel("speedup vs DFW round")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return path


def _write_markdown(
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    problem: StructuralSequenceSVMProblem,
    path: Path,
) -> Path:
    cols = [
        "method",
        "batch",
        "calls_per_round",
        "max_agent_lmo_ms",
        "max_agent_line_search_ms",
        "max_agent_update_rest_ms",
        "communication_mixing_us",
        "payload_kib",
        "round_compute_plus_comm_ms",
        "speedup_vs_dfw_compute_comm",
    ]
    text = [
        "# OCR Structural SVM round-time microbenchmark",
        "",
        f"- agents: {problem.agents}",
        f"- local word blocks per agent: {problem.block_count}",
        f"- model dimension: {problem.dim}",
        "- communication is the simulator's consensus/mixing operation (`weights @ points`), not a real network stack.",
        "",
        summary[cols].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Communication thresholds",
        "",
        "For each block size B, threshold C solves `(L_full + C)/(L_B + C) >= target_speedup`, where L is max-agent compute time.",
        "",
        thresholds.to_markdown(index=False, floatfmt=".4g"),
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
