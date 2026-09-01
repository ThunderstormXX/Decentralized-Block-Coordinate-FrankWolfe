from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dbcfw_bench.algorithms.structural_svm import _line_search_gamma
from dbcfw_bench.comm_graphs import GraphConfig, GraphSequence
from dbcfw_bench.data_ocr import load_taskar_ocr
from dbcfw_bench.objective_structural_svm import StructuralSequenceSVMProblem


@dataclass(frozen=True)
class TraceSpec:
    method: str
    batch: int
    label: str
    color: str


TRACE_SPECS = [
    TraceSpec("dbcfw", 1, "DBCFW, B=1", "#008b84"),
    TraceSpec("dbcfw", 10, "DBCFW, B=10", "#3b82f6"),
    TraceSpec("dbcfw", 89, "DBCFW, B=89", "#8b5cf6"),
    TraceSpec("dfw", 893, "DFW, B=893", "#d95f02"),
]


def main() -> None:
    args = _parser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    round_summary = pd.read_csv(args.round_summary)
    agent_sweep = pd.read_csv(args.agent_sweep)
    fair_results = pd.read_csv(args.fair_results)

    fig4 = out / "structural_svm_round_time_vs_block_budget.png"
    fig5 = out / "structural_svm_agent_sweep_round_time.png"
    fig6 = out / "structural_svm_ocr_time_panels.png"

    _plot_round_time(round_summary, fig4)
    _plot_agent_sweep(agent_sweep, fig5)
    trace = _build_value_trace(args, round_summary, fair_results)
    trace_path = out / "structural_svm_ocr_time_trace.csv"
    trace.to_csv(trace_path, index=False)
    _plot_time_panels(trace, fig6)

    print(f"wrote {fig4}")
    print(f"wrote {fig5}")
    print(f"wrote {trace_path}")
    print(f"wrote {fig6}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--round-summary",
        type=Path,
        default=Path("runs_paper_ocr/ocr_round_timing_with_b89/round_time_summary.csv"),
    )
    parser.add_argument(
        "--agent-sweep",
        type=Path,
        default=Path("runs_paper_ocr/ocr_round_timing_agents/agent_sweep_compact.csv"),
    )
    parser.add_argument(
        "--fair-results",
        type=Path,
        default=Path("runs_paper_ocr/ocr_global_ls_lambda005_combined/results.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("runs_paper_ocr/crm_figures"))
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=893)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--graph-seed", type=int, default=2207)
    parser.add_argument("--edge-prob", type=float, default=0.35)
    parser.add_argument("--lambda-label", type=str, default="0.05")
    return parser


def _plot_round_time(summary: pd.DataFrame, path: Path) -> None:
    df = summary.sort_values("batch").copy()
    full = df[df["method"] == "dfw"].iloc[0]
    nonfull = df[df["batch"] < int(full["batch"])]

    fig, ax = plt.subplots(figsize=(5.7, 3.2), dpi=220)
    ax.plot(
        nonfull["batch"],
        nonfull["round_compute_plus_comm_ms"],
        marker="o",
        linewidth=1.8,
        color="#008b84",
        label="DBCFW round time",
    )
    ax.scatter(
        [full["batch"]],
        [full["round_compute_plus_comm_ms"]],
        s=72,
        marker="*",
        color="#d95f02",
        zorder=4,
        label="DFW endpoint, B=n",
    )
    ax.annotate(
        "full local oracle\nB=n=893",
        xy=(full["batch"], full["round_compute_plus_comm_ms"]),
        xytext=(-74, -5),
        textcoords="offset points",
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#555"},
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$B$")
    ax.set_ylabel("round time (ms)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_agent_sweep(agent_sweep: pd.DataFrame, path: Path) -> None:
    df = agent_sweep.sort_values("agents")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=220)

    axes[0].plot(df["agents"], df["dfw_round_ms"], marker="o", color="#1f77b4", label="DFW, B=K")
    axes[0].plot(df["agents"], df["dbcfw_b1_round_ms"], marker="s", color="#008b84", label="DBCFW, B=1")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("agents $N$")
    axes[0].set_ylabel("round time (ms)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(df["agents"], df["b1_speedup"], marker="o", color="#008b84")
    axes[1].set_xlabel("agents $N$")
    axes[1].set_ylabel("speedup")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title("DBCFW vs DFW", fontsize=9, pad=4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _build_value_trace(
    args: argparse.Namespace,
    round_summary: pd.DataFrame,
    fair_results: pd.DataFrame,
) -> pd.DataFrame:
    round_ms = {
        int(row.batch): float(row.round_compute_plus_comm_ms)
        for row in round_summary.itertuples(index=False)
    }
    rows = []
    if "train_primal" in fair_results.columns and fair_results["train_primal"].notna().any():
        for spec in TRACE_SPECS:
            source = fair_results[
                _lambda_mask(fair_results, args.lambda_label)
                & (fair_results["method"] == spec.method)
                & (fair_results["batch"] == spec.batch)
            ].sort_values("iteration")
            for row in source.itertuples(index=False):
                rows.append(
                    {
                        "method": spec.method,
                        "batch": spec.batch,
                        "label": spec.label,
                        "color": spec.color,
                        "iteration": int(row.iteration),
                        "time_ms": int(row.iteration) * round_ms[spec.batch],
                        "test_error": float(row.test_error),
                        "raw_duality_gap": max(float(row.objective_gap), 1e-12),
                        "dual_value": float("nan"),
                        "primal_value": max(float(row.train_primal), 1e-12),
                    }
                )
        return _finish_trace(pd.DataFrame(rows))

    problem = _load_problem(args)
    for spec in TRACE_SPECS:
        source = fair_results[
            _lambda_mask(fair_results, args.lambda_label)
            & (fair_results["method"] == spec.method)
            & (fair_results["batch"] == spec.batch)
        ].sort_values("iteration")
        log_rounds = [int(v) for v in source["iteration"].tolist()]
        if not log_rounds:
            continue
        value_trace = _replay_values(problem, args, spec.batch, log_rounds)
        merged = source.merge(value_trace, on="iteration", how="left")
        for row in merged.itertuples(index=False):
            dual_value = -float(row.dual_min_objective)
            rows.append(
                {
                    "method": spec.method,
                    "batch": spec.batch,
                    "label": spec.label,
                    "color": spec.color,
                    "iteration": int(row.iteration),
                    "time_ms": int(row.iteration) * round_ms[spec.batch],
                    "test_error": float(row.test_error),
                    "raw_duality_gap": max(float(row.objective_gap), 1e-12),
                    "dual_value": dual_value,
                    "primal_value": max(float(row.primal_objective), 1e-12),
                }
            )
    return _finish_trace(pd.DataFrame(rows))


def _finish_trace(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return trace
    trace = trace.sort_values(["batch", "iteration"]).copy()
    trace["best_test_error"] = trace.groupby("batch")["test_error"].cummin()
    trace["best_duality_gap"] = trace.groupby("batch")["raw_duality_gap"].cummin()
    trace["best_primal_value"] = trace.groupby("batch")["primal_value"].cummin()
    return trace


def _lambda_mask(frame: pd.DataFrame, label: str) -> pd.Series:
    if "lambda_label" in frame:
        mask = frame["lambda_label"].astype(str) == label
        if mask.any():
            return mask
    if "lambda" in frame:
        try:
            value = float(label)
        except ValueError:
            return pd.Series(False, index=frame.index)
        return np.isclose(frame["lambda"].astype(float), value)
    return pd.Series(False, index=frame.index)


def _load_problem(args: argparse.Namespace) -> StructuralSequenceSVMProblem:
    train_x, train_y, test_x, test_y = load_taskar_ocr(args.data_dir, "ocr2")
    total = args.agents * args.blocks
    train_x = train_x[:total]
    train_y = train_y[:total]
    reg = 1.0 / len(train_x) if args.lambda_label in {"1/n", "inv_n"} else float(args.lambda_label)
    x_parts = [list(part) for part in np.array_split(np.asarray(train_x, dtype=object), args.agents)]
    y_parts = [list(part) for part in np.array_split(np.asarray(train_y, dtype=object), args.agents)]
    return StructuralSequenceSVMProblem(
        x_parts,
        y_parts,
        reg,
        classes=26,
        position_bias=True,
        test_x=test_x,
        test_y=test_y,
    )


def _replay_values(
    problem: StructuralSequenceSVMProblem,
    args: argparse.Namespace,
    batch: int,
    log_rounds: list[int],
) -> pd.DataFrame:
    target_rounds = set(log_rounds)
    max_round = max(log_rounds)
    rng = np.random.default_rng(args.seed + 1207)
    graph_seq = GraphSequence(
        args.agents,
        GraphConfig("erdos", args.edge_prob, seed=args.graph_seed),
    )

    block_w = np.zeros((args.agents, problem.block_count, problem.dim), dtype=float)
    block_ell = np.zeros((args.agents, problem.block_count), dtype=float)
    local_w = np.zeros((args.agents, problem.dim), dtype=float)
    points = np.zeros((args.agents, problem.dim), dtype=float)
    rows = [_value_row(problem, 0, local_w, block_ell, points)] if 0 in target_rounds else []

    for iteration in range(1, max_round + 1):
        weights, _ = graph_seq.next()
        mixed = weights @ points
        next_points = mixed.copy()
        for agent in range(args.agents):
            selected = _selected_blocks(rng, problem.block_count, batch)
            delta = _agent_update(problem, agent, selected, mixed[agent], block_w, block_ell)
            local_w[agent] += delta
            next_points[agent] += args.agents * delta
        points = next_points
        if iteration in target_rounds:
            rows.append(_value_row(problem, iteration, local_w, block_ell, points))
    return pd.DataFrame(rows)


def _agent_update(
    problem: StructuralSequenceSVMProblem,
    agent: int,
    selected: np.ndarray,
    mixed_point: np.ndarray,
    block_w: np.ndarray,
    block_ell: np.ndarray,
) -> np.ndarray:
    old_w = block_w[agent, selected].sum(axis=0)
    old_ell = float(block_ell[agent, selected].sum())
    target_blocks = np.zeros((len(selected), problem.dim), dtype=float)
    target_ells = np.zeros(len(selected), dtype=float)
    for pos, block in enumerate(selected):
        vertex, ell_s, _, _ = problem.oracle_vertex(agent, int(block), mixed_point)
        target_blocks[pos] = vertex
        target_ells[pos] = ell_s
    target_w = target_blocks.sum(axis=0)
    target_ell = float(target_ells.sum())
    gamma = _line_search_gamma(problem.reg, old_w, old_ell, target_w, target_ell, mixed_point)
    block_w[agent, selected] = (1.0 - gamma) * block_w[agent, selected] + gamma * target_blocks
    block_ell[agent, selected] = (1.0 - gamma) * block_ell[agent, selected] + gamma * target_ells
    return gamma * (target_w - old_w)


def _value_row(
    problem: StructuralSequenceSVMProblem,
    iteration: int,
    local_w: np.ndarray,
    block_ell: np.ndarray,
    points: np.ndarray,
) -> dict[str, float | int]:
    w_global = local_w.sum(axis=0)
    ell = float(block_ell.sum())
    dual_min_objective = 0.5 * problem.reg * float(w_global @ w_global) - ell
    primal_objective = problem.objective(points.mean(axis=0))
    return {
        "iteration": int(iteration),
        "dual_min_objective": float(dual_min_objective),
        "primal_objective": float(primal_objective),
    }


def _selected_blocks(rng: np.random.Generator, block_count: int, batch: int) -> np.ndarray:
    if batch >= block_count:
        return np.arange(block_count)
    return rng.choice(block_count, size=batch, replace=False)


def _plot_time_panels(trace: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0), dpi=220, sharex=True)
    for spec in TRACE_SPECS:
        df = trace[trace["batch"] == spec.batch].sort_values("time_ms")
        if df.empty:
            continue
        markevery = max(len(df) // 18, 1)
        axes[0].plot(
            df["time_ms"],
            100.0 * df["best_test_error"],
            marker="o",
            markevery=markevery,
            markersize=3.0,
            linewidth=1.5,
            color=spec.color,
            label=spec.label,
        )
        axes[1].plot(
            df["time_ms"],
            df["best_duality_gap"],
            marker="o",
            markevery=markevery,
            markersize=3.0,
            linewidth=1.5,
            color=spec.color,
        )
        axes[2].plot(
            df["time_ms"],
            df["best_primal_value"],
            marker="o",
            markevery=markevery,
            markersize=3.0,
            linewidth=1.5,
            color=spec.color,
        )

    axes[0].set_ylabel("best OCR test error (%)")
    axes[1].set_ylabel("best FW duality gap")
    axes[2].set_ylabel("best train primal value")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1.0)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("cumulative round-time proxy (ms)")
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
