from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dbcfw_bench.comm_graphs import GraphConfig
from dbcfw_bench.config import RunConfig
from dbcfw_bench.data_ocr import load_taskar_ocr
from dbcfw_bench.algorithms.structural_svm import run_structural_svm_fw
from dbcfw_bench.objective_structural_svm import StructuralSequenceSVMProblem


@dataclass
class SolverState:
    w: np.ndarray
    ell: float


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_x, train_y, test_x, test_y = load_taskar_ocr(args.data_dir, "ocr2")
    if args.max_train:
        train_x = train_x[: args.max_train]
        train_y = train_y[: args.max_train]
    lambdas = [_lambda_value(value, len(train_x)) for value in args.lambdas]
    frames = []
    if "central" in args.mode:
        for lambd in lambdas:
            problem = _problem([train_x], [train_y], test_x, test_y, lambd)
            if "bcfw" in args.methods:
                frames.append(run_bcfw(problem, lambd, args.passes, args.seed, args.log_every))
            if "fw" in args.methods:
                frames.append(run_fw(problem, lambd, args.passes, args.log_every))
    if "decentralized" in args.mode:
        for lambd in lambdas:
            for method in ["dfw", "dbcfw"]:
                for batch in _batches(args, method):
                    problem = _decentralized_problem(
                        train_x, train_y, test_x, test_y, lambd, args.agents, args.blocks
                    )
                    cfg = RunConfig(
                        objective="ocr_structural_svm",
                        method=method,
                        agents=args.agents,
                        dim=problem.dim,
                        blocks=args.blocks,
                        batch=batch,
                        iters=args.decentralized_iters,
                        reg=lambd,
                        lmo="simplex",
                        graph=args.graph,
                        edge_prob=args.edge_prob,
                        graph_seed=args.graph_seed,
                        seed=args.seed,
                        log_every=args.decentralized_log_every,
                        data_dir=str(args.data_dir),
                    )
                    graph_seed = cfg.graph_seed if cfg.graph_seed is not None else cfg.seed
                    rows = run_structural_svm_fw(
                        problem,
                        cfg,
                        GraphConfig(cfg.graph, cfg.edge_prob, cfg.geometric_radius, graph_seed),
                        0.0,
                    )
                    frame = pd.DataFrame([row.to_dict() for row in rows])
                    frame["lambda"] = lambd
                    frame["paper_family"] = "ocr2"
                    frame["solver"] = f"decentralized_{method}_B{batch}"
                    frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    best_dual = result.groupby("lambda")["dual"].transform("max") if "dual" in result else np.nan
    if "primal" in result:
        result["primal_suboptimality"] = result["primal"] - best_dual
    path = out / "results.csv"
    result.to_csv(path, index=False)
    plot_paths = plot_paper_ocr(result, out / "plots")
    print(f"wrote {path} with {len(result)} rows")
    print("wrote " + ", ".join(str(path) for path in plot_paths))


def run_bcfw(
    problem: StructuralSequenceSVMProblem,
    lambd: float,
    passes: int,
    seed: int,
    log_every: int,
) -> pd.DataFrame:
    n, dim = problem.total_examples, problem.dim
    rng = np.random.default_rng(seed)
    w = np.zeros(dim, dtype=float)
    w_blocks = np.zeros((n, dim), dtype=float)
    ell = 0.0
    ell_blocks = np.zeros(n, dtype=float)
    w_avg = w.copy()
    ell_avg = 0.0
    rows = [_central_row(problem, "bcfw", lambd, 0.0, 0, w, ell, w_avg, ell_avg)]
    for k in range(passes * n):
        block = int(rng.integers(0, n))
        vertex, ell_s, _, _ = problem.oracle_vertex(0, block, w)
        old_w = w_blocks[block].copy()
        old_ell = float(ell_blocks[block])
        diff = old_w - vertex
        denom = float(diff @ diff) + np.finfo(float).eps
        gamma = float(np.clip((w @ diff - (old_ell - ell_s) / lambd) / denom, 0.0, 1.0))
        w -= old_w
        w_blocks[block] = (1.0 - gamma) * old_w + gamma * vertex
        w += w_blocks[block]
        ell -= old_ell
        ell_blocks[block] = (1.0 - gamma) * old_ell + gamma * ell_s
        ell += ell_blocks[block]
        rho = 2.0 / (k + 2.0)
        w_avg = (1.0 - rho) * w_avg + rho * w
        ell_avg = (1.0 - rho) * ell_avg + rho * ell
        iteration = k + 1
        if iteration % (log_every * n) == 0:
            eff_pass = iteration / n
            rows.append(_central_row(problem, "bcfw", lambd, eff_pass, iteration, w, ell, w_avg, ell_avg))
    return pd.DataFrame(rows)


def run_fw(
    problem: StructuralSequenceSVMProblem,
    lambd: float,
    passes: int,
    log_every: int,
) -> pd.DataFrame:
    del log_every
    w = np.zeros(problem.dim, dtype=float)
    ell = 0.0
    rows = [_central_row(problem, "fw", lambd, 0.0, 0, w, ell, None, None)]
    for k in range(passes):
        w_s, ell_s, _ = problem.full_oracle(w)
        gap = lambd * float(w @ (w - w_s)) - ell + ell_s
        denom = lambd * float((w - w_s) @ (w - w_s)) + np.finfo(float).eps
        gamma = float(np.clip(gap / denom, 0.0, 1.0))
        w = (1.0 - gamma) * w + gamma * w_s
        ell = (1.0 - gamma) * ell + gamma * ell_s
        rows.append(_central_row(problem, "fw", lambd, float(k + 1), (k + 1) * problem.total_examples, w, ell, None, None))
    return pd.DataFrame(rows)


def _central_row(
    problem: StructuralSequenceSVMProblem,
    method: str,
    lambd: float,
    eff_pass: float,
    oracle_calls: int,
    w: np.ndarray,
    ell: float,
    w_avg: np.ndarray | None,
    ell_avg: float | None,
) -> dict[str, float | int | str]:
    state = SolverState(w_avg, float(ell_avg)) if w_avg is not None and ell_avg is not None else SolverState(w, ell)
    gap, gap_calls = problem.duality_gap(state.w, state.ell)
    dual = state.ell - 0.5 * lambd * float(state.w @ state.w)
    primal = dual + gap
    rows = {
        "paper_family": "ocr2",
        "method": method,
        "solver": method,
        "lambda": lambd,
        "iteration": oracle_calls,
        "effective_passes": eff_pass,
        "oracle_calls": oracle_calls + gap_calls,
        "primal": primal,
        "dual": dual,
        "duality_gap": gap,
        "objective_gap": gap,
        "train_error": problem.average_sequence_loss(list(problem.x_parts[0]), list(problem.y_parts[0]), state.w),
        "test_error": problem.test_error(state.w),
        "dim": problem.dim,
        "train_examples": problem.total_examples,
    }
    return rows


def plot_paper_ocr(frame: pd.DataFrame, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    if _has_metric(frame, "primal_suboptimality"):
        paths.append(_plot_metric(
            frame, "effective_passes", "primal_suboptimality",
            out / "ocr_primal_suboptimality.png", logy=True,
        ))
    if _has_metric(frame, "objective_gap"):
        paths.append(_plot_metric(
            frame, "effective_passes", "objective_gap",
            out / "ocr_objective_gap.png", logy=True,
        ))
    if _has_metric(frame, "test_error"):
        paths.append(_plot_metric(frame, "effective_passes", "test_error", out / "ocr_test_error.png", logy=False))
    if _has_metric(frame, "duality_gap"):
        paths.append(_plot_metric(frame, "effective_passes", "duality_gap", out / "ocr_duality_gap.png", logy=True))
    return paths


def _has_metric(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and frame[column].notna().any()


def _plot_metric(frame: pd.DataFrame, x_col: str, y_col: str, path: Path, logy: bool) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_name = x_col if x_col in frame.columns else "iteration"
    for (lambd, solver), group in frame.groupby(["lambda", "solver"], sort=False):
        if y_col not in group:
            continue
        data = group.sort_values(x_name)
        y = data[y_col].clip(lower=1e-12) if logy else data[y_col]
        ax.plot(data[x_name], y, marker="o", markersize=2.5, linewidth=1.2, label=f"{solver}, lambda={lambd:g}")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_col)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _problem(
    x_parts: list[list[np.ndarray]],
    y_parts: list[list[np.ndarray]],
    test_x: list[np.ndarray],
    test_y: list[np.ndarray],
    lambd: float,
) -> StructuralSequenceSVMProblem:
    return StructuralSequenceSVMProblem(x_parts, y_parts, lambd, 26, True, test_x, test_y)


def _decentralized_problem(
    train_x: list[np.ndarray],
    train_y: list[np.ndarray],
    test_x: list[np.ndarray],
    test_y: list[np.ndarray],
    lambd: float,
    agents: int,
    blocks: int,
) -> StructuralSequenceSVMProblem:
    total = agents * blocks
    if total > len(train_x):
        raise ValueError(f"requested {total} examples, only {len(train_x)} available")
    x_chunks = [train_x[i * blocks : (i + 1) * blocks] for i in range(agents)]
    y_chunks = [train_y[i * blocks : (i + 1) * blocks] for i in range(agents)]
    return _problem(x_chunks, y_chunks, test_x, test_y, lambd)


def _lambda_value(value: str, n: int) -> float:
    return 1.0 / n if value in {"1/n", "inv_n"} else float(value)


def _batches(args: argparse.Namespace, method: str) -> list[int]:
    if method == "dfw":
        return [args.blocks]
    return args.batches


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dbcfw_bench.paper_ocr")
    parser.add_argument("--mode", nargs="+", choices=["central", "decentralized"], default=["central", "decentralized"])
    parser.add_argument("--methods", nargs="+", choices=["bcfw", "fw"], default=["bcfw", "fw"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--lambdas", nargs="+", default=["0.01", "0.001", "1/n"])
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=893)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 10, 89])
    parser.add_argument("--decentralized-iters", type=int, default=20)
    parser.add_argument("--decentralized-log-every", type=int, default=5)
    parser.add_argument("--graph", default="erdos")
    parser.add_argument("--edge-prob", type=float, default=0.7)
    parser.add_argument("--graph-seed", type=int, default=2207)
    return parser


if __name__ == "__main__":
    main()
