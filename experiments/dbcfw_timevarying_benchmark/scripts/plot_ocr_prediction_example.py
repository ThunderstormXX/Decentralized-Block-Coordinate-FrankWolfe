from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dbcfw_bench.comm_graphs import GraphConfig, GraphSequence
from dbcfw_bench.data_ocr import load_taskar_ocr
from dbcfw_bench.objective_structural_svm import StructuralSequenceSVMProblem

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def main() -> None:
    args = _parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    problem = _load_problem(args)
    if args.model == "dbcfw":
        model = _replay_dbcfw(args, problem)
        method_lines = [
            "method=DBCFW",
            f"batch={args.batch}",
            f"rounds={args.rounds}",
            f"agents={args.agents}",
            f"blocks_per_agent={args.blocks}",
        ]
    else:
        model = _replay_central_bcfw_avg(args, problem)
        method_lines = [
            "method=central_bcfw_avg",
            f"updates={args.central_iters}",
            f"effective_passes={args.central_iters / problem.total_examples:.6f}",
        ]
    x_seq = problem.test_x[args.test_index]
    y_true = problem.test_y[args.test_index]
    y_pred = problem.decode(x_seq, model)
    _plot_prediction(x_seq, y_true, y_pred, args.out)

    metadata = args.out.with_suffix(".txt")
    metadata.write_text(
        "\n".join(
            [
                *method_lines,
                f"lambda={args.reg}",
                f"test_index={args.test_index}",
                f"true={_word(y_true)}",
                f"pred={_word(y_pred)}",
                f"sequence_error={np.mean(y_true != y_pred):.6f}",
                f"test_error={problem.test_error(model):.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    print(f"wrote {metadata}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs_paper_ocr/crm_figures_global_ls/ocr_model_prediction_example.png"))
    parser.add_argument("--model", choices=["dbcfw", "central-bcfw-avg"], default="dbcfw")
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=893)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=900)
    parser.add_argument("--central-iters", type=int, default=228018)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--graph-seed", type=int, default=2207)
    parser.add_argument("--edge-prob", type=float, default=0.35)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--test-index", type=int, default=2)
    return parser


def _load_problem(args: argparse.Namespace) -> StructuralSequenceSVMProblem:
    train_x, train_y, test_x, test_y = load_taskar_ocr(args.data_dir, "ocr2")
    total = args.agents * args.blocks
    train_x = train_x[:total]
    train_y = train_y[:total]
    if args.model == "central-bcfw-avg":
        x_parts = [train_x]
        y_parts = [train_y]
    else:
        x_parts = [train_x[i * args.blocks : (i + 1) * args.blocks] for i in range(args.agents)]
        y_parts = [train_y[i * args.blocks : (i + 1) * args.blocks] for i in range(args.agents)]
    return StructuralSequenceSVMProblem(
        x_parts,
        y_parts,
        args.reg,
        classes=26,
        position_bias=True,
        test_x=test_x,
        test_y=test_y,
    )


def _replay_dbcfw(args: argparse.Namespace, problem: StructuralSequenceSVMProblem) -> np.ndarray:
    rng = np.random.default_rng(args.seed + 1207)
    graph_seq = GraphSequence(
        args.agents,
        GraphConfig("erdos", args.edge_prob, seed=args.graph_seed),
    )
    block_w = np.zeros((args.agents, problem.block_count, problem.dim), dtype=float)
    block_ell = np.zeros((args.agents, problem.block_count), dtype=float)
    local_w = np.zeros((args.agents, problem.dim), dtype=float)
    points = np.zeros((args.agents, problem.dim), dtype=float)

    for _ in range(args.rounds):
        weights, _ = graph_seq.next()
        mixed = weights @ points
        next_points = mixed.copy()
        local_updates = []

        for agent in range(args.agents):
            selected = _selected_blocks(rng, problem.block_count, args.batch)
            target_blocks, target_ells = _oracle_blocks(problem, agent, selected, mixed[agent])
            local_updates.append((agent, selected, target_blocks, target_ells))

        gamma = _global_line_search(problem, local_updates, block_w, block_ell, local_w)

        for agent, selected, target_blocks, target_ells in local_updates:
            old_blocks = block_w[agent, selected].copy()
            old_ells = block_ell[agent, selected].copy()
            block_w[agent, selected] = (1.0 - gamma) * old_blocks + gamma * target_blocks
            block_ell[agent, selected] = (1.0 - gamma) * old_ells + gamma * target_ells
            delta = gamma * (target_blocks.sum(axis=0) - old_blocks.sum(axis=0))
            local_w[agent] += delta
            next_points[agent] += args.agents * delta

        points = next_points

    return points.mean(axis=0)


def _replay_central_bcfw_avg(args: argparse.Namespace, problem: StructuralSequenceSVMProblem) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    n = problem.total_examples
    w = np.zeros(problem.dim, dtype=float)
    w_blocks = np.zeros((n, problem.dim), dtype=float)
    ell_blocks = np.zeros(n, dtype=float)
    w_avg = w.copy()

    for iteration in range(args.central_iters):
        block = int(rng.integers(0, n))
        vertex, ell_s, _, _ = problem.oracle_vertex(0, block, w)
        old_w = w_blocks[block].copy()
        old_ell = float(ell_blocks[block])
        diff = old_w - vertex
        denom = float(diff @ diff) + np.finfo(float).eps
        gamma = float(np.clip((w @ diff - (old_ell - ell_s) / problem.reg) / denom, 0.0, 1.0))

        w -= old_w
        w_blocks[block] = (1.0 - gamma) * old_w + gamma * vertex
        w += w_blocks[block]
        ell_blocks[block] = (1.0 - gamma) * old_ell + gamma * ell_s

        rho = 2.0 / (iteration + 2.0)
        w_avg = (1.0 - rho) * w_avg + rho * w

    return w_avg


def _oracle_blocks(
    problem: StructuralSequenceSVMProblem,
    agent: int,
    selected: np.ndarray,
    model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_blocks = np.zeros((len(selected), problem.dim), dtype=float)
    target_ells = np.zeros(len(selected), dtype=float)
    for pos, block in enumerate(selected):
        vertex, ell_s, _, _ = problem.oracle_vertex(agent, int(block), model)
        target_blocks[pos] = vertex
        target_ells[pos] = ell_s
    return target_blocks, target_ells


def _global_line_search(
    problem: StructuralSequenceSVMProblem,
    local_updates: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    block_w: np.ndarray,
    block_ell: np.ndarray,
    local_w: np.ndarray,
) -> float:
    model = local_w.sum(axis=0)
    old_selected_w = np.zeros(problem.dim, dtype=float)
    target_selected_w = np.zeros(problem.dim, dtype=float)
    old_selected_ell = 0.0
    target_selected_ell = 0.0

    for agent, selected, target_blocks, target_ells in local_updates:
        old_selected_w += block_w[agent, selected].sum(axis=0)
        target_selected_w += target_blocks.sum(axis=0)
        old_selected_ell += float(block_ell[agent, selected].sum())
        target_selected_ell += float(target_ells.sum())

    diff = old_selected_w - target_selected_w
    denom = problem.reg * float(diff @ diff)
    if denom <= 1e-18:
        return 0.0
    numerator = problem.reg * float(diff @ model) - old_selected_ell + target_selected_ell
    return float(np.clip(numerator / denom, 0.0, 1.0))


def _plot_prediction(x_seq: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    length = len(y_true)
    fig = plt.figure(figsize=(7.2, 1.75), dpi=240)
    grid = fig.add_gridspec(3, length, height_ratios=[4.4, 0.7, 0.7], hspace=0.05, wspace=0.08)

    for token in range(length):
        image_ax = fig.add_subplot(grid[0, token])
        patch = x_seq[token, :128].reshape(16, 8)
        image_ax.imshow(patch, cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        mismatch = y_true[token] != y_pred[token]
        spine_color = "#b23a2e" if mismatch else "#777777"
        for spine in image_ax.spines.values():
            spine.set_linewidth(1.4 if mismatch else 0.6)
            spine.set_color(spine_color)

        true_ax = fig.add_subplot(grid[1, token])
        pred_ax = fig.add_subplot(grid[2, token])
        for ax in (true_ax, pred_ax):
            ax.set_axis_off()
        true_ax.text(0.5, 0.5, LETTERS[int(y_true[token])], ha="center", va="center", fontsize=8)
        pred_ax.text(
            0.5,
            0.5,
            LETTERS[int(y_pred[token])],
            ha="center",
            va="center",
            fontsize=8,
            color="#b23a2e" if mismatch else "black",
            fontweight="bold" if mismatch else "normal",
        )

    fig.text(0.025, 0.345, "true", ha="left", va="center", fontsize=8)
    fig.text(0.025, 0.155, "pred", ha="left", va="center", fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.98, bottom=0.07)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _selected_blocks(rng: np.random.Generator, block_count: int, batch: int) -> np.ndarray:
    if batch >= block_count:
        return np.arange(block_count)
    return rng.choice(block_count, size=batch, replace=False)


def _word(labels: np.ndarray) -> str:
    return "".join(LETTERS[int(label)] for label in labels)


if __name__ == "__main__":
    main()
