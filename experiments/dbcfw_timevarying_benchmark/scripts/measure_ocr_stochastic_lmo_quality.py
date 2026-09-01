from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dbcfw_bench.data_ocr import load_taskar_ocr
from dbcfw_bench.objective_structural_svm import StructuralSequenceSVMProblem


def main() -> None:
    args = _parser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    problem = _load_problem(args)
    batches = sorted({int(value) for value in args.batches if int(value) <= problem.block_count})
    if problem.block_count not in batches:
        batches.append(problem.block_count)

    rng = np.random.default_rng(args.seed)
    w = rng.normal(0.0, args.weight_scale, size=problem.dim)

    full_vertices, full_losses, full_gaps = _full_local_oracles(problem, w)
    rows: list[dict[str, float | int]] = []
    for batch in batches:
        rows.extend(_measure_batch(problem, rng, full_vertices, full_losses, full_gaps, batch, args.samples))

    frame = pd.DataFrame(rows)
    raw_path = out / "stochastic_lmo_quality.csv"
    frame.to_csv(raw_path, index=False)

    summary = _summarize(frame)
    summary_path = out / "stochastic_lmo_quality_summary.csv"
    summary.to_csv(summary_path, index=False)

    md_path = out / "stochastic_lmo_quality.md"
    md_path.write_text(
        "\n".join(
            [
                "# OCR Structural SVM stochastic LMO quality",
                "",
                f"- agents: {problem.agents}",
                f"- local word blocks per agent: {problem.block_count}",
                f"- model dimension: {problem.dim}",
                f"- samples per B: {args.samples}",
                "",
                "The sampled direction is scaled by `K / B`, where `K` is the local block count,",
                "so it is an unbiased estimator of the full local LMO aggregate direction.",
                "",
                summary.to_markdown(index=False, floatfmt=".4g"),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {md_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs_paper_ocr/ocr_stochastic_lmo_quality"))
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=893)
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1207)
    parser.add_argument("--reg", type=str, default="1/n")
    parser.add_argument("--weight-scale", type=float, default=0.01)
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 25, 50, 100, 200, 400],
    )
    return parser


def _load_problem(args: argparse.Namespace) -> StructuralSequenceSVMProblem:
    train_x, train_y, test_x, test_y = load_taskar_ocr(args.data_dir, "ocr2")
    total = args.agents * args.blocks
    train_x = train_x[:total]
    train_y = train_y[:total]
    reg = 1.0 / len(train_x) if args.reg == "1/n" else float(args.reg)
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


def _full_local_oracles(
    problem: StructuralSequenceSVMProblem,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.zeros((problem.agents, problem.block_count, problem.dim), dtype=float)
    losses = np.zeros((problem.agents, problem.block_count), dtype=float)
    gaps = np.zeros((problem.agents, problem.block_count), dtype=float)
    for agent in range(problem.agents):
        for block in range(problem.block_count):
            vertex, ell_s, _, _ = problem.oracle_vertex(agent, block, w)
            vertices[agent, block] = vertex
            losses[agent, block] = ell_s
            gaps[agent, block] = problem.reg * float(vertex @ w) - ell_s
    return vertices, losses, gaps


def _measure_batch(
    problem: StructuralSequenceSVMProblem,
    rng: np.random.Generator,
    full_vertices: np.ndarray,
    full_losses: np.ndarray,
    full_gaps: np.ndarray,
    batch: int,
    samples: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    block_count = problem.block_count
    for _ in range(samples):
        agent = int(rng.integers(0, problem.agents))
        if batch >= block_count:
            selected = np.arange(block_count)
        else:
            selected = rng.choice(block_count, size=batch, replace=False)
        scale = block_count / float(len(selected))

        full_direction = full_vertices[agent].sum(axis=0)
        full_loss = float(full_losses[agent].sum())
        full_gap = float(full_gaps[agent].sum())

        sampled_direction = scale * full_vertices[agent, selected].sum(axis=0)
        sampled_loss = scale * float(full_losses[agent, selected].sum())
        sampled_gap = scale * float(full_gaps[agent, selected].sum())

        full_norm = float(np.linalg.norm(full_direction))
        direction_error = float(np.linalg.norm(sampled_direction - full_direction) / max(full_norm, 1e-18))
        loss_error = abs(sampled_loss - full_loss) / max(abs(full_loss), 1e-18)
        gap_error = abs(sampled_gap - full_gap) / max(abs(full_gap), 1e-18)
        rows.append(
            {
                "batch": int(batch),
                "agent": agent,
                "relative_direction_error": direction_error,
                "relative_loss_error": float(loss_error),
                "relative_linear_objective_error": float(gap_error),
            }
        )
    return rows


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("batch", as_index=False).agg(
        samples=("batch", "count"),
        rel_dir_error_mean=("relative_direction_error", "mean"),
        rel_dir_error_p90=("relative_direction_error", lambda s: float(np.percentile(s, 90))),
        rel_loss_error_mean=("relative_loss_error", "mean"),
        rel_linear_obj_error_mean=("relative_linear_objective_error", "mean"),
    )


if __name__ == "__main__":
    main()
