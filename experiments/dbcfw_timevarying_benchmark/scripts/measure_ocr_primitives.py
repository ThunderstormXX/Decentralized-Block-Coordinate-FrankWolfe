from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dbcfw_bench.data_ocr import load_taskar_ocr
from dbcfw_bench.objective_structural_svm import StructuralSequenceSVMProblem


def main() -> None:
    args = _parser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_x, train_y, test_x, test_y = load_taskar_ocr(args.data_dir, "ocr2")
    total = args.agents * args.blocks
    train_x = train_x[:total]
    train_y = train_y[:total]
    reg = 1.0 / len(train_x) if args.reg == "1/n" else float(args.reg)
    x_parts = [list(part) for part in np.array_split(np.asarray(train_x, dtype=object), args.agents)]
    y_parts = [list(part) for part in np.array_split(np.asarray(train_y, dtype=object), args.agents)]
    problem = StructuralSequenceSVMProblem(
        x_parts,
        y_parts,
        reg,
        classes=26,
        position_bias=True,
        test_x=test_x,
        test_y=test_y,
    )

    rng = np.random.default_rng(args.seed)
    w = rng.normal(0.0, 0.01, size=problem.dim)

    rows = []
    rows.append(_measure_score(problem, w, args.samples, rng))
    rows.append(_measure_decode(problem, w, args.samples, rng, loss_augmented=False))
    rows.append(_measure_decode(problem, w, args.samples, rng, loss_augmented=True))
    rows.append(_measure_local_full_lmo(problem, w, args.repeats))
    rows.append(_measure_global_full_lmo(problem, w, args.repeats))

    frame = pd.DataFrame(rows)
    path = out / "primitive_times.csv"
    frame.to_csv(path, index=False)
    md = out / "primitive_times.md"
    md.write_text(
        "\n".join(
            [
                "# OCR Structural SVM primitive timings",
                "",
                f"- agents: {args.agents}",
                f"- blocks per agent: {problem.block_count}",
                f"- model dimension: {problem.dim}",
                f"- sampled words for per-word primitives: {args.samples}",
                "",
                frame.to_markdown(index=False, floatfmt=".4g"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {path}")
    print(f"wrote {md}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs_paper_ocr/ocr_primitive_timing"))
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=893)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1207)
    parser.add_argument("--reg", type=str, default="1/n")
    return parser


def _random_word(problem: StructuralSequenceSVMProblem, rng: np.random.Generator) -> tuple[int, int]:
    agent = int(rng.integers(0, problem.agents))
    block = int(rng.integers(0, problem.block_count))
    return agent, block


def _measure_score(
    problem: StructuralSequenceSVMProblem,
    w: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float | str | int]:
    times = []
    for _ in range(samples):
        agent, block = _random_word(problem, rng)
        x_seq = problem.x_parts[agent][block]
        y_seq = problem.y_parts[agent][block]
        t0 = time.perf_counter()
        score = float(w @ problem.feature_map(x_seq, y_seq))
        if not np.isfinite(score):
            raise RuntimeError("non-finite score")
        times.append(time.perf_counter() - t0)
    return _row("fixed_label_score", samples, times)


def _measure_decode(
    problem: StructuralSequenceSVMProblem,
    w: np.ndarray,
    samples: int,
    rng: np.random.Generator,
    *,
    loss_augmented: bool,
) -> dict[str, float | str | int]:
    times = []
    for _ in range(samples):
        agent, block = _random_word(problem, rng)
        t0 = time.perf_counter()
        if loss_augmented:
            problem.oracle_vertex(agent, block, w)
        else:
            problem.decode(problem.x_parts[agent][block], w)
        times.append(time.perf_counter() - t0)
    return _row("loss_augmented_block_lmo" if loss_augmented else "prediction_viterbi", samples, times)


def _measure_local_full_lmo(
    problem: StructuralSequenceSVMProblem,
    w: np.ndarray,
    repeats: int,
) -> dict[str, float | str | int]:
    times = []
    calls = problem.block_count
    for repeat in range(repeats):
        agent = repeat % problem.agents
        t0 = time.perf_counter()
        for block in range(problem.block_count):
            problem.oracle_vertex(agent, block, w)
        times.append(time.perf_counter() - t0)
    return _row("local_full_lmo_one_agent", calls, times)


def _measure_global_full_lmo(
    problem: StructuralSequenceSVMProblem,
    w: np.ndarray,
    repeats: int,
) -> dict[str, float | str | int]:
    times = []
    calls = problem.agents * problem.block_count
    for _ in range(repeats):
        t0 = time.perf_counter()
        problem.full_oracle(w)
        times.append(time.perf_counter() - t0)
    return _row("global_full_lmo_all_agents", calls, times)


def _row(name: str, calls_or_samples: int, times: list[float]) -> dict[str, float | str | int]:
    values = np.asarray(times, dtype=float)
    return {
        "primitive": name,
        "calls_or_samples": int(calls_or_samples),
        "mean_ms": 1000.0 * float(values.mean()),
        "median_ms": 1000.0 * float(np.median(values)),
        "p90_ms": 1000.0 * float(np.percentile(values, 90)),
        "p99_ms": 1000.0 * float(np.percentile(values, 99)),
    }


if __name__ == "__main__":
    main()
