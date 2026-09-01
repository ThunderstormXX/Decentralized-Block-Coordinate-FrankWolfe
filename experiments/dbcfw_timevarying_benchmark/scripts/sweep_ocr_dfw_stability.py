from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dbcfw_bench.algorithms.structural_svm import _line_search_gamma
from dbcfw_bench.comm_graphs import GraphConfig, GraphSequence
from dbcfw_bench.data_ocr import load_taskar_ocr
from dbcfw_bench.metrics import consensus_error
from dbcfw_bench.objective_structural_svm import StructuralSequenceSVMProblem


@dataclass(frozen=True)
class StepSpec:
    rule: str
    scale: float
    cap: float
    offset: float

    @property
    def label(self) -> str:
        if self.rule == "decay":
            return f"decay_offset_{self.offset:g}"
        parts = [f"{self.rule}_scale_{self.scale:g}"]
        if self.cap < 1.0:
            parts.append(f"cap_{self.cap:g}")
        return "_".join(parts)


def main() -> None:
    args = _parser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_x, train_y, test_x, test_y = load_taskar_ocr(args.data_dir, "ocr2")
    total = args.agents * args.blocks
    train_x = train_x[:total]
    train_y = train_y[:total]

    rows = []
    for lambda_label in args.lambdas:
        lambd = _lambda_value(lambda_label, len(train_x))
        problem = _make_problem(train_x, train_y, test_x, test_y, lambd, args.agents, args.blocks)
        batch_iters = _batch_iters(args)
        for batch in args.batches:
            for spec in _step_specs(args):
                iters = batch_iters.get(batch, args.iters)
                print(
                    f"running lambda={lambda_label} ({lambd:g}), B={batch}, "
                    f"iters={iters}, {spec.label}",
                    flush=True,
                )
                rows.extend(_run_method(problem, args, lambda_label, lambd, spec, batch, iters))

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "dfw_stability_sweep.csv", index=False)
    summary = _summarize(frame)
    summary.to_csv(out / "dfw_stability_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"wrote {out / 'dfw_stability_sweep.csv'}")
    print(f"wrote {out / 'dfw_stability_summary.csv'}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=893)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--batches", nargs="+", type=int, default=[893])
    parser.add_argument("--batch-iters", nargs="*", default=[])
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--graph-seed", type=int, default=2207)
    parser.add_argument("--edge-prob", type=float, default=0.35)
    parser.add_argument("--lambdas", nargs="+", default=["1/n", "0.001", "0.01"])
    parser.add_argument("--rules", nargs="+", default=["local_ls"])
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.3, 0.1, 0.03, 0.01])
    parser.add_argument("--caps", nargs="+", type=float, default=[1.0])
    parser.add_argument("--decay-offsets", nargs="+", type=float, default=[100.0, 1000.0])
    parser.add_argument("--eval-primal", action="store_true")
    return parser


def _make_problem(
    train_x: list[np.ndarray],
    train_y: list[np.ndarray],
    test_x: list[np.ndarray],
    test_y: list[np.ndarray],
    lambd: float,
    agents: int,
    blocks: int,
) -> StructuralSequenceSVMProblem:
    x_parts = [train_x[i * blocks : (i + 1) * blocks] for i in range(agents)]
    y_parts = [train_y[i * blocks : (i + 1) * blocks] for i in range(agents)]
    return StructuralSequenceSVMProblem(x_parts, y_parts, lambd, 26, True, test_x, test_y)


def _step_specs(args: argparse.Namespace) -> list[StepSpec]:
    specs: list[StepSpec] = []
    for rule in args.rules:
        if rule == "decay":
            specs.extend(StepSpec(rule, 1.0, 1.0, offset) for offset in args.decay_offsets)
        else:
            for scale in args.scales:
                for cap in args.caps:
                    specs.append(StepSpec(rule, scale, cap, 0.0))
    return specs


def _batch_iters(args: argparse.Namespace) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for item in args.batch_iters:
        batch, iters = item.split(":", 1)
        parsed[int(batch)] = int(iters)
    return parsed


def _run_method(
    problem: StructuralSequenceSVMProblem,
    args: argparse.Namespace,
    lambda_label: str,
    lambd: float,
    spec: StepSpec,
    batch: int,
    iters: int,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(args.seed + 1207)
    graph_seq = GraphSequence(
        args.agents,
        GraphConfig("erdos", args.edge_prob, seed=args.graph_seed),
    )
    block_w = np.zeros((args.agents, problem.block_count, problem.dim), dtype=float)
    block_ell = np.zeros((args.agents, problem.block_count), dtype=float)
    local_w = np.zeros((args.agents, problem.dim), dtype=float)
    points = np.zeros((args.agents, problem.dim), dtype=float)
    rows = []
    rows.append(_row(problem, args, lambda_label, lambd, spec, batch, 0, np.nan, np.nan, points, local_w, block_ell))

    for iteration in range(1, iters + 1):
        weights, _ = graph_seq.next()
        mixed = weights @ points
        next_points = mixed.copy()
        local_updates = []
        raw_gammas = []
        gammas = []
        for agent in range(args.agents):
            selected = _selected_blocks(rng, problem.block_count, batch)
            raw_gamma, delta, new_blocks, new_ells = _agent_candidate(
                problem, agent, selected, mixed[agent], block_w, block_ell
            )
            local_updates.append((agent, selected, delta, new_blocks, new_ells, raw_gamma))
            raw_gammas.append(raw_gamma)

        if spec.rule == "global_ls":
            gamma = _global_line_search(problem, local_updates, block_w, block_ell, local_w)
            gamma = _damp_gamma(gamma, spec)
            gammas = [gamma] * args.agents
        elif spec.rule == "decay":
            gamma = min(1.0, 2.0 / (iteration + spec.offset))
            gammas = [gamma] * args.agents
        else:
            gammas = [_damp_gamma(raw, spec) for raw in raw_gammas]

        for (agent, blocks, _, new_blocks, new_ells, _), gamma in zip(local_updates, gammas):
            old_blocks = block_w[agent, blocks].copy()
            old_ells = block_ell[agent, blocks].copy()
            block_w[agent, blocks] = (1.0 - gamma) * old_blocks + gamma * new_blocks
            block_ell[agent, blocks] = (1.0 - gamma) * old_ells + gamma * new_ells
            delta = gamma * (new_blocks.sum(axis=0) - old_blocks.sum(axis=0))
            local_w[agent] += delta
            next_points[agent] += args.agents * delta

        points = next_points
        if iteration % args.log_every == 0 or iteration == iters:
            rows.append(
                _row(
                    problem,
                    args,
                    lambda_label,
                    lambd,
                    spec,
                    batch,
                    iteration,
                    float(np.mean(raw_gammas)),
                    float(np.mean(gammas)),
                    points,
                    local_w,
                    block_ell,
                )
            )
    return rows


def _agent_candidate(
    problem: StructuralSequenceSVMProblem,
    agent: int,
    selected: np.ndarray,
    mixed_point: np.ndarray,
    block_w: np.ndarray,
    block_ell: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
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
    raw_gamma = _line_search_gamma(problem.reg, old_w, old_ell, target_w, target_ell, mixed_point)
    return raw_gamma, target_w - old_w, target_blocks, target_ells


def _global_line_search(
    problem: StructuralSequenceSVMProblem,
    local_updates: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
    block_w: np.ndarray,
    block_ell: np.ndarray,
    local_w: np.ndarray,
) -> float:
    model = local_w.sum(axis=0)
    old_selected_w = np.zeros(problem.dim, dtype=float)
    target_selected_w = np.zeros(problem.dim, dtype=float)
    old_selected_ell = 0.0
    target_selected_ell = 0.0
    for agent, selected, _, new_blocks, new_ells, _ in local_updates:
        old_selected_w += block_w[agent, selected].sum(axis=0)
        target_selected_w += new_blocks.sum(axis=0)
        old_selected_ell += float(block_ell[agent, selected].sum())
        target_selected_ell += float(new_ells.sum())
    diff = old_selected_w - target_selected_w
    denom = problem.reg * float(diff @ diff)
    if denom <= 1e-18:
        return 0.0
    numerator = problem.reg * float(diff @ model) - old_selected_ell + target_selected_ell
    return float(np.clip(numerator / denom, 0.0, 1.0))


def _damp_gamma(raw_gamma: float, spec: StepSpec) -> float:
    return float(np.clip(spec.scale * raw_gamma, 0.0, spec.cap))


def _selected_blocks(rng: np.random.Generator, block_count: int, batch: int) -> np.ndarray:
    if batch >= block_count:
        return np.arange(block_count)
    return rng.choice(block_count, size=batch, replace=False)


def _row(
    problem: StructuralSequenceSVMProblem,
    args: argparse.Namespace,
    lambda_label: str,
    lambd: float,
    spec: StepSpec,
    batch: int,
    iteration: int,
    raw_gamma: float,
    gamma: float,
    points: np.ndarray,
    local_w: np.ndarray,
    block_ell: np.ndarray,
) -> dict[str, float | int | str]:
    w_global = local_w.sum(axis=0)
    ell = float(block_ell.sum())
    gap, _ = problem.duality_gap(w_global, ell)
    w_avg = points.mean(axis=0)
    primal = problem.objective(w_avg) if args.eval_primal else float("nan")
    actual_batch = min(batch, problem.block_count)
    return {
        "lambda_label": lambda_label,
        "lambda": lambd,
        "method": "dfw" if actual_batch >= problem.block_count else "dbcfw",
        "batch": actual_batch,
        "step_rule": spec.rule,
        "gamma_scale": spec.scale,
        "gamma_cap": spec.cap,
        "gamma_offset": spec.offset,
        "step_label": spec.label,
        "iteration": iteration,
        "raw_gamma": raw_gamma,
        "gamma": gamma,
        "objective_gap": gap,
        "test_error": problem.test_error(w_avg),
        "train_primal": primal,
        "consensus_error": consensus_error(points),
        "training_oracle_calls": iteration * args.agents * actual_batch,
    }


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    keys = [
        "lambda_label",
        "lambda",
        "method",
        "batch",
        "step_rule",
        "gamma_scale",
        "gamma_cap",
        "gamma_offset",
        "step_label",
    ]
    for values, group in frame.groupby(keys, sort=False):
        data = group.sort_values("iteration")
        gap = data["objective_gap"]
        primal = data["train_primal"]
        out.append(
            dict(
                zip(keys, values),
                rows=len(data),
                first_gap=float(gap.iloc[0]),
                best_gap=float(gap.min()),
                last_gap=float(gap.iloc[-1]),
                last_over_best=float(gap.iloc[-1] / max(gap.min(), 1e-12)),
                best_test_error=float(data["test_error"].min()),
                last_test_error=float(data["test_error"].iloc[-1]),
                best_primal=float(primal.min()) if primal.notna().any() else float("nan"),
                last_consensus=float(data["consensus_error"].iloc[-1]),
                max_gamma=float(data["gamma"].max()),
                stable=bool(np.isfinite(gap.iloc[-1]) and gap.iloc[-1] < 10.0 * gap.iloc[0]),
            )
        )
    return pd.DataFrame(out).sort_values(["stable", "best_gap", "best_test_error"], ascending=[False, True, True])


def _lambda_value(value: str, n: int) -> float:
    return 1.0 / n if value in {"1/n", "inv_n"} else float(value)


if __name__ == "__main__":
    main()
