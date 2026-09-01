from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import yaml

from dbcfw_bench.comm_graphs import GraphConfig, GraphSequence
from dbcfw_bench.config import graph_name


@dataclass
class OTRunConfig:
    methods: tuple[str, ...] = ("dfw", "dbcfw")
    m: int = 28
    n: int = 28
    agents: int = 8
    epochs: int = 80
    batch: int = 1
    relaxation: float = 0.08
    cost_noise: float = 0.03
    stepsize: str = "line_search"
    graph: str = "erdos"
    edge_prob: float = 0.45
    geometric_radius: float = 0.55
    seed: int = 42
    graph_seed: int | None = None
    log_every: int = 5


@dataclass
class OTPaperConfig:
    m: int = 28
    n: int = 28
    agents: int = 8
    epochs: int = 80
    batch: int = 1
    relaxations: tuple[float, ...] = (0.02, 0.04, 0.08, 0.16, 0.32)
    convergence_relaxation: float = 0.08
    transition_relaxations: tuple[float, float] = (0.02, 0.32)
    cost_noise: float = 0.03
    stepsize: str = "line_search"
    graph: str = "erdos"
    edge_prob: float = 0.45
    geometric_radius: float = 0.55
    seed: int = 42
    graph_seed: int | None = None
    log_every: int = 5


@dataclass
class SemiRelaxedOTProblem:
    source_weights: np.ndarray
    target_weights: np.ndarray
    source_points: np.ndarray
    target_points: np.ndarray
    local_costs: np.ndarray
    relaxation: float

    @property
    def agents(self) -> int:
        return int(self.local_costs.shape[0])

    @property
    def m(self) -> int:
        return int(self.source_weights.size)

    @property
    def n(self) -> int:
        return int(self.target_weights.size)

    @property
    def cost(self) -> np.ndarray:
        return self.local_costs.mean(axis=0)

    def initial_plan(self) -> np.ndarray:
        plan = np.zeros((self.m, self.n), dtype=float)
        plan[0, :] = self.target_weights
        return plan

    def objective(self, plan: np.ndarray) -> float:
        residual = row_residual(plan, self.source_weights)
        return float(np.sum(plan * self.cost) + 0.5 / self.relaxation * residual @ residual)

    def local_objective(self, agent: int, plan: np.ndarray) -> float:
        residual = row_residual(plan, self.source_weights)
        return float(
            np.sum(plan * self.local_costs[agent])
            + 0.5 / self.relaxation * residual @ residual
        )

    def gradient(self, plan: np.ndarray) -> np.ndarray:
        residual = row_residual(plan, self.source_weights)
        return self.cost + residual[:, None] / self.relaxation

    def local_gradient(self, agent: int, plan: np.ndarray) -> np.ndarray:
        residual = row_residual(plan, self.source_weights)
        return self.local_costs[agent] + residual[:, None] / self.relaxation

    def local_gradients(self, plans: np.ndarray) -> np.ndarray:
        out = np.empty_like(plans)
        for agent in range(self.agents):
            out[agent] = self.local_gradient(agent, plans[agent])
        return out

    def lmo(self, gradient: np.ndarray) -> np.ndarray:
        plan = np.zeros_like(gradient)
        rows = np.argmin(gradient, axis=0)
        plan[rows, np.arange(self.n)] = self.target_weights
        return plan

    def block_lmo(self, gradient_column: np.ndarray, column: int) -> np.ndarray:
        atom = np.zeros(self.m, dtype=float)
        atom[int(np.argmin(gradient_column))] = self.target_weights[column]
        return atom

    def duality_gap(self, plan: np.ndarray) -> float:
        gradient = self.gradient(plan)
        atom = self.lmo(gradient)
        return float(np.sum((plan - atom) * gradient))


def make_semirelaxed_ot_problem(config: OTRunConfig) -> SemiRelaxedOTProblem:
    rng = np.random.default_rng(config.seed)
    source_points = _curve_points(config.m, rng, phase=0.0, jitter=0.015)
    target_points = _curve_points(config.n, rng, phase=0.09, jitter=0.015)
    source_weights = _smooth_weights(config.m, rng)
    target_weights = _smooth_weights(config.n, rng)
    base_cost = _squared_cost(source_points, target_points)
    local_costs = np.empty((config.agents, config.m, config.n), dtype=float)
    for agent in range(config.agents):
        perturbation = config.cost_noise * rng.normal(size=base_cost.shape)
        local_costs[agent] = np.clip(base_cost + perturbation, 0.0, None)
    return SemiRelaxedOTProblem(
        source_weights=source_weights,
        target_weights=target_weights,
        source_points=source_points,
        target_points=target_points,
        local_costs=local_costs,
        relaxation=config.relaxation,
    )


def run_ot_experiment(config: OTRunConfig, out_dir: str | Path) -> tuple[pd.DataFrame, list[Path]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    problem = make_semirelaxed_ot_problem(config)
    reference_plan = solve_balanced_ot_lp(problem)
    frames: list[pd.DataFrame] = []
    final_plans: dict[str, np.ndarray] = {}
    for method in config.methods:
        name = method.lower()
        if name == "fw":
            frame, final_plan = run_fw(problem, config, reference_plan)
        elif name == "bcfw":
            frame, final_plan = run_bcfw(problem, config, reference_plan)
        elif name == "dfw":
            frame, final_plan = run_dfw_ot(problem, config, reference_plan)
        elif name == "dbcfw":
            frame, final_plan = run_dbcfw_ot(problem, config, reference_plan)
        else:
            raise ValueError(f"unknown OT method: {method}")
        frames.append(frame)
        final_plans[name] = final_plan
    result = pd.concat(frames, ignore_index=True)
    best_objective = float(result["objective"].min())
    result["objective_gap_to_best"] = result["objective"] - best_objective
    result.to_csv(out / "results.csv", index=False)
    _dump_ot_config(config, out / "run_config.yaml")
    _save_problem(problem, out / "problem.npz")
    np.savez(out / "final_plans.npz", **final_plans)
    plots = plot_ot_results(result, out / "plots")
    plots.append(plot_transport_plans(final_plans, problem, out / "plots" / "transport_plans.png"))
    return result, plots


def run_fw(
    problem: SemiRelaxedOTProblem,
    config: OTRunConfig,
    reference_plan: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    return _run_central(problem, config, method="fw", iterations=config.epochs, reference_plan=reference_plan)


def run_bcfw(
    problem: SemiRelaxedOTProblem,
    config: OTRunConfig,
    reference_plan: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    batch = max(1, min(config.batch, problem.n))
    iterations = int(np.ceil(config.epochs * problem.n / batch))
    return _run_central(
        problem, config, method="bcfw", iterations=iterations, reference_plan=reference_plan
    )


def _run_central(
    problem: SemiRelaxedOTProblem,
    config: OTRunConfig,
    method: str,
    iterations: int,
    reference_plan: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(config.seed + (17 if method == "bcfw" else 11))
    plan = problem.initial_plan()
    rows: list[dict[str, object]] = []
    total_oracle_columns = 0
    total_oracle_time = 0.0
    total_gradient_time = 0.0
    total_update_time = 0.0
    started = time.perf_counter()
    _append_ot_row(
        rows,
        problem,
        config,
        method,
        plan,
        elapsed=0.0,
        iteration=0,
        gamma=np.nan,
        oracle_columns=0,
        total_oracle_columns=0,
        lambda2=np.nan,
        consensus_error_value=0.0,
        reference_plan=reference_plan,
    )
    for iteration in range(iterations):
        gradient_started = time.perf_counter()
        gradient = problem.gradient(plan)
        gradient_time = time.perf_counter() - gradient_started
        if method == "fw":
            oracle_started = time.perf_counter()
            atom = problem.lmo(gradient)
            oracle_time = time.perf_counter() - oracle_started
            direction = atom - plan
            oracle_columns = problem.n
            update_started = time.perf_counter()
            gamma = _central_step_size(problem, config, direction, gradient, iteration, method)
            update_time = time.perf_counter() - update_started
        else:
            block_ids = rng.choice(problem.n, size=max(1, min(config.batch, problem.n)), replace=False)
            direction = np.zeros_like(plan)
            oracle_started = time.perf_counter()
            for column in block_ids:
                atom_col = problem.block_lmo(gradient[:, int(column)], int(column))
                direction[:, int(column)] = atom_col - plan[:, int(column)]
            oracle_time = time.perf_counter() - oracle_started
            oracle_columns = int(block_ids.size)
            update_started = time.perf_counter()
            gamma = _central_step_size(problem, config, direction, gradient, iteration, method)
            update_time = time.perf_counter() - update_started
        repair_started = time.perf_counter()
        plan = _repair_columns(plan + gamma * direction, problem.target_weights)
        update_time += time.perf_counter() - repair_started
        total_oracle_columns += oracle_columns
        total_oracle_time += oracle_time
        total_gradient_time += gradient_time
        total_update_time += update_time
        should_log = (iteration + 1) % config.log_every == 0 or iteration + 1 == iterations
        if should_log:
            _append_ot_row(
                rows,
                problem,
                config,
                method,
                plan,
                elapsed=time.perf_counter() - started,
                iteration=iteration + 1,
                gamma=gamma,
                oracle_columns=oracle_columns,
                total_oracle_columns=total_oracle_columns,
                lambda2=np.nan,
                consensus_error_value=0.0,
                reference_plan=reference_plan,
                oracle_time_sec=oracle_time,
                total_oracle_time_sec=total_oracle_time,
                gradient_time_sec=gradient_time,
                total_gradient_time_sec=total_gradient_time,
                update_time_sec=update_time,
                total_update_time_sec=total_update_time,
            )
    return pd.DataFrame(rows), plan


def run_dfw_ot(
    problem: SemiRelaxedOTProblem,
    config: OTRunConfig,
    reference_plan: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame, final_plan, _ = _run_decentralized_ot(
        problem,
        replace(config, batch=problem.n),
        method="dfw",
        reference_plan=reference_plan,
    )
    return frame, final_plan


def run_dbcfw_ot(
    problem: SemiRelaxedOTProblem,
    config: OTRunConfig,
    reference_plan: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame, final_plan, _ = _run_decentralized_ot(
        problem, config, method="dbcfw", reference_plan=reference_plan
    )
    return frame, final_plan


def _run_decentralized_ot(
    problem: SemiRelaxedOTProblem,
    config: OTRunConfig,
    method: str,
    reference_plan: np.ndarray | None = None,
    checkpoint_epochs: Iterable[float] = (),
) -> tuple[pd.DataFrame, np.ndarray, dict[float, np.ndarray]]:
    batch = max(1, min(config.batch, problem.n))
    iterations = int(np.ceil(config.epochs * problem.n / batch))
    graph_seed = config.graph_seed if config.graph_seed is not None else config.seed
    graph_config = GraphConfig(graph_name(config.graph), config.edge_prob, config.geometric_radius, graph_seed)
    graph_sequence = GraphSequence(problem.agents, graph_config)
    block_rng = np.random.default_rng(config.seed + 1009)
    plans = np.repeat(problem.initial_plan()[None, :, :], problem.agents, axis=0)
    mixed_prev = plans.copy()
    grad_prev = problem.local_gradients(mixed_prev)
    trackers = grad_prev.copy()
    rows: list[dict[str, object]] = []
    total_oracle_columns = 0
    total_oracle_time = 0.0
    total_gradient_time = 0.0
    total_communication_time = 0.0
    total_update_time = 0.0
    checkpoints = sorted(float(x) for x in checkpoint_epochs)
    checkpoint_plans: dict[float, np.ndarray] = {}
    started = time.perf_counter()
    _append_ot_row(
        rows,
        problem,
        config,
        method,
        plans.mean(axis=0),
        elapsed=0.0,
        iteration=0,
        gamma=np.nan,
        oracle_columns=0,
        total_oracle_columns=0,
        lambda2=np.nan,
        consensus_error_value=transport_consensus_error(plans),
        reference_plan=reference_plan,
    )
    while checkpoints and checkpoints[0] <= 0.0:
        checkpoint_plans[checkpoints.pop(0)] = plans.mean(axis=0).copy()
    for iteration in range(iterations):
        communication_started = time.perf_counter()
        weights, lambda2 = graph_sequence.next()
        mixed = _mix(weights, plans)
        communication_time = time.perf_counter() - communication_started
        gradient_started = time.perf_counter()
        grad_new = problem.local_gradients(mixed)
        gradient_time = time.perf_counter() - gradient_started
        corrected = trackers + grad_new - grad_prev
        communication_started = time.perf_counter()
        trackers = _mix(weights, corrected)
        communication_time += time.perf_counter() - communication_started
        next_plans = mixed.copy()
        gamma_values = []
        oracle_columns = 0
        oracle_time = 0.0
        update_time = 0.0
        for agent in range(problem.agents):
            block_ids = block_rng.choice(problem.n, size=batch, replace=False)
            direction = np.zeros((problem.m, problem.n), dtype=float)
            oracle_started = time.perf_counter()
            for column in block_ids:
                column = int(column)
                atom_col = problem.block_lmo(trackers[agent, :, column], column)
                direction[:, column] = atom_col - mixed[agent, :, column]
            oracle_time += time.perf_counter() - oracle_started
            update_started = time.perf_counter()
            gamma = _tracked_step_size(problem, config, direction, trackers[agent], iteration)
            next_plans[agent] = _repair_columns(mixed[agent] + gamma * direction, problem.target_weights)
            update_time += time.perf_counter() - update_started
            gamma_values.append(gamma)
            oracle_columns += int(block_ids.size)
        plans = next_plans
        mixed_prev = mixed
        grad_prev = grad_new
        total_oracle_columns += oracle_columns
        total_oracle_time += oracle_time
        total_gradient_time += gradient_time
        total_communication_time += communication_time
        total_update_time += update_time
        should_log = (iteration + 1) % config.log_every == 0 or iteration + 1 == iterations
        if should_log:
            _append_ot_row(
                rows,
                problem,
                config,
                method,
                plans.mean(axis=0),
                elapsed=time.perf_counter() - started,
                iteration=iteration + 1,
                gamma=float(np.mean(gamma_values)),
                oracle_columns=oracle_columns,
                total_oracle_columns=total_oracle_columns,
                lambda2=lambda2,
                consensus_error_value=transport_consensus_error(plans),
                reference_plan=reference_plan,
                oracle_time_sec=oracle_time,
                total_oracle_time_sec=total_oracle_time,
                gradient_time_sec=gradient_time,
                total_gradient_time_sec=total_gradient_time,
                communication_time_sec=communication_time,
                total_communication_time_sec=total_communication_time,
                update_time_sec=update_time,
                total_update_time_sec=total_update_time,
            )
        current_epoch = total_oracle_columns / (problem.agents * problem.n)
        while checkpoints and current_epoch >= checkpoints[0]:
            checkpoint_plans[checkpoints.pop(0)] = plans.mean(axis=0).copy()
    return pd.DataFrame(rows), plans.mean(axis=0), checkpoint_plans


def plot_ot_results(frame_or_csv: pd.DataFrame | str | Path, out_dir: str | Path) -> list[Path]:
    frame = pd.read_csv(frame_or_csv) if not isinstance(frame_or_csv, pd.DataFrame) else frame_or_csv.copy()
    if "objective_gap_to_best" not in frame.columns:
        frame["objective_gap_to_best"] = frame["objective"] - float(frame["objective"].min())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        _line_plot(frame, "oracle_epochs", "duality_gap", out / "duality_gap_vs_oracle_epochs.png"),
        _line_plot(frame, "iteration", "duality_gap", out / "duality_gap_vs_iteration.png"),
        _line_plot(frame, "wall_time_sec", "duality_gap", out / "duality_gap_vs_time.png"),
        _line_plot(frame, "oracle_epochs", "objective_gap_to_best", out / "objective_gap_vs_oracle_epochs.png"),
        _line_plot(frame, "oracle_epochs", "objective", out / "objective_vs_oracle_epochs.png", log_y=False),
        _line_plot(frame, "oracle_epochs", "consensus_error", out / "consensus_vs_oracle_epochs.png"),
    ]
    if "total_oracle_time_sec" in frame.columns:
        paths.append(_line_plot(
            frame,
            "total_oracle_time_sec",
            "duality_gap",
            out / "duality_gap_vs_oracle_time.png",
        ))
        paths.append(plot_ot_timing_breakdown(frame, out / "timing_breakdown.png"))
    return paths


def plot_ot_timing_breakdown(frame_or_csv: pd.DataFrame | str | Path, path: str | Path) -> Path:
    frame = pd.read_csv(frame_or_csv) if not isinstance(frame_or_csv, pd.DataFrame) else frame_or_csv.copy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final = _final_method_rows(frame)
    final = final.set_index("method").reindex([name for name in ("fw", "bcfw", "dfw", "dbcfw") if name in set(final["method"])])
    components = [
        ("total_oracle_time_sec", "LMO/oracle"),
        ("total_gradient_time_sec", "gradient"),
        ("total_communication_time_sec", "communication"),
        ("total_update_time_sec", "stepsize/update"),
    ]
    colors = ["#5B8DEF", "#F2A65A", "#5BB974", "#C06C84"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(final))
    bottom = np.zeros(len(final), dtype=float)
    for (column, label), color in zip(components, colors):
        values = final[column].to_numpy(dtype=float) if column in final.columns else np.zeros(len(final))
        ax.bar(x, values, bottom=bottom, width=0.62, label=label, color=color)
        bottom += values
    if "wall_time_sec" in final.columns:
        ax.scatter(x, final["wall_time_sec"].to_numpy(dtype=float), color="black", marker="x", s=55, label="wall clock")
    ax.set_xticks(x)
    ax.set_xticklabels([_simple_method_label(str(method)) for method in final.index])
    ax.set_ylabel("seconds")
    ax.set_title("OT timing breakdown at final logged iterate")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True, ncols=3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_transport_plans(
    final_plans: dict[str, np.ndarray],
    problem: SemiRelaxedOTProblem,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final_plans = {key: value for key, value in final_plans.items() if value is not None}
    methods = [name for name in ("fw", "bcfw", "dfw", "dbcfw") if name in final_plans]
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.4), squeeze=False)
    vmax = max(float(plan.max()) for plan in final_plans.values())
    for ax, method in zip(axes[0], methods):
        image = ax.imshow(final_plans[method], aspect="auto", origin="lower", cmap="viridis", vmax=vmax)
        ax.set_title(method.upper())
        ax.set_xlabel("target column")
        ax.set_ylabel("source row")
        ax.text(
            0.02,
            0.98,
            f"gap={problem.duality_gap(final_plans[method]):.2e}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.4, "edgecolor": "none", "pad": 3},
        )
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.83, label="transport mass")
    fig.suptitle("Semi-relaxed OT final transport plans")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def run_ot_paper_suite(config: OTPaperConfig, out_dir: str | Path) -> tuple[pd.DataFrame, list[Path]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    final_plans: dict[str, np.ndarray] = {}
    reference_problem: SemiRelaxedOTProblem | None = None
    reference_plan: np.ndarray | None = None
    for relaxation in config.relaxations:
        run_config = _paper_run_config(config, relaxation)
        problem = make_semirelaxed_ot_problem(run_config)
        lp_plan = solve_balanced_ot_lp(problem)
        if np.isclose(relaxation, config.convergence_relaxation):
            reference_problem = problem
            reference_plan = lp_plan
        for method in ("dfw", "dbcfw"):
            if method == "dfw":
                frame, plan = run_dfw_ot(problem, run_config, lp_plan)
            else:
                frame, plan = run_dbcfw_ot(problem, run_config, lp_plan)
            frame["sweep_relaxation"] = relaxation
            frames.append(frame)
            final_plans[f"{method}_lambda_{_slug_float(relaxation)}"] = plan
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(out / "paper_suite_results.csv", index=False)
    _dump_paper_config(config, out / "paper_config.yaml")
    if reference_problem is None or reference_plan is None:
        run_config = _paper_run_config(config, config.convergence_relaxation)
        reference_problem = make_semirelaxed_ot_problem(run_config)
        reference_plan = solve_balanced_ot_lp(reference_problem)
    np.savez(out / "balanced_ot_lp_reference.npz", reference_plan=reference_plan)
    plots = [
        plot_paper_lambda_sweep(result, out / "plots" / "paper_figure1_lambda_sweep.png"),
        plot_paper_convergence(
            result[result["sweep_relaxation"] == config.convergence_relaxation].copy(),
            out / "plots" / "paper_figure2_convergence.png",
        ),
        plot_paper_gap_variance(
            result[result["sweep_relaxation"] == config.convergence_relaxation].copy(),
            out / "plots" / "paper_figure4_gap_variance.png",
        ),
        plot_paper_setup(reference_problem, reference_plan, out / "plots" / "paper_figure5_reference_setup.png"),
    ]
    transition_epochs = _transition_epochs(config.epochs)
    for idx, relaxation in enumerate(config.transition_relaxations, start=6):
        run_config = _paper_run_config(config, relaxation)
        problem = make_semirelaxed_ot_problem(run_config)
        lp_plan = solve_balanced_ot_lp(problem)
        transition_frames: list[pd.DataFrame] = []
        checkpoints: dict[str, dict[float, np.ndarray]] = {}
        for method in ("dfw", "dbcfw"):
            batch_config = replace(run_config, batch=problem.n) if method == "dfw" else run_config
            frame, _, method_checkpoints = _run_decentralized_ot(
                problem,
                batch_config,
                method=method,
                reference_plan=lp_plan,
                checkpoint_epochs=transition_epochs,
            )
            frame["sweep_relaxation"] = relaxation
            transition_frames.append(frame)
            checkpoints[method] = method_checkpoints
        transition_frame = pd.concat(transition_frames, ignore_index=True)
        slug = _slug_float(relaxation)
        transition_frame.to_csv(out / f"transition_lambda_{slug}.csv", index=False)
        plots.append(plot_paper_transition_heatmaps(
            checkpoints,
            problem,
            out / "plots" / f"paper_figure{idx}_transition_heatmaps_lambda_{slug}.png",
        ))
        plots.append(plot_paper_transition_curves(
            transition_frame,
            out / "plots" / f"paper_figure{idx}_objective_gradient_lambda_{slug}.png",
        ))
    plot_transport_plans(
        {
            "dfw": final_plans.get(f"dfw_lambda_{_slug_float(config.convergence_relaxation)}"),
            "dbcfw": final_plans.get(f"dbcfw_lambda_{_slug_float(config.convergence_relaxation)}"),
        },
        reference_problem,
        out / "plots" / "paper_final_transport_plans.png",
    )
    plots.append(out / "plots" / "paper_final_transport_plans.png")
    return result, plots


def plot_paper_lambda_sweep(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final = _final_rows(frame)
    panels = [
        ("objective", "objective value: f(T)", True),
        ("duality_gap", "duality gap: g(T)", True),
        ("marginal_constraint_error", "marginal constraint error: e_c", True),
        ("sparsity", "sparsity", False),
        ("transport_matrix_error", "matrix error: e_M", False),
        ("value_error", "value error: e_V", True),
        ("wall_time_sec", "computational time [sec]", False),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6))
    axes_flat = axes.ravel()
    for ax, (metric, title, log_y) in zip(axes_flat, panels):
        _plot_relaxation_lines(ax, final, metric, log_y)
        ax.set_title(title)
    axes_flat[-1].axis("off")
    fig.suptitle("Paper Figure 1 analogue: relaxation sweep, decentralized DFW vs DBCFW")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_paper_convergence(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("oracle_epochs", "objective", "objective value: f(T)", False),
        ("wall_time_sec", "objective", "objective value (time): f(T)", False),
        ("oracle_epochs", "duality_gap", "duality gap: g(T)", True),
        ("wall_time_sec", "duality_gap", "duality gap (time): g(T)", True),
        ("oracle_epochs", "marginal_constraint_error", "marginal constraint error: e_c", True),
        ("oracle_epochs", "sparsity", "sparsity", False),
        ("oracle_epochs", "transport_matrix_error", "matrix error: e_M", False),
        ("wall_time_sec", "transport_matrix_error", "matrix error (time): e_M", False),
        ("oracle_epochs", "value_error", "value error: e_V", True),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for ax, (x_col, y_col, title, log_y) in zip(axes.ravel(), panels):
        _plot_method_lines(ax, frame, x_col, y_col, log_y)
        ax.set_title(title)
    fig.suptitle("Paper Figure 2 analogue: convergence, decentralized DFW vs DBCFW")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_paper_gap_variance(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("oracle_epochs", "objective", "objective value", False),
        ("wall_time_sec", "objective", "objective value (time)", False),
        ("oracle_epochs", "duality_gap", "duality gap", True),
        ("oracle_epochs", "column_gap_variance", "variance of column duality gap", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (x_col, y_col, title, log_y) in zip(axes.ravel(), panels):
        _plot_method_lines(ax, frame, x_col, y_col, log_y)
        ax.set_title(title)
    fig.suptitle("Paper Figure 4 analogue: gap-adaptive diagnostic, DFW vs DBCFW")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_paper_setup(problem: SemiRelaxedOTProblem, reference_plan: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].scatter(
        problem.source_points[:, 0],
        problem.source_points[:, 1],
        s=900 * problem.source_weights,
        c=np.arange(problem.m),
        cmap="viridis",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_title("source support a")
    axes[1].scatter(
        problem.target_points[:, 0],
        problem.target_points[:, 1],
        s=900 * problem.target_weights,
        c=np.arange(problem.n),
        cmap="plasma",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].set_title("reference support b")
    image = axes[2].imshow(reference_plan, origin="lower", aspect="auto", cmap="viridis")
    axes[2].set_title("balanced OT LP reference")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.colorbar(image, ax=axes[2], shrink=0.82, label="transport mass")
    fig.suptitle("Paper Figure 5 analogue: source/reference setup and LP plan")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_paper_transition_heatmaps(
    checkpoints: dict[str, dict[float, np.ndarray]],
    problem: SemiRelaxedOTProblem,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = sorted({epoch for method_maps in checkpoints.values() for epoch in method_maps})
    row_labels = [("dfw", "DFW T"), ("dbcfw", "DBCFW T"), ("dfw_norm", "DFW row-normalized"), ("dbcfw_norm", "DBCFW row-normalized")]
    fig, axes = plt.subplots(len(row_labels), len(epochs), figsize=(4.0 * len(epochs), 11.5), squeeze=False)
    vmax = max(float(plan.max()) for method_maps in checkpoints.values() for plan in method_maps.values())
    for row, (key, label) in enumerate(row_labels):
        method = key.replace("_norm", "")
        for col, epoch in enumerate(epochs):
            ax = axes[row, col]
            plan = checkpoints.get(method, {}).get(epoch)
            if plan is None:
                ax.axis("off")
                continue
            image_plan = row_normalized_plan(plan) if key.endswith("_norm") else plan
            vmax_row = 1.0 if key.endswith("_norm") else vmax
            ax.imshow(image_plan, origin="lower", aspect="auto", cmap="viridis", vmax=vmax_row)
            ax.set_title(f"{label}, epoch={epoch:g}")
            ax.set_xlabel("target column")
            ax.set_ylabel("source row")
    fig.suptitle(f"Paper transition analogue: transport matrices, lambda={problem.relaxation:g}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_paper_transition_curves(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
    for method, group in frame.groupby("method", sort=False):
        data = group.sort_values("oracle_epochs")
        label = _simple_method_label(method)
        axes[0].plot(data["oracle_epochs"], data["objective"], label=f"{label}: f(T)", linewidth=1.7)
        axes[0].plot(data["oracle_epochs"], data["transport_cost"], linestyle="--", label=f"{label}: <T,C>", linewidth=1.2)
        axes[0].plot(data["oracle_epochs"], data["relaxation_penalty"], linestyle=":", label=f"{label}: penalty", linewidth=1.5)
        axes[1].plot(data["oracle_epochs"], data["gradient_norm"].clip(lower=1e-16), label=f"{label}: full", linewidth=1.7)
        axes[1].plot(data["oracle_epochs"], data["penalty_gradient_norm"].clip(lower=1e-16), linestyle="--", label=f"{label}: penalty", linewidth=1.2)
    axes[0].set_title("objective value components")
    axes[0].set_xlabel("oracle epochs")
    axes[0].set_ylabel("value")
    axes[0].set_yscale("log")
    axes[1].set_title("norm of gradient")
    axes[1].set_xlabel("oracle epochs")
    axes[1].set_ylabel("norm")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=True)
    first = frame.iloc[0]
    fig.suptitle(f"Paper transition analogue: objective and gradient, lambda={float(first['relaxation']):g}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def row_residual(plan: np.ndarray, source_weights: np.ndarray) -> np.ndarray:
    return plan.sum(axis=1) - source_weights


def transport_consensus_error(plans: np.ndarray) -> float:
    avg = plans.mean(axis=0)
    flat = (plans - avg[None, :, :]).reshape(plans.shape[0], -1)
    return float(np.linalg.norm(flat, axis=1).mean())


def solve_balanced_ot_lp(problem: SemiRelaxedOTProblem) -> np.ndarray:
    m, n = problem.m, problem.n
    c = problem.cost.reshape(-1)
    constraints: list[np.ndarray] = []
    rhs: list[float] = []
    for row in range(m):
        coeff = np.zeros((m, n), dtype=float)
        coeff[row, :] = 1.0
        constraints.append(coeff.reshape(-1))
        rhs.append(float(problem.source_weights[row]))
    for column in range(n):
        coeff = np.zeros((m, n), dtype=float)
        coeff[:, column] = 1.0
        constraints.append(coeff.reshape(-1))
        rhs.append(float(problem.target_weights[column]))
    result = linprog(
        c,
        A_eq=np.vstack(constraints),
        b_eq=np.array(rhs),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"balanced OT LP failed: {result.message}")
    return result.x.reshape(m, n)


def column_duality_gaps(problem: SemiRelaxedOTProblem, plan: np.ndarray) -> np.ndarray:
    gradient = problem.gradient(plan)
    gaps = np.empty(problem.n, dtype=float)
    for column in range(problem.n):
        atom_col = problem.block_lmo(gradient[:, column], column)
        gaps[column] = float((plan[:, column] - atom_col) @ gradient[:, column])
    return np.maximum(gaps, 0.0)


def row_normalized_plan(plan: np.ndarray) -> np.ndarray:
    row_sums = plan.sum(axis=1, keepdims=True)
    return np.divide(plan, row_sums, out=np.zeros_like(plan), where=row_sums > 1e-18)


def barycentric_projection(plan: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    row_sums = plan.sum(axis=1, keepdims=True)
    projected = plan @ target_points
    return np.divide(projected, row_sums, out=np.zeros_like(projected), where=row_sums > 1e-18)


def _paper_metrics(
    problem: SemiRelaxedOTProblem,
    plan: np.ndarray,
    reference_plan: np.ndarray | None,
) -> dict[str, float]:
    residual = row_residual(plan, problem.source_weights)
    cost = float(np.sum(plan * problem.cost))
    penalty = float(0.5 / problem.relaxation * residual @ residual)
    cost_grad_norm = float(np.linalg.norm(problem.cost))
    penalty_gradient = residual[:, None] / problem.relaxation
    penalty_grad_norm = float(np.linalg.norm(np.repeat(penalty_gradient, problem.n, axis=1)))
    metrics = {
        "marginal_constraint_error": float(np.linalg.norm(residual)),
        "sparsity": float(np.mean(plan <= 1e-12)),
        "transport_matrix_error": float("nan"),
        "value_error": float("nan"),
        "transport_cost": cost,
        "relaxation_penalty": penalty,
        "cost_gradient_norm": cost_grad_norm,
        "penalty_gradient_norm": penalty_grad_norm,
        "gradient_norm": float(np.linalg.norm(problem.gradient(plan))),
        "column_gap_variance": float(np.var(column_duality_gaps(problem, plan))),
    }
    if reference_plan is None:
        return metrics
    reference_norm = max(float(np.linalg.norm(reference_plan)), 1e-18)
    metrics["transport_matrix_error"] = float(np.linalg.norm(plan - reference_plan) / reference_norm)
    ref_cost = max(abs(float(np.sum(reference_plan * problem.cost))), 1e-18)
    metrics["value_error"] = abs(cost - float(np.sum(reference_plan * problem.cost))) / ref_cost
    ref_projection = barycentric_projection(reference_plan, problem.target_points)
    projection = barycentric_projection(plan, problem.target_points)
    projection_norm = max(float(np.linalg.norm(ref_projection)), 1e-18)
    metrics["barycentric_projection_error"] = float(np.linalg.norm(projection - ref_projection) / projection_norm)
    return metrics


def _append_ot_row(
    rows: list[dict[str, object]],
    problem: SemiRelaxedOTProblem,
    config: OTRunConfig,
    method: str,
    plan: np.ndarray,
    elapsed: float,
    iteration: int,
    gamma: float,
    oracle_columns: int,
    total_oracle_columns: int,
    lambda2: float,
    consensus_error_value: float,
    reference_plan: np.ndarray | None = None,
    oracle_time_sec: float = 0.0,
    total_oracle_time_sec: float = 0.0,
    gradient_time_sec: float = 0.0,
    total_gradient_time_sec: float = 0.0,
    communication_time_sec: float = 0.0,
    total_communication_time_sec: float = 0.0,
    update_time_sec: float = 0.0,
    total_update_time_sec: float = 0.0,
) -> None:
    denominator = problem.agents * problem.n if method in {"dfw", "dbcfw"} else problem.n
    metrics = _paper_metrics(problem, plan, reference_plan)
    rows.append({
        "run_id": _ot_run_id(config, method),
        "method": method,
        "agents": problem.agents,
        "m": problem.m,
        "n": problem.n,
        "batch": problem.n if method == "fw" else max(1, min(config.batch, problem.n)),
        "seed": config.seed,
        "graph_seed": config.graph_seed if config.graph_seed is not None else config.seed,
        "graph": "centralized" if method in {"fw", "bcfw"} else graph_name(config.graph),
        "iteration": iteration,
        "algorithm_rounds": iteration,
        "communication_rounds": iteration if method in {"dfw", "dbcfw"} else 0,
        "gamma": gamma,
        "wall_time_sec": elapsed,
        "objective": problem.objective(plan),
        "duality_gap": max(problem.duality_gap(plan), 0.0),
        "consensus_error": consensus_error_value,
        "row_marginal_error": float(np.linalg.norm(row_residual(plan, problem.source_weights))),
        "column_marginal_error": float(np.abs(plan.sum(axis=0) - problem.target_weights).max()),
        "transport_density": float(np.mean(plan > 1e-12)),
        "marginal_constraint_error": metrics["marginal_constraint_error"],
        "sparsity": metrics["sparsity"],
        "transport_matrix_error": metrics["transport_matrix_error"],
        "value_error": metrics["value_error"],
        "barycentric_projection_error": metrics.get("barycentric_projection_error", float("nan")),
        "transport_cost": metrics["transport_cost"],
        "relaxation_penalty": metrics["relaxation_penalty"],
        "cost_gradient_norm": metrics["cost_gradient_norm"],
        "penalty_gradient_norm": metrics["penalty_gradient_norm"],
        "gradient_norm": metrics["gradient_norm"],
        "column_gap_variance": metrics["column_gap_variance"],
        "oracle_columns_per_iter": oracle_columns,
        "total_oracle_columns": total_oracle_columns,
        "oracle_epochs": total_oracle_columns / denominator,
        "oracle_time_sec": oracle_time_sec,
        "total_oracle_time_sec": total_oracle_time_sec,
        "gradient_time_sec": gradient_time_sec,
        "total_gradient_time_sec": total_gradient_time_sec,
        "communication_time_sec": communication_time_sec,
        "total_communication_time_sec": total_communication_time_sec,
        "update_time_sec": update_time_sec,
        "total_update_time_sec": total_update_time_sec,
        "profiled_algorithm_time_sec": (
            total_oracle_time_sec
            + total_gradient_time_sec
            + total_communication_time_sec
            + total_update_time_sec
        ),
        "lambda2": lambda2,
        "relaxation": config.relaxation,
        "cost_noise": config.cost_noise,
        "stepsize": config.stepsize,
    })


def _central_step_size(
    problem: SemiRelaxedOTProblem,
    config: OTRunConfig,
    direction: np.ndarray,
    gradient: np.ndarray,
    iteration: int,
    method: str,
) -> float:
    if config.stepsize == "decay":
        return _decay_step(iteration, problem.n if method == "bcfw" else 1)
    return _line_search_from_gradient(problem.relaxation, direction, gradient)


def _tracked_step_size(
    problem: SemiRelaxedOTProblem,
    config: OTRunConfig,
    direction: np.ndarray,
    tracker: np.ndarray,
    iteration: int,
) -> float:
    if config.stepsize == "decay":
        return _decay_step(iteration, problem.n)
    return _line_search_from_gradient(problem.relaxation, direction, tracker)


def _line_search_from_gradient(relaxation: float, direction: np.ndarray, gradient: np.ndarray) -> float:
    row_direction = direction.sum(axis=1)
    denom = float(row_direction @ row_direction)
    directional_derivative = float(np.sum(direction * gradient))
    if denom <= 1e-18:
        return 0.0 if directional_derivative >= 0.0 else 1.0
    return float(np.clip(-relaxation * directional_derivative / denom, 0.0, 1.0))


def _decay_step(iteration: int, block_count: int) -> float:
    n_blocks = max(1, int(block_count))
    return float(min(1.0, 2.0 * n_blocks / (iteration + 2.0 * n_blocks)))


def _repair_columns(plan: np.ndarray, target_weights: np.ndarray) -> np.ndarray:
    repaired = np.clip(plan, 0.0, None)
    sums = repaired.sum(axis=0)
    for column, target in enumerate(target_weights):
        if sums[column] <= 1e-18:
            repaired[:, column] = 0.0
            repaired[0, column] = target
        else:
            repaired[:, column] *= target / sums[column]
    return repaired


def _mix(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jmn->imn", weights, values)


def _line_plot(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    path: Path,
    log_y: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for method, group in frame.groupby("method", sort=False):
        data = group.sort_values(x_col)
        y = data[y_col]
        if log_y:
            y = y.clip(lower=1e-16)
        ax.plot(
            data[x_col],
            y,
            marker="o",
            markersize=2.5,
            linewidth=1.6,
            label=_ot_method_label(method, data),
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(_ot_plot_title(frame, y_col))
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_relaxation_lines(ax, frame: pd.DataFrame, metric: str, log_y: bool) -> None:
    for method, group in frame.groupby("method", sort=False):
        data = group.sort_values("sweep_relaxation")
        y = data[metric]
        if log_y:
            y = y.clip(lower=1e-16)
        ax.plot(
            data["sweep_relaxation"],
            y,
            marker="o",
            linewidth=1.7,
            label=_simple_method_label(method),
        )
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("relaxation lambda")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=True)


def _plot_method_lines(
    ax,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    log_y: bool,
) -> None:
    for method, group in frame.groupby("method", sort=False):
        data = group.sort_values(x_col)
        y = data[y_col]
        if log_y:
            y = y.clip(lower=1e-16)
        ax.plot(data[x_col], y, marker="o", markersize=2, linewidth=1.5, label=_simple_method_label(method))
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("oracle epochs" if x_col == "oracle_epochs" else x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=True)


def _final_rows(frame: pd.DataFrame) -> pd.DataFrame:
    idx = frame.groupby(["method", "sweep_relaxation"])["oracle_epochs"].idxmax()
    return frame.loc[idx].copy()


def _final_method_rows(frame: pd.DataFrame) -> pd.DataFrame:
    idx = frame.groupby("method")["iteration"].idxmax()
    return frame.loc[idx].copy()


def _simple_method_label(method: str) -> str:
    labels = {
        "fw": "FW",
        "bcfw": "BCFW",
        "dfw": "DFW",
        "dbcfw": "DBCFW",
    }
    return labels.get(method, method.upper())


def _paper_run_config(config: OTPaperConfig, relaxation: float) -> OTRunConfig:
    return OTRunConfig(
        methods=("dfw", "dbcfw"),
        m=config.m,
        n=config.n,
        agents=config.agents,
        epochs=config.epochs,
        batch=config.batch,
        relaxation=relaxation,
        cost_noise=config.cost_noise,
        stepsize=config.stepsize,
        graph=config.graph,
        edge_prob=config.edge_prob,
        geometric_radius=config.geometric_radius,
        seed=config.seed,
        graph_seed=config.graph_seed,
        log_every=config.log_every,
    )


def _transition_epochs(epochs: int) -> tuple[float, ...]:
    values = [0.0, 1.0, max(2.0, 0.1 * epochs), max(3.0, 0.4 * epochs), float(epochs)]
    deduped = sorted({round(value, 6) for value in values})
    return tuple(deduped)


def _slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _dump_paper_config(config: OTPaperConfig, path: Path) -> None:
    data = asdict(config)
    data["relaxations"] = list(config.relaxations)
    data["transition_relaxations"] = list(config.transition_relaxations)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def _ot_method_label(method: str, frame: pd.DataFrame) -> str:
    if method == "fw":
        return "FW: full columns"
    if method == "bcfw":
        batch = int(frame["batch"].iloc[0])
        return f"BCFW: B={batch}"
    if method == "dfw":
        graph = str(frame["graph"].iloc[0])
        return f"DFW: full columns, {graph}"
    if method == "dbcfw":
        batch = int(frame["batch"].iloc[0])
        graph = str(frame["graph"].iloc[0])
        return f"DBCFW: B={batch}, {graph}"
    return method.upper()


def _ot_plot_title(frame: pd.DataFrame, metric: str) -> str:
    first = frame.iloc[0]
    return (
        f"Semi-relaxed OT {metric}: m={int(first['m'])}, n={int(first['n'])}, "
        f"agents={int(first['agents'])}, lambda={float(first['relaxation']):g}, "
        f"stepsize={first['stepsize']}"
    )


def _curve_points(count: int, rng: np.random.Generator, phase: float, jitter: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count)
    points = np.column_stack((t, 0.25 * np.sin(2.0 * np.pi * (t + phase))))
    points += jitter * rng.normal(size=points.shape)
    return points


def _smooth_weights(count: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.gamma(shape=8.0, scale=1.0, size=count)
    return raw / raw.sum()


def _squared_cost(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    diff = source_points[:, None, :] - target_points[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    scale = float(cost.max())
    return cost / scale if scale > 0.0 else cost


def _dump_ot_config(config: OTRunConfig, path: Path) -> None:
    data = asdict(config)
    data["methods"] = list(config.methods)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def _save_problem(problem: SemiRelaxedOTProblem, path: Path) -> None:
    np.savez(
        path,
        source_weights=problem.source_weights,
        target_weights=problem.target_weights,
        source_points=problem.source_points,
        target_points=problem.target_points,
        local_costs=problem.local_costs,
        relaxation=np.array(problem.relaxation),
    )


def _ot_run_id(config: OTRunConfig, method: str) -> str:
    graph = "central" if method in {"fw", "bcfw"} else graph_name(config.graph)
    parts: Iterable[str] = (
        "semirelaxed_ot",
        method,
        f"A{config.agents}",
        f"m{config.m}",
        f"n{config.n}",
        f"B{config.batch if method != 'fw' else config.n}",
        graph,
        f"s{config.seed}",
    )
    return "_".join(parts)
