from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

from dbcfw_bench.ot_experiment import (
    SemiRelaxedOTProblem,
    row_residual,
    solve_balanced_ot_lp,
)


METHOD_ORDER = ("fw", "bcfw", "dfw", "dbcfw")
SOURCE_COLOR = "#15AABF"
TARGET_COLOR = "#F2C94C"
FLOW_COLOR = "#61646B"
REFERENCE_COLOR = "#343A40"
EXTRA_COLOR = "#2EAD66"
MISSING_COLOR = "#D84A4A"


def build_ot_route_report(run_dir: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    root = Path(run_dir)
    out = Path(out_dir) if out_dir is not None else root / "route_report"
    out.mkdir(parents=True, exist_ok=True)
    _set_route_style()

    problem = load_saved_problem(root / "problem.npz")
    final_plans = load_saved_plans(root / "final_plans.npz")
    frame = pd.read_csv(root / "results.csv") if (root / "results.csv").exists() else pd.DataFrame()

    balanced_reference = solve_balanced_ot_lp(problem)
    semirelaxed_reference, qp_message = solve_semirelaxed_ot_reference(problem, balanced_reference)
    references = {
        "balanced_lp": balanced_reference,
        "semi_relaxed_qp": semirelaxed_reference,
    }
    np.savez(out / "route_references.npz", **references)

    paths = [
        plot_optimal_references(problem, balanced_reference, semirelaxed_reference, out / "01_optimal_references.png"),
        plot_methods_against_reference(
            problem,
            final_plans,
            semirelaxed_reference,
            frame,
            out / "02_methods_vs_semirelaxed_optimum.png",
        ),
        plot_route_differences(
            problem,
            final_plans,
            semirelaxed_reference,
            out / "03_route_difference_to_semirelaxed_optimum.png",
        ),
        plot_flow_snapshots(
            problem,
            semirelaxed_reference,
            out / "04_semirelaxed_optimal_flow_snapshots.png",
            title="Semi-relaxed optimum: mass motion along optimal routes",
        ),
        plot_agent_local_preferences(problem, out / "05_agent_local_route_preferences.png"),
    ]
    if not frame.empty:
        paths.append(plot_route_convergence_diagnostics(frame, out / "06_convergence_accounting.png"))
    summary_path = write_route_summary(
        problem,
        final_plans,
        semirelaxed_reference,
        balanced_reference,
        frame,
        out / "route_report_summary.csv",
        qp_message,
    )
    paths.append(summary_path)
    return paths


def load_saved_problem(path: str | Path) -> SemiRelaxedOTProblem:
    data = np.load(path)
    return SemiRelaxedOTProblem(
        source_weights=np.asarray(data["source_weights"], dtype=float),
        target_weights=np.asarray(data["target_weights"], dtype=float),
        source_points=np.asarray(data["source_points"], dtype=float),
        target_points=np.asarray(data["target_points"], dtype=float),
        local_costs=np.asarray(data["local_costs"], dtype=float),
        relaxation=float(np.asarray(data["relaxation"])),
    )


def load_saved_plans(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.asarray(data[key], dtype=float) for key in data.files}


def solve_semirelaxed_ot_reference(
    problem: SemiRelaxedOTProblem,
    initial_plan: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    x0 = problem.initial_plan() if initial_plan is None else initial_plan
    x0 = _repair_columns(x0, problem.target_weights)
    m, n = problem.m, problem.n
    cost = problem.cost

    def objective(flat: np.ndarray) -> float:
        plan = flat.reshape(m, n)
        residual = row_residual(plan, problem.source_weights)
        return float(np.sum(plan * cost) + 0.5 / problem.relaxation * residual @ residual)

    def gradient(flat: np.ndarray) -> np.ndarray:
        plan = flat.reshape(m, n)
        residual = row_residual(plan, problem.source_weights)
        return (cost + residual[:, None] / problem.relaxation).reshape(-1)

    eq = np.zeros((n, m * n), dtype=float)
    for column in range(n):
        eq[column, column::n] = 1.0
    constraints = [LinearConstraint(eq, problem.target_weights, problem.target_weights)]
    result = minimize(
        objective,
        x0.reshape(-1),
        jac=gradient,
        bounds=Bounds(0.0, np.inf),
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1200, "disp": False},
    )
    if not result.success:
        return x0, f"semi-relaxed QP fallback to balanced LP: {result.message}"
    return _repair_columns(result.x.reshape(m, n), problem.target_weights), str(result.message)


def plot_optimal_references(
    problem: SemiRelaxedOTProblem,
    balanced_reference: np.ndarray,
    semirelaxed_reference: np.ndarray,
    path: str | Path,
) -> Path:
    path = Path(path)
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.3), sharex=True, sharey=True)
    _draw_supports(axes[0], problem, title="Supports and masses")
    _draw_route_plan(
        axes[1],
        problem,
        balanced_reference,
        "Balanced OT LP reference",
        subtitle="hard source and target marginals",
        route_color=REFERENCE_COLOR,
    )
    _draw_route_plan(
        axes[2],
        problem,
        semirelaxed_reference,
        "Semi-relaxed optimum",
        subtitle=f"same objective as methods, lambda={problem.relaxation:g}",
        route_color=REFERENCE_COLOR,
    )
    fig.suptitle("What optimal routing should look like before comparing algorithms", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_methods_against_reference(
    problem: SemiRelaxedOTProblem,
    final_plans: dict[str, np.ndarray],
    reference_plan: np.ndarray,
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    path = Path(path)
    methods = [name for name in METHOD_ORDER if name in final_plans]
    panels = [("optimum", reference_plan)] + [(name, final_plans[name]) for name in methods]
    rows, cols = _grid_shape(len(panels), max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(5.7 * cols, 5.25 * rows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (name, plan) in zip(axes_flat, panels):
        title = "Semi-relaxed optimum" if name == "optimum" else name.upper()
        subtitle = _method_subtitle(name, frame) if name != "optimum" else "reference for this objective"
        _draw_route_plan(
            ax,
            problem,
            plan,
            title,
            subtitle=subtitle,
            reference_plan=None if name == "optimum" else reference_plan,
            route_color=REFERENCE_COLOR if name == "optimum" else FLOW_COLOR,
        )
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("Final transport routes: method plans compared to the semi-relaxed optimum", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_route_differences(
    problem: SemiRelaxedOTProblem,
    final_plans: dict[str, np.ndarray],
    reference_plan: np.ndarray,
    path: str | Path,
) -> Path:
    path = Path(path)
    methods = [name for name in METHOD_ORDER if name in final_plans]
    rows, cols = _grid_shape(len(methods), max_cols=2)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.45 * rows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, method in zip(axes_flat, methods):
        _draw_route_difference(ax, problem, final_plans[method], reference_plan, method.upper())
    for ax in axes_flat[len(methods):]:
        ax.axis("off")
    fig.suptitle("Where each method differs from the semi-relaxed optimum", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_flow_snapshots(
    problem: SemiRelaxedOTProblem,
    plan: np.ndarray,
    path: str | Path,
    title: str,
) -> Path:
    path = Path(path)
    times = np.linspace(0.0, 1.0, 5)
    entries = _route_entries(plan, max_entries=150)
    max_mass = max((mass for mass, _, _ in entries), default=1e-16)
    fig, axes = plt.subplots(1, len(times), figsize=(18.5, 4.3), sharex=True, sharey=True)
    for ax, time_value in zip(axes, times):
        _draw_base_routes(ax, problem, plan, color="#B8BDC7", alpha=0.13, linewidth=0.42, max_entries=100)
        points = []
        sizes = []
        for mass, source, target in entries:
            start = problem.source_points[source]
            end = problem.target_points[target]
            points.append((1.0 - time_value) * start + time_value * end)
            sizes.append(36.0 + 520.0 * np.sqrt(mass / max_mass))
        if points:
            color = _interpolate_color(SOURCE_COLOR, TARGET_COLOR, float(time_value))
            ax.scatter(
                np.asarray(points)[:, 0],
                np.asarray(points)[:, 1],
                s=np.asarray(sizes),
                color=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.82,
                zorder=4,
            )
        _draw_support_points(ax, problem, source_alpha=0.32, target_alpha=0.32, labels=False)
        ax.set_title(f"t={time_value:.2g}")
        _finish_geometry_axis(ax, problem)
    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_agent_local_preferences(problem: SemiRelaxedOTProblem, path: str | Path) -> Path:
    path = Path(path)
    count = min(problem.agents, 6)
    rows, cols = _grid_shape(count, max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(5.65 * cols, 5.15 * rows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    local_plans = []
    for agent in range(count):
        local_problem = SemiRelaxedOTProblem(
            source_weights=problem.source_weights,
            target_weights=problem.target_weights,
            source_points=problem.source_points,
            target_points=problem.target_points,
            local_costs=problem.local_costs[agent : agent + 1],
            relaxation=problem.relaxation,
        )
        local_plans.append(solve_balanced_ot_lp(local_problem))
    for ax, plan, agent in zip(axes_flat, local_plans, range(count)):
        _draw_route_plan(
            ax,
            problem,
            plan,
            f"Agent {agent}",
            subtitle="local-cost balanced optimum",
            route_color=FLOW_COLOR,
        )
    for ax in axes_flat[count:]:
        ax.axis("off")
    fig.suptitle("What each agent would route using only its local cost", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_route_convergence_diagnostics(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2))
    _plot_metric_lines(axes[0, 0], frame, "oracle_epochs", "duality_gap", "Dual gap vs oracle epochs")
    _plot_metric_lines(axes[0, 1], frame, "algorithm_rounds", "duality_gap", "Dual gap vs algorithm rounds")
    _plot_metric_lines(axes[1, 0], frame, "wall_time_sec", "duality_gap", "Dual gap vs wall time")
    _plot_metric_lines(axes[1, 1], frame[frame["iteration"] > 0], "iteration", "gamma", "Line-search gamma")
    axes[1, 1].set_yscale("linear")
    fig.suptitle("Convergence accounting: oracle work is not the same as communication rounds", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def write_route_summary(
    problem: SemiRelaxedOTProblem,
    final_plans: dict[str, np.ndarray],
    semirelaxed_reference: np.ndarray,
    balanced_reference: np.ndarray,
    frame: pd.DataFrame,
    path: str | Path,
    qp_message: str,
) -> Path:
    path = Path(path)
    rows: list[dict[str, object]] = []
    final_frame = pd.DataFrame()
    if not frame.empty:
        final_frame = frame.sort_values("iteration").groupby("method", as_index=False).tail(1).set_index("method")
    for method in [name for name in METHOD_ORDER if name in final_plans]:
        plan = final_plans[method]
        row = {
            "method": method,
            "objective": problem.objective(plan),
            "duality_gap": max(problem.duality_gap(plan), 0.0),
            "row_marginal_error": float(np.linalg.norm(row_residual(plan, problem.source_weights))),
            "matrix_error_to_semirelaxed_qp": _relative_matrix_error(plan, semirelaxed_reference),
            "matrix_error_to_balanced_lp": _relative_matrix_error(plan, balanced_reference),
            "support_routes": int(np.sum(plan > 1e-12)),
            "qp_reference_message": qp_message,
        }
        if not final_frame.empty and method in final_frame.index:
            for column in (
                "oracle_epochs",
                "algorithm_rounds",
                "communication_rounds",
                "wall_time_sec",
                "total_oracle_time_sec",
                "total_communication_time_sec",
                "gamma",
            ):
                if column in final_frame.columns:
                    row[column] = final_frame.loc[method, column]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _draw_supports(ax, problem: SemiRelaxedOTProblem, title: str) -> None:
    _draw_support_points(ax, problem, labels=True)
    ax.set_title(title)
    ax.text(
        0.02,
        0.02,
        f"{problem.m} sources, {problem.n} targets\npoint size = marginal mass",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#D7DBE2", "alpha": 0.92, "pad": 4},
    )
    _finish_geometry_axis(ax, problem)


def _draw_route_plan(
    ax,
    problem: SemiRelaxedOTProblem,
    plan: np.ndarray,
    title: str,
    subtitle: str = "",
    reference_plan: np.ndarray | None = None,
    route_color: str = FLOW_COLOR,
) -> None:
    _draw_base_routes(ax, problem, plan, color=route_color, alpha=0.18, linewidth=0.36, max_entries=None)
    entries = _route_entries(plan, max_entries=65)
    max_mass = max((mass for mass, _, _ in entries), default=1e-16)
    for mass, source, target in reversed(entries):
        start = problem.source_points[source]
        end = problem.target_points[target]
        scale = np.sqrt(mass / max_mass)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=route_color,
            alpha=0.28 + 0.42 * scale,
            linewidth=0.45 + 3.25 * scale,
            solid_capstyle="round",
            zorder=2,
        )
    _draw_support_points(ax, problem, labels=False)
    ax.set_title(_route_title(problem, plan, title, subtitle, reference_plan), fontsize=10.5)
    _finish_geometry_axis(ax, problem)


def _draw_route_difference(
    ax,
    problem: SemiRelaxedOTProblem,
    plan: np.ndarray,
    reference_plan: np.ndarray,
    title: str,
) -> None:
    _draw_base_routes(ax, problem, reference_plan, color="#9AA1AD", alpha=0.16, linewidth=0.42, max_entries=120)
    diff = plan - reference_plan
    max_abs = max(float(np.max(np.abs(diff))), 1e-16)
    for sign, color, label in ((1.0, EXTRA_COLOR, "extra"), (-1.0, MISSING_COLOR, "missing")):
        entries = []
        for source in range(problem.m):
            for target in range(problem.n):
                value = sign * float(diff[source, target])
                if value > 1e-12:
                    entries.append((value, source, target))
        entries.sort(reverse=True)
        for value, source, target in entries[:80]:
            start = problem.source_points[source]
            end = problem.target_points[target]
            scale = np.sqrt(value / max_abs)
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                alpha=0.22 + 0.52 * scale,
                linewidth=0.35 + 3.4 * scale,
                solid_capstyle="round",
                zorder=3,
                label=label,
            )
    _draw_support_points(ax, problem, labels=False)
    err = _relative_matrix_error(plan, reference_plan)
    ax.set_title(f"{title}\nrelative matrix error to optimum={err:.3g}", fontsize=10.5)
    handles = [
        Line2D([0], [0], color="#9AA1AD", lw=2, label="optimal route"),
        Line2D([0], [0], color=EXTRA_COLOR, lw=2, label="extra mass"),
        Line2D([0], [0], color=MISSING_COLOR, lw=2, label="missing mass"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    _finish_geometry_axis(ax, problem)


def _draw_base_routes(
    ax,
    problem: SemiRelaxedOTProblem,
    plan: np.ndarray,
    color: str,
    alpha: float,
    linewidth: float,
    max_entries: int | None,
) -> None:
    for mass, source, target in _route_entries(plan, max_entries=max_entries):
        if mass <= 1e-12:
            continue
        start = problem.source_points[source]
        end = problem.target_points[target]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=1,
        )


def _draw_support_points(
    ax,
    problem: SemiRelaxedOTProblem,
    source_alpha: float = 1.0,
    target_alpha: float = 1.0,
    labels: bool = True,
) -> None:
    source_sizes = 180.0 + 1750.0 * problem.source_weights
    target_sizes = 180.0 + 1750.0 * problem.target_weights
    ax.scatter(
        problem.source_points[:, 0],
        problem.source_points[:, 1],
        s=source_sizes,
        color=SOURCE_COLOR,
        edgecolor="white",
        linewidth=0.9,
        alpha=source_alpha,
        zorder=5,
        label="source",
    )
    ax.scatter(
        problem.target_points[:, 0],
        problem.target_points[:, 1],
        s=target_sizes,
        color=TARGET_COLOR,
        marker="s",
        edgecolor="#4B4F56",
        linewidth=0.65,
        alpha=target_alpha,
        zorder=6,
        label="target",
    )
    if labels:
        for idx, point in enumerate(problem.source_points):
            ax.text(point[0], point[1], f"s{idx}", ha="center", va="center", fontsize=7.5, color="white", zorder=7)
        for idx, point in enumerate(problem.target_points):
            ax.text(point[0], point[1], f"t{idx}", ha="center", va="center", fontsize=7.2, color="#2E3138", zorder=7)
    ax.legend(loc="upper left", fontsize=8, frameon=True)


def _route_title(
    problem: SemiRelaxedOTProblem,
    plan: np.ndarray,
    title: str,
    subtitle: str,
    reference_plan: np.ndarray | None,
) -> str:
    parts = [
        title,
        subtitle,
        f"obj={problem.objective(plan):.4g}, gap={max(problem.duality_gap(plan), 0.0):.2e}",
        f"row err={np.linalg.norm(row_residual(plan, problem.source_weights)):.2e}, routes={np.sum(plan > 1e-12)}",
    ]
    if reference_plan is not None:
        parts.append(f"matrix err to optimum={_relative_matrix_error(plan, reference_plan):.3g}")
    return "\n".join(part for part in parts if part)


def _plot_metric_lines(
    ax,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
) -> None:
    if x_col not in frame.columns or y_col not in frame.columns:
        ax.axis("off")
        return
    for method in [name for name in METHOD_ORDER if name in set(frame["method"])]:
        data = frame[frame["method"] == method].sort_values(x_col)
        y = data[y_col]
        if y_col == "duality_gap":
            y = y.clip(lower=1e-16)
        ax.plot(data[x_col], y, marker="o", markersize=2.3, linewidth=1.55, label=method.upper())
    if y_col == "duality_gap":
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True, fontsize=8)


def _method_subtitle(method: str, frame: pd.DataFrame) -> str:
    if frame.empty or method not in set(frame["method"]):
        return ""
    row = frame[frame["method"] == method].sort_values("iteration").iloc[-1]
    pieces = []
    if "oracle_epochs" in row:
        pieces.append(f"oracle epochs={float(row['oracle_epochs']):.3g}")
    if "algorithm_rounds" in row:
        pieces.append(f"rounds={int(row['algorithm_rounds'])}")
    if "wall_time_sec" in row:
        pieces.append(f"time={float(row['wall_time_sec']):.3g}s")
    return ", ".join(pieces)


def _route_entries(plan: np.ndarray, max_entries: int | None = None) -> list[tuple[float, int, int]]:
    entries = [
        (float(plan[source, target]), source, target)
        for source in range(plan.shape[0])
        for target in range(plan.shape[1])
        if float(plan[source, target]) > 1e-12
    ]
    entries.sort(reverse=True)
    if max_entries is not None:
        entries = entries[:max_entries]
    return entries


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


def _relative_matrix_error(plan: np.ndarray, reference_plan: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference_plan)), 1e-18)
    return float(np.linalg.norm(plan - reference_plan) / denominator)


def _grid_shape(count: int, max_cols: int) -> tuple[int, int]:
    cols = min(max_cols, max(1, count))
    rows = int(np.ceil(count / cols))
    return rows, cols


def _finish_geometry_axis(ax, problem: SemiRelaxedOTProblem) -> None:
    all_points = np.vstack((problem.source_points, problem.target_points))
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    pad = 0.14 * span
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _interpolate_color(left: str, right: str, ratio: float) -> tuple[float, float, float]:
    import matplotlib.colors as mcolors

    left_rgb = np.asarray(mcolors.to_rgb(left))
    right_rgb = np.asarray(mcolors.to_rgb(right))
    return tuple((1.0 - ratio) * left_rgb + ratio * right_rgb)


def _set_route_style() -> None:
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "figure.facecolor": "white",
        "font.size": 10,
        "legend.framealpha": 0.94,
    })
