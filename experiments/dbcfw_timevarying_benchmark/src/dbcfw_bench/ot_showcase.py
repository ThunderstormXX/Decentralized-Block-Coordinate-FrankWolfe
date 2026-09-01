from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

from dbcfw_bench.ot_experiment import (
    OTRunConfig,
    SemiRelaxedOTProblem,
    run_bcfw,
    run_dbcfw_ot,
    run_dfw_ot,
    run_fw,
    solve_balanced_ot_lp,
)
from dbcfw_bench.ot_routes import solve_semirelaxed_ot_reference


@dataclass
class OTShowcaseConfig:
    seed: int = 2026
    agents: int = 6
    epochs: int = 70
    batch: int = 4
    relaxation: float = 0.055
    cost_noise: float = 0.025
    bunny_points: int = 28
    maze_points: int = 24
    maze_width: int = 35
    maze_height: int = 23


SOURCE_COLOR = "#11B3C7"
TARGET_COLOR = "#F4C542"
FLOW_COLOR = "#5C6068"
OPTIMAL_COLOR = "#242933"
EXTRA_COLOR = "#2BAE66"
MISSING_COLOR = "#DD4A48"
METHOD_ORDER = ("fw", "bcfw", "dfw", "dbcfw")


def build_ot_showcase(out_dir: str | Path, config: OTShowcaseConfig | None = None) -> list[Path]:
    config = OTShowcaseConfig() if config is None else config
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _set_showcase_style()

    bunny = make_bunny_problem(config)
    bunny_reference, bunny_message = solve_semirelaxed_ot_reference(
        bunny.problem,
        solve_balanced_ot_lp(bunny.problem),
    )
    bunny_frames, bunny_plans = run_showcase_methods(bunny.problem, config, bunny_reference, seed_offset=0)
    bunny_frames.to_csv(out / "bunny_results.csv", index=False)
    np.savez(out / "bunny_final_plans.npz", optimum=bunny_reference, **bunny_plans)

    maze = make_maze_problem(config)
    maze_reference, maze_message = solve_semirelaxed_ot_reference(
        maze.problem,
        solve_balanced_ot_lp(maze.problem),
    )
    maze_frames, maze_plans = run_showcase_methods(maze.problem, config, maze_reference, seed_offset=700)
    maze_frames.to_csv(out / "maze_results.csv", index=False)
    np.savez(out / "maze_final_plans.npz", optimum=maze_reference, **maze_plans)

    paths = [
        plot_bunny_routes(
            bunny,
            bunny_reference,
            out / "01_bunny_3d_optimal_routes.png",
            title="Stylized 3D bunny: optimal semi-relaxed transport routes",
        ),
        plot_bunny_methods(
            bunny,
            bunny_reference,
            bunny_plans,
            bunny_frames,
            out / "02_bunny_3d_methods_vs_optimum.png",
        ),
        plot_bunny_flow_snapshots(
            bunny,
            bunny_reference,
            out / "03_bunny_3d_flow_snapshots.png",
        ),
        plot_bunny_paper_spectral_distances(
            bunny,
            out / "08_bunny_paper_spectral_distances.png",
        ),
        plot_bunny_paper_eigen_density_samples(
            bunny,
            bunny_reference,
            bunny_plans["dbcfw"],
            out / "09_bunny_paper_eigen_density_samples.png",
        ),
        plot_maze_routes(
            maze,
            maze_reference,
            out / "04_maze_2d_optimal_routes.png",
            title="2D maze: optimal mass routes through corridors",
        ),
        plot_maze_methods(
            maze,
            maze_reference,
            maze_plans,
            maze_frames,
            out / "05_maze_2d_methods_vs_optimum.png",
        ),
        plot_maze_route_differences(
            maze,
            maze_reference,
            maze_plans,
            out / "06_maze_2d_route_differences.png",
        ),
        plot_maze_paper_trajectories(
            maze,
            maze_reference,
            maze_plans["dbcfw"],
            out / "10_maze_paper_trajectories.png",
        ),
        plot_showcase_accounting(
            {"bunny": bunny_frames, "maze": maze_frames},
            out / "07_showcase_convergence_accounting.png",
        ),
    ]
    summary = write_showcase_summary(
        bunny.problem,
        bunny_reference,
        bunny_plans,
        bunny_frames,
        maze.problem,
        maze_reference,
        maze_plans,
        maze_frames,
        out / "showcase_summary.csv",
        bunny_message,
        maze_message,
    )
    paths.append(summary)
    return paths


@dataclass
class SpectralGeometry:
    points: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    source_indices: np.ndarray
    target_indices: np.ndarray
    path_graph: object
    surface_paths: dict[tuple[int, int], np.ndarray]


@dataclass
class BunnyShowcase:
    problem: SemiRelaxedOTProblem
    surface_points: np.ndarray
    spectral: SpectralGeometry


@dataclass
class MazeShowcase:
    problem: SemiRelaxedOTProblem
    maze: np.ndarray
    source_cells: list[tuple[int, int]]
    target_cells: list[tuple[int, int]]
    route_paths: dict[tuple[int, int], np.ndarray]


def make_bunny_problem(config: OTShowcaseConfig) -> BunnyShowcase:
    rng = np.random.default_rng(config.seed)
    surface = np.vstack((
        _ellipsoid_surface((0.00, 0.00, 0.00), (1.05, 0.47, 0.58), 1250, rng),
        _ellipsoid_surface((0.88, 0.00, 0.35), (0.42, 0.32, 0.34), 520, rng),
        _ellipsoid_surface((0.94, -0.18, 0.95), (0.13, 0.075, 0.55), 260, rng),
        _ellipsoid_surface((0.94, 0.18, 0.95), (0.13, 0.075, 0.55), 260, rng),
        _ellipsoid_surface((-1.03, 0.00, 0.16), (0.23, 0.20, 0.22), 180, rng),
        _ellipsoid_surface((-0.18, -0.45, -0.44), (0.38, 0.12, 0.12), 180, rng),
        _ellipsoid_surface((-0.18, 0.45, -0.44), (0.38, 0.12, 0.12), 180, rng),
    ))
    surface += 0.012 * rng.normal(size=surface.shape)

    source_score = (
        np.exp(-3.5 * (surface[:, 0] + 0.55) ** 2 - 2.8 * surface[:, 1] ** 2)
        + 0.45 * np.exp(-14.0 * (surface[:, 2] + 0.32) ** 2)
    )
    target_score = (
        np.exp(-4.0 * (surface[:, 0] - 0.72) ** 2 - 2.2 * surface[:, 1] ** 2)
        + 0.85 * np.exp(-7.5 * (surface[:, 2] - 0.88) ** 2)
        + 0.55 * np.exp(-15.0 * (surface[:, 0] - 1.02) ** 2)
    )
    source_points = _weighted_sample(surface, source_score, config.bunny_points, rng)
    target_points = _weighted_sample(surface, target_score, config.bunny_points, rng)
    source_weights = _smooth_weights(config.bunny_points, rng)
    target_weights = _smooth_weights(config.bunny_points, rng)
    spectral = _build_bunny_spectral_geometry(surface, source_points, target_points, rng)
    base_cost = _biharmonic_pair_cost(spectral)
    local_costs = _agent_costs(base_cost, config.agents, config.cost_noise, rng)
    problem = SemiRelaxedOTProblem(
        source_weights=source_weights,
        target_weights=target_weights,
        source_points=source_points,
        target_points=target_points,
        local_costs=local_costs,
        relaxation=config.relaxation,
    )
    return BunnyShowcase(problem=problem, surface_points=surface, spectral=spectral)


def make_maze_problem(config: OTShowcaseConfig) -> MazeShowcase:
    rng = np.random.default_rng(config.seed + 100)
    maze = _generate_maze(config.maze_height, config.maze_width, rng)
    open_cells = [(row, col) for row, col in zip(*np.where(~maze))]
    source_candidates = [cell for cell in open_cells if cell[1] < config.maze_width * 0.32]
    target_candidates = [cell for cell in open_cells if cell[1] > config.maze_width * 0.68]
    source_cells = _sample_cells(source_candidates, config.maze_points, rng)
    target_cells = _sample_cells(target_candidates, config.maze_points, rng)

    costs = np.zeros((config.maze_points, config.maze_points), dtype=float)
    route_paths: dict[tuple[int, int], np.ndarray] = {}
    for source_idx, source_cell in enumerate(source_cells):
        dist, previous = _bfs_maze(maze, source_cell)
        for target_idx, target_cell in enumerate(target_cells):
            distance = dist[target_cell]
            if not np.isfinite(distance):
                distance = config.maze_width * config.maze_height
            costs[source_idx, target_idx] = distance
            route_paths[(source_idx, target_idx)] = _reconstruct_path(previous, source_cell, target_cell)
    costs = costs / max(float(costs.max()), 1.0)
    costs += 0.03 * _normalized_squared_cost(_cell_points(source_cells), _cell_points(target_cells))

    source_weights = _smooth_weights(config.maze_points, rng)
    target_weights = _smooth_weights(config.maze_points, rng)
    local_costs = _agent_costs(costs, config.agents, config.cost_noise, rng)
    problem = SemiRelaxedOTProblem(
        source_weights=source_weights,
        target_weights=target_weights,
        source_points=_cell_points(source_cells),
        target_points=_cell_points(target_cells),
        local_costs=local_costs,
        relaxation=config.relaxation,
    )
    return MazeShowcase(
        problem=problem,
        maze=maze,
        source_cells=source_cells,
        target_cells=target_cells,
        route_paths=route_paths,
    )


def run_showcase_methods(
    problem: SemiRelaxedOTProblem,
    config: OTShowcaseConfig,
    reference_plan: np.ndarray,
    seed_offset: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    run_config = OTRunConfig(
        methods=METHOD_ORDER,
        m=problem.m,
        n=problem.n,
        agents=problem.agents,
        epochs=config.epochs,
        batch=max(1, min(config.batch, problem.n)),
        relaxation=problem.relaxation,
        cost_noise=config.cost_noise,
        edge_prob=0.68,
        seed=config.seed + seed_offset,
        graph_seed=config.seed + seed_offset + 31,
        log_every=5,
    )
    frames: list[pd.DataFrame] = []
    plans: dict[str, np.ndarray] = {}
    runners = {
        "fw": run_fw,
        "bcfw": run_bcfw,
        "dfw": run_dfw_ot,
        "dbcfw": run_dbcfw_ot,
    }
    for method in METHOD_ORDER:
        frame, plan = runners[method](problem, run_config, reference_plan)
        frames.append(frame)
        plans[method] = plan
    return pd.concat(frames, ignore_index=True), plans


def plot_bunny_routes(
    bunny: BunnyShowcase,
    plan: np.ndarray,
    path: str | Path,
    title: str,
) -> Path:
    path = Path(path)
    fig = plt.figure(figsize=(12, 8.3))
    ax = fig.add_subplot(111, projection="3d")
    _draw_bunny_surface(ax, bunny.surface_points)
    _draw_bunny_plan(ax, bunny.problem, plan, route_color=OPTIMAL_COLOR, max_routes=90, spectral=bunny.spectral)
    _finish_bunny_axis(ax)
    ax.set_title(_plan_title(bunny.problem, plan, title), pad=18, fontsize=15)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.90)
    fig.savefig(path, dpi=195, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_bunny_methods(
    bunny: BunnyShowcase,
    reference_plan: np.ndarray,
    plans: dict[str, np.ndarray],
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    path = Path(path)
    panels = [("optimum", reference_plan)] + [(name, plans[name]) for name in METHOD_ORDER if name in plans]
    fig = plt.figure(figsize=(18, 10.7))
    for idx, (name, plan) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        _draw_bunny_surface(ax, bunny.surface_points, alpha=0.075)
        color = OPTIMAL_COLOR if name == "optimum" else FLOW_COLOR
        _draw_bunny_plan(ax, bunny.problem, plan, route_color=color, max_routes=58, spectral=bunny.spectral)
        _finish_bunny_axis(ax)
        title = "OPTIMUM" if name == "optimum" else name.upper()
        subtitle = "" if name == "optimum" else _method_accounting(name, frame)
        error = "" if name == "optimum" else f"\nerr to opt={_relative_error(plan, reference_plan):.3g}"
        ax.set_title(_plan_title(bunny.problem, plan, title, subtitle) + error, fontsize=10.5, pad=12)
    fig.suptitle("3D bunny transport: final method routes versus optimum", fontsize=16, y=0.985)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.84, wspace=0.02, hspace=0.30)
    fig.savefig(path, dpi=195, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_bunny_flow_snapshots(bunny: BunnyShowcase, plan: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    times = np.linspace(0.0, 1.0, 4)
    fig = plt.figure(figsize=(18, 5.7))
    entries = _plan_entries(plan, max_entries=120)
    max_mass = max((mass for mass, _, _ in entries), default=1e-16)
    for idx, time_value in enumerate(times, start=1):
        ax = fig.add_subplot(1, len(times), idx, projection="3d")
        _draw_bunny_surface(ax, bunny.surface_points, alpha=0.055)
        moving = []
        sizes = []
        for mass, source, target in entries:
            path_points = bunny.spectral.surface_paths.get(
                (source, target),
                np.vstack((bunny.problem.source_points[source], bunny.problem.target_points[target])),
            )
            moving.append(_point_along_path(path_points, float(time_value)))
            sizes.append(24.0 + 210.0 * np.sqrt(mass / max_mass))
        moving_arr = np.asarray(moving)
        color = _interpolate_color(SOURCE_COLOR, TARGET_COLOR, float(time_value))
        ax.scatter(
            moving_arr[:, 0],
            moving_arr[:, 1],
            moving_arr[:, 2],
            s=sizes,
            c=[color],
            edgecolors="white",
            linewidths=0.25,
            alpha=0.84,
            zorder=5,
        )
        _draw_bunny_supports(ax, bunny.problem, alpha=0.30)
        _finish_bunny_axis(ax)
        ax.set_title(f"t={time_value:.2g}", fontsize=12)
    fig.suptitle("3D bunny flow snapshots along optimal routes", fontsize=16, y=0.99)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.88, wspace=0.0)
    fig.savefig(path, dpi=195, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_bunny_paper_spectral_distances(bunny: BunnyShowcase, path: str | Path) -> Path:
    path = Path(path)
    spectral = bunny.spectral
    source = int(spectral.source_indices[np.argmin(bunny.problem.source_points[:, 0])])
    values = {
        "ambient Euclidean": np.linalg.norm(spectral.points - spectral.points[source], axis=1),
        "biharmonic spectral": _biharmonic_distance_to_source(spectral, source),
        "diffusion tau=0.03": _diffusion_distance_to_source(spectral, source, tau=0.03),
        "diffusion tau=0.18": _diffusion_distance_to_source(spectral, source, tau=0.18),
    }
    fig = plt.figure(figsize=(18.2, 5.7))
    for idx, (title, distance) in enumerate(values.items(), start=1):
        ax = fig.add_subplot(1, len(values), idx, projection="3d")
        normalized = distance / max(float(np.quantile(distance, 0.98)), 1e-16)
        scatter = ax.scatter(
            spectral.points[:, 0],
            spectral.points[:, 1],
            spectral.points[:, 2],
            s=8,
            c=np.clip(normalized, 0.0, 1.0),
            cmap="magma",
            alpha=0.78,
            linewidths=0,
            depthshade=True,
        )
        point = spectral.points[source]
        ax.scatter(
            [point[0]],
            [point[1]],
            [point[2]],
            s=185,
            color=SOURCE_COLOR,
            edgecolors="white",
            linewidths=1.0,
            depthshade=True,
        )
        _finish_bunny_axis(ax, show_legend=False)
        ax.set_title(title, fontsize=11)
        fig.colorbar(scatter, ax=ax, shrink=0.52, pad=0.0, fraction=0.045)
    fig.suptitle("Paper-style Figure 3 analogue: premetric distances to one bunny source point", fontsize=16, y=0.985)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.86, wspace=0.0)
    fig.savefig(path, dpi=195, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_bunny_paper_eigen_density_samples(
    bunny: BunnyShowcase,
    reference_plan: np.ndarray,
    dbcfw_plan: np.ndarray,
    path: str | Path,
) -> Path:
    path = Path(path)
    spectral = bunny.spectral
    eigen_indices = [idx for idx in (2, 5, 9) if idx < spectral.eigenvectors.shape[1]]
    while len(eigen_indices) < 3:
        eigen_indices.append(len(eigen_indices) + 1)
    fig = plt.figure(figsize=(18, 10.1))
    for panel, eigen_index in enumerate(eigen_indices[:3], start=1):
        ax = fig.add_subplot(2, 3, panel, projection="3d")
        values = spectral.eigenvectors[:, eigen_index]
        scatter = ax.scatter(
            spectral.points[:, 0],
            spectral.points[:, 1],
            spectral.points[:, 2],
            s=8,
            c=values,
            cmap="coolwarm",
            alpha=0.84,
            linewidths=0,
            depthshade=True,
        )
        _finish_bunny_axis(ax, show_legend=False)
        ax.set_title(f"eigenfunction phi_{eigen_index}", fontsize=11)
        fig.colorbar(scatter, ax=ax, shrink=0.52, pad=0.0, fraction=0.045)

    density = _support_kernel_density(spectral.points, bunny.problem.target_points, bunny.problem.target_weights)
    bottom_specs = [
        ("target density on bunny", density, None),
        ("optimal transported samples", None, reference_plan),
        ("DBCFW transported samples", None, dbcfw_plan),
    ]
    for offset, (title, density_values, plan) in enumerate(bottom_specs, start=4):
        ax = fig.add_subplot(2, 3, offset, projection="3d")
        if density_values is not None:
            scatter = ax.scatter(
                spectral.points[:, 0],
                spectral.points[:, 1],
                spectral.points[:, 2],
                s=8,
                c=density_values,
                cmap="viridis",
                alpha=0.82,
                linewidths=0,
                depthshade=True,
            )
            fig.colorbar(scatter, ax=ax, shrink=0.52, pad=0.0, fraction=0.045)
            _draw_bunny_supports(ax, bunny.problem, alpha=0.82)
        else:
            samples = _sample_transport_endpoints(plan, bunny.problem, count=420, seed=909 + offset)
            _draw_bunny_surface(ax, bunny.surface_points, alpha=0.045)
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                samples[:, 2],
                s=34,
                color=TARGET_COLOR,
                edgecolors="#353941",
                linewidths=0.25,
                alpha=0.78,
                depthshade=True,
            )
            _draw_bunny_supports(ax, bunny.problem, alpha=0.28)
        _finish_bunny_axis(ax, show_legend=offset == 4)
        ax.set_title(title, fontsize=11)
    fig.suptitle("Paper-style Figure 4 analogue: bunny eigenfunctions, density, and transported samples", fontsize=16, y=0.985)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.88, wspace=0.02, hspace=0.22)
    fig.savefig(path, dpi=195, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_maze_routes(maze: MazeShowcase, plan: np.ndarray, path: str | Path, title: str) -> Path:
    path = Path(path)
    fig, ax = plt.subplots(figsize=(13.2, 8.2))
    _draw_maze_background(ax, maze.maze)
    _draw_maze_plan(ax, maze, plan, route_color=OPTIMAL_COLOR, max_routes=95)
    ax.set_title(_plan_title(maze.problem, plan, title), fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=195)
    plt.close(fig)
    return path


def plot_maze_methods(
    maze: MazeShowcase,
    reference_plan: np.ndarray,
    plans: dict[str, np.ndarray],
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    path = Path(path)
    panels = [("optimum", reference_plan)] + [(name, plans[name]) for name in METHOD_ORDER if name in plans]
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10.6))
    axes_flat = axes.ravel()
    for ax, (name, plan) in zip(axes_flat, panels):
        _draw_maze_background(ax, maze.maze)
        _draw_maze_plan(
            ax,
            maze,
            plan,
            route_color=OPTIMAL_COLOR if name == "optimum" else FLOW_COLOR,
            max_routes=78,
        )
        title = "OPTIMUM" if name == "optimum" else name.upper()
        subtitle = "" if name == "optimum" else _method_accounting(name, frame)
        error = "" if name == "optimum" else f"\nerr to opt={_relative_error(plan, reference_plan):.3g}"
        ax.set_title(_plan_title(maze.problem, plan, title, subtitle) + error, fontsize=10.5)
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("2D maze transport: routes must follow corridors", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=195)
    plt.close(fig)
    return path


def plot_maze_route_differences(
    maze: MazeShowcase,
    reference_plan: np.ndarray,
    plans: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    path = Path(path)
    fig, axes = plt.subplots(2, 2, figsize=(15.7, 10.2))
    for ax, method in zip(axes.ravel(), METHOD_ORDER):
        _draw_maze_background(ax, maze.maze)
        _draw_maze_difference(ax, maze, plans[method], reference_plan)
        ax.set_title(f"{method.upper()}\nerr to optimum={_relative_error(plans[method], reference_plan):.3g}")
    fig.suptitle("2D maze differences: green extra mass, red missing mass", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=195)
    plt.close(fig)
    return path


def plot_maze_paper_trajectories(
    maze: MazeShowcase,
    reference_plan: np.ndarray,
    dbcfw_plan: np.ndarray,
    path: str | Path,
) -> Path:
    path = Path(path)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.6))
    panels = [
        ("(a) source and target distributions", "supports", reference_plan),
        ("(b) optimal sample trajectories", "routes", reference_plan),
        ("(c) same setup, DBCFW final plan", "supports", dbcfw_plan),
        ("(d) DBCFW sample trajectories", "routes", dbcfw_plan),
    ]
    for ax, (title, mode, plan) in zip(axes.ravel(), panels):
        _draw_maze_background(ax, maze.maze)
        if mode == "supports":
            if plan is dbcfw_plan:
                _draw_maze_plan(ax, maze, dbcfw_plan, route_color="#A0A6B1", max_routes=36)
            else:
                _draw_maze_supports(ax, maze.problem)
        else:
            _draw_maze_plan(ax, maze, plan, route_color=OPTIMAL_COLOR if plan is reference_plan else FLOW_COLOR, max_routes=70)
        ax.set_title(title, fontsize=12)
    fig.suptitle("Paper-style Figure 6 analogue: maze distributions and sample trajectories", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=195)
    plt.close(fig)
    return path


def plot_showcase_accounting(frames: dict[str, pd.DataFrame], path: str | Path) -> Path:
    path = Path(path)
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5), sharey="row")
    x_axes = [("oracle_epochs", "oracle epochs"), ("algorithm_rounds", "algorithm rounds"), ("wall_time_sec", "wall time")]
    for row_idx, (name, frame) in enumerate(frames.items()):
        for col_idx, (x_col, label) in enumerate(x_axes):
            ax = axes[row_idx, col_idx]
            for method in METHOD_ORDER:
                data = frame[frame["method"] == method].sort_values(x_col)
                ax.plot(
                    data[x_col],
                    data["duality_gap"].clip(lower=1e-16),
                    marker="o",
                    markersize=2.2,
                    linewidth=1.45,
                    label=method.upper(),
                )
            ax.set_yscale("log")
            ax.set_title(f"{name}: dual gap vs {label}")
            ax.set_xlabel(label)
            ax.grid(True, alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("duality gap")
            if row_idx == 0 and col_idx == 2:
                ax.legend(frameon=True, fontsize=8)
    fig.suptitle("Showcase accounting: oracle work, rounds, and time tell different stories", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def write_showcase_summary(
    bunny_problem: SemiRelaxedOTProblem,
    bunny_reference: np.ndarray,
    bunny_plans: dict[str, np.ndarray],
    bunny_frame: pd.DataFrame,
    maze_problem: SemiRelaxedOTProblem,
    maze_reference: np.ndarray,
    maze_plans: dict[str, np.ndarray],
    maze_frame: pd.DataFrame,
    path: str | Path,
    bunny_message: str,
    maze_message: str,
) -> Path:
    rows = []
    for setup, problem, reference, plans, frame, message in (
        ("bunny", bunny_problem, bunny_reference, bunny_plans, bunny_frame, bunny_message),
        ("maze", maze_problem, maze_reference, maze_plans, maze_frame, maze_message),
    ):
        final = frame.sort_values("iteration").groupby("method", as_index=False).tail(1).set_index("method")
        for method in METHOD_ORDER:
            plan = plans[method]
            row = {
                "setup": setup,
                "method": method,
                "objective": problem.objective(plan),
                "duality_gap": max(problem.duality_gap(plan), 0.0),
                "matrix_error_to_optimum": _relative_error(plan, reference),
                "support_routes": int(np.sum(plan > 1e-12)),
                "reference_message": message,
            }
            for column in ("oracle_epochs", "algorithm_rounds", "communication_rounds", "wall_time_sec", "gamma"):
                row[column] = final.loc[method, column]
            rows.append(row)
    path = Path(path)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _ellipsoid_surface(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.arange(count, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / count)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * indices
    sphere = np.column_stack((
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ))
    sphere += 0.025 * rng.normal(size=sphere.shape)
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    return np.asarray(center) + sphere * np.asarray(radii)


def _weighted_sample(
    points: np.ndarray,
    score: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    probability = np.maximum(score, 1e-10)
    probability = probability / probability.sum()
    indices = rng.choice(points.shape[0], size=count, replace=False, p=probability)
    return points[indices]


def _build_bunny_spectral_geometry(
    surface_points: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    rng: np.random.Generator,
    eigen_count: int = 30,
    background_count: int = 1200,
) -> SpectralGeometry:
    count = min(background_count, surface_points.shape[0])
    background = surface_points[rng.choice(surface_points.shape[0], size=count, replace=False)]
    points = np.vstack((source_points, target_points, background))
    eigenvalues, eigenvectors, path_graph = _graph_laplacian_eigenvectors(points, eigen_count=eigen_count)
    source_indices = np.arange(source_points.shape[0])
    target_indices = np.arange(source_points.shape[0], source_points.shape[0] + target_points.shape[0])
    return SpectralGeometry(
        points=points,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        source_indices=source_indices,
        target_indices=target_indices,
        path_graph=path_graph,
        surface_paths=_precompute_surface_paths(path_graph, points, source_indices, target_indices),
    )


def _graph_laplacian_eigenvectors(
    points: np.ndarray,
    eigen_count: int,
    neighbors: int = 12,
) -> tuple[np.ndarray, np.ndarray, object]:
    affinity_graph, path_graph = _knn_graphs(points, neighbors)
    laplacian = csgraph.laplacian(affinity_graph, normed=True)
    k = max(3, min(eigen_count, points.shape[0] - 2))
    try:
        eigenvalues, eigenvectors = eigsh(laplacian, k=k, which="SM", tol=1e-3, maxiter=5000)
    except Exception:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)
    for idx in range(eigenvectors.shape[1]):
        if float(np.sum(eigenvectors[:, idx])) < 0.0:
            eigenvectors[:, idx] *= -1.0
    return eigenvalues, eigenvectors, path_graph


def _knn_graphs(
    points: np.ndarray,
    neighbors: int,
) -> tuple[object, object]:
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(neighbors + 1, points.shape[0]))
    sigma = max(float(np.median(distances[:, 1:])), 1e-6)
    rows = np.repeat(np.arange(points.shape[0]), indices.shape[1] - 1)
    cols = indices[:, 1:].reshape(-1)
    edge_lengths = np.maximum(distances[:, 1:].reshape(-1), 1e-8)
    affinity_weights = np.exp(-(edge_lengths ** 2) / (2.0 * sigma * sigma))
    affinity_graph = coo_matrix((affinity_weights, (rows, cols)), shape=(points.shape[0], points.shape[0]))
    path_graph = coo_matrix((edge_lengths, (rows, cols)), shape=(points.shape[0], points.shape[0]))
    return (0.5 * (affinity_graph + affinity_graph.T)).tocsr(), path_graph.minimum(path_graph.T).tocsr()


def _biharmonic_pair_cost(spectral: SpectralGeometry) -> np.ndarray:
    features = _biharmonic_features(spectral)
    source_features = features[spectral.source_indices]
    target_features = features[spectral.target_indices]
    diff = source_features[:, None, :] - target_features[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    return cost / max(float(cost.max()), 1e-16)


def _precompute_surface_paths(
    path_graph,
    points: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    _, predecessors = csgraph.dijkstra(
        path_graph,
        directed=False,
        indices=source_indices,
        return_predecessors=True,
    )
    paths: dict[tuple[int, int], np.ndarray] = {}
    for source_pos, source_index in enumerate(source_indices):
        predecessor_row = predecessors[source_pos]
        for target_pos, target_index in enumerate(target_indices):
            node_path = _reconstruct_graph_path(predecessor_row, int(source_index), int(target_index))
            if not node_path:
                path_points = np.vstack((points[int(source_index)], points[int(target_index)]))
            else:
                path_points = points[np.asarray(node_path, dtype=int)]
            paths[(source_pos, target_pos)] = path_points
    return paths


def _reconstruct_graph_path(
    predecessors: np.ndarray,
    source: int,
    target: int,
) -> list[int]:
    if source == target:
        return [source]
    current = int(target)
    path = [current]
    limit = predecessors.size + 1
    while current != source and len(path) <= limit:
        current = int(predecessors[current])
        if current < 0:
            return []
        path.append(current)
    if path[-1] != source:
        return []
    path.reverse()
    return path


def _biharmonic_features(spectral: SpectralGeometry) -> np.ndarray:
    eigenvalues = spectral.eigenvalues[1:]
    eigenvectors = spectral.eigenvectors[:, 1:]
    weights = 1.0 / np.maximum(eigenvalues, 1e-5)
    return eigenvectors * weights[None, :]


def _biharmonic_distance_to_source(spectral: SpectralGeometry, source: int) -> np.ndarray:
    features = _biharmonic_features(spectral)
    diff = features - features[source]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=1), 0.0))


def _diffusion_distance_to_source(spectral: SpectralGeometry, source: int, tau: float) -> np.ndarray:
    eigenvalues = spectral.eigenvalues[1:]
    eigenvectors = spectral.eigenvectors[:, 1:]
    weights = np.exp(-tau * np.maximum(eigenvalues, 0.0))
    features = eigenvectors * weights[None, :]
    diff = features - features[source]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=1), 0.0))


def _support_kernel_density(
    points: np.ndarray,
    supports: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    diff = points[:, None, :] - supports[None, :, :]
    squared = np.sum(diff * diff, axis=2)
    bandwidth = max(float(np.quantile(np.sqrt(squared), 0.18)), 1e-3)
    density = np.exp(-squared / (2.0 * bandwidth * bandwidth)) @ weights
    return density / max(float(density.max()), 1e-16)


def _sample_transport_endpoints(
    plan: np.ndarray,
    problem: SemiRelaxedOTProblem,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = plan.reshape(-1)
    weights = weights / max(float(weights.sum()), 1e-16)
    choices = rng.choice(weights.size, size=count, replace=True, p=weights)
    _, target_indices = np.unravel_index(choices, plan.shape)
    samples = problem.target_points[target_indices].copy()
    samples += 0.012 * rng.normal(size=samples.shape)
    return samples


def _smooth_weights(count: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.gamma(shape=7.5, scale=1.0, size=count)
    return raw / raw.sum()


def _normalized_squared_cost(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    diff = source_points[:, None, :] - target_points[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    max_cost = max(float(cost.max()), 1e-16)
    return cost / max_cost


def _agent_costs(
    base_cost: np.ndarray,
    agents: int,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    local_costs = np.empty((agents, *base_cost.shape), dtype=float)
    for agent in range(agents):
        perturbation = noise * rng.normal(size=base_cost.shape)
        local_costs[agent] = np.clip(base_cost + perturbation, 0.0, None)
    return local_costs


def _generate_maze(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    height = height if height % 2 == 1 else height + 1
    width = width if width % 2 == 1 else width + 1
    maze = np.ones((height, width), dtype=bool)
    start = (1, 1)
    maze[start] = False
    stack = [start]
    directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    while stack:
        row, col = stack[-1]
        order = rng.permutation(len(directions))
        carved = False
        for idx in order:
            dr, dc = directions[int(idx)]
            nr, nc = row + dr, col + dc
            if 1 <= nr < height - 1 and 1 <= nc < width - 1 and maze[nr, nc]:
                maze[row + dr // 2, col + dc // 2] = False
                maze[nr, nc] = False
                stack.append((nr, nc))
                carved = True
                break
        if not carved:
            stack.pop()
    return maze


def _sample_cells(cells: list[tuple[int, int]], count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if len(cells) < count:
        raise ValueError(f"not enough open maze cells: need {count}, have {len(cells)}")
    indices = rng.choice(len(cells), size=count, replace=False)
    return [cells[int(idx)] for idx in indices]


def _bfs_maze(
    maze: np.ndarray,
    start: tuple[int, int],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, int]]]:
    dist = {start: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        row, col = queue.popleft()
        for nr, nc in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
            if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and not maze[nr, nc]:
                if (nr, nc) not in dist:
                    dist[(nr, nc)] = dist[(row, col)] + 1.0
                    previous[(nr, nc)] = (row, col)
                    queue.append((nr, nc))
    return dist, previous


def _reconstruct_path(
    previous: dict[tuple[int, int], tuple[int, int]],
    source: tuple[int, int],
    target: tuple[int, int],
) -> np.ndarray:
    current = target
    path = [current]
    while current != source and current in previous:
        current = previous[current]
        path.append(current)
    path.reverse()
    return _cell_points(path)


def _cell_points(cells: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([(col, -row) for row, col in cells], dtype=float)


def _draw_bunny_surface(ax, surface: np.ndarray, alpha: float = 0.14) -> None:
    ax.scatter(
        surface[:, 0],
        surface[:, 1],
        surface[:, 2],
        s=6,
        c=surface[:, 2],
        cmap="Greys",
        alpha=alpha,
        linewidths=0,
        depthshade=True,
        zorder=0,
    )


def _draw_bunny_plan(
    ax,
    problem: SemiRelaxedOTProblem,
    plan: np.ndarray,
    route_color: str,
    max_routes: int,
    spectral: SpectralGeometry | None = None,
) -> None:
    entries = _plan_entries(plan, max_entries=max_routes)
    max_mass = max((mass for mass, _, _ in entries), default=1e-16)
    for mass, source, target in reversed(entries):
        if spectral is None:
            path_points = np.vstack((problem.source_points[source], problem.target_points[target]))
        else:
            path_points = spectral.surface_paths.get(
                (source, target),
                np.vstack((problem.source_points[source], problem.target_points[target])),
            )
        scale = np.sqrt(mass / max_mass)
        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            path_points[:, 2],
            color=route_color,
            alpha=0.18 + 0.50 * scale,
            linewidth=0.35 + 2.8 * scale,
            solid_capstyle="round",
            zorder=2,
        )
    _draw_bunny_supports(ax, problem, alpha=1.0)


def _draw_bunny_supports(ax, problem: SemiRelaxedOTProblem, alpha: float) -> None:
    ax.scatter(
        problem.source_points[:, 0],
        problem.source_points[:, 1],
        problem.source_points[:, 2],
        s=70 + 1200 * problem.source_weights,
        color=SOURCE_COLOR,
        edgecolors="white",
        linewidths=0.65,
        alpha=alpha,
        depthshade=True,
        label="source",
        zorder=5,
    )
    ax.scatter(
        problem.target_points[:, 0],
        problem.target_points[:, 1],
        problem.target_points[:, 2],
        s=70 + 1200 * problem.target_weights,
        color=TARGET_COLOR,
        marker="s",
        edgecolors="#353941",
        linewidths=0.55,
        alpha=alpha,
        depthshade=True,
        label="target",
        zorder=6,
    )


def _finish_bunny_axis(ax, show_legend: bool = True) -> None:
    ax.view_init(elev=18, azim=-58)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.78, 0.78)
    ax.set_zlim(-0.66, 1.62)
    ax.set_axis_off()
    ax.set_box_aspect((2.35, 1.45, 1.65))
    ax.set_proj_type("persp", focal_length=0.72)
    try:
        ax.dist = 6.0
    except AttributeError:
        pass
    if show_legend:
        ax.legend(loc="upper left", frameon=True, fontsize=8)


def _draw_maze_background(ax, maze: np.ndarray) -> None:
    image = np.where(maze, 0.0, 1.0)
    ax.imshow(image, cmap="gray", interpolation="nearest", origin="upper")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_maze_plan(
    ax,
    maze: MazeShowcase,
    plan: np.ndarray,
    route_color: str,
    max_routes: int,
) -> None:
    entries = _plan_entries(plan, max_entries=max_routes)
    max_mass = max((mass for mass, _, _ in entries), default=1e-16)
    for mass, source, target in reversed(entries):
        points = maze.route_paths[(source, target)]
        scale = np.sqrt(mass / max_mass)
        ax.plot(
            points[:, 0],
            -points[:, 1],
            color=route_color,
            alpha=0.14 + 0.50 * scale,
            linewidth=0.35 + 3.4 * scale,
            solid_capstyle="round",
            zorder=2,
        )
    _draw_maze_supports(ax, maze.problem)


def _draw_maze_supports(ax, problem: SemiRelaxedOTProblem) -> None:
    ax.scatter(
        problem.source_points[:, 0],
        -problem.source_points[:, 1],
        s=48 + 1120 * problem.source_weights,
        color=SOURCE_COLOR,
        edgecolors="white",
        linewidths=0.75,
        zorder=5,
        label="source",
    )
    ax.scatter(
        problem.target_points[:, 0],
        -problem.target_points[:, 1],
        s=48 + 1120 * problem.target_weights,
        color=TARGET_COLOR,
        marker="s",
        edgecolors="#353941",
        linewidths=0.55,
        zorder=6,
        label="target",
    )
    ax.legend(loc="upper left", frameon=True, fontsize=8)


def _draw_maze_difference(
    ax,
    maze: MazeShowcase,
    plan: np.ndarray,
    reference_plan: np.ndarray,
) -> None:
    _draw_maze_plan(ax, maze, reference_plan, route_color="#8C939E", max_routes=70)
    diff = plan - reference_plan
    max_abs = max(float(np.max(np.abs(diff))), 1e-16)
    for sign, color in ((1.0, EXTRA_COLOR), (-1.0, MISSING_COLOR)):
        entries = []
        for source in range(diff.shape[0]):
            for target in range(diff.shape[1]):
                value = sign * float(diff[source, target])
                if value > 1e-12:
                    entries.append((value, source, target))
        entries.sort(reverse=True)
        for value, source, target in entries[:70]:
            points = maze.route_paths[(source, target)]
            scale = np.sqrt(value / max_abs)
            ax.plot(
                points[:, 0],
                -points[:, 1],
                color=color,
                alpha=0.18 + 0.56 * scale,
                linewidth=0.35 + 3.0 * scale,
                solid_capstyle="round",
                zorder=4,
            )
    _draw_maze_supports(ax, maze.problem)
    handles = [
        Line2D([0], [0], color="#8C939E", lw=2.0, label="optimal route"),
        Line2D([0], [0], color=EXTRA_COLOR, lw=2.0, label="extra mass"),
        Line2D([0], [0], color=MISSING_COLOR, lw=2.0, label="missing mass"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)


def _plan_entries(plan: np.ndarray, max_entries: int | None = None) -> list[tuple[float, int, int]]:
    entries = [
        (float(plan[source, target]), source, target)
        for source in range(plan.shape[0])
        for target in range(plan.shape[1])
        if float(plan[source, target]) > 1e-12
    ]
    entries.sort(reverse=True)
    if max_entries is not None:
        return entries[:max_entries]
    return entries


def _point_along_path(path_points: np.ndarray, time_value: float) -> np.ndarray:
    if path_points.shape[0] <= 1:
        return path_points[0]
    segments = path_points[1:] - path_points[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    total = float(lengths.sum())
    if total <= 1e-16:
        return path_points[-1].copy()
    target = float(np.clip(time_value, 0.0, 1.0)) * total
    cumulative = np.cumsum(lengths)
    index = int(np.searchsorted(cumulative, target, side="left"))
    if index >= lengths.size:
        return path_points[-1].copy()
    previous = 0.0 if index == 0 else float(cumulative[index - 1])
    local = (target - previous) / max(float(lengths[index]), 1e-16)
    return (1.0 - local) * path_points[index] + local * path_points[index + 1]


def _plan_title(
    problem: SemiRelaxedOTProblem,
    plan: np.ndarray,
    title: str,
    subtitle: str = "",
) -> str:
    parts = [
        title,
        subtitle,
        f"obj={problem.objective(plan):.4g}, gap={max(problem.duality_gap(plan), 0.0):.2e}, routes={np.sum(plan > 1e-12)}",
    ]
    return "\n".join(part for part in parts if part)


def _method_accounting(method: str, frame: pd.DataFrame) -> str:
    row = frame[frame["method"] == method].sort_values("iteration").iloc[-1]
    return (
        f"epochs={float(row['oracle_epochs']):.3g}, "
        f"rounds={int(row['algorithm_rounds'])}, "
        f"time={float(row['wall_time_sec']):.3g}s"
    )


def _relative_error(plan: np.ndarray, reference_plan: np.ndarray) -> float:
    return float(np.linalg.norm(plan - reference_plan) / max(float(np.linalg.norm(reference_plan)), 1e-18))


def _interpolate_color(left: str, right: str, ratio: float) -> tuple[float, float, float]:
    import matplotlib.colors as mcolors

    left_rgb = np.asarray(mcolors.to_rgb(left))
    right_rgb = np.asarray(mcolors.to_rgb(right))
    return tuple((1.0 - ratio) * left_rgb + ratio * right_rgb)


def _set_showcase_style() -> None:
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "figure.facecolor": "white",
        "font.size": 10,
        "legend.framealpha": 0.94,
    })
