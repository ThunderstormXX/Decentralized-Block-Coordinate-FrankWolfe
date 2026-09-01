from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import networkx as nx
import numpy as np
import pandas as pd
import yaml

from dbcfw_bench.comm_graphs import GraphConfig, sample_graph
from dbcfw_bench.config import graph_name
from dbcfw_bench.ot_experiment import (
    OTPaperConfig,
    _final_rows,
    _paper_run_config,
    make_semirelaxed_ot_problem,
    run_dbcfw_ot,
    run_dfw_ot,
    solve_balanced_ot_lp,
)


def build_ot_gallery(run_dir: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    root = Path(run_dir)
    out = Path(out_dir) if out_dir is not None else root / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / "paper_suite_results.csv")
    config = _load_paper_config(root / "paper_config.yaml")
    _set_style()
    paths = [
        plot_scorecard(frame, out / "gallery_dbcfw_vs_dfw_scorecard.png"),
        plot_gap_landscape(frame, out / "gallery_duality_gap_landscape.png"),
        plot_consensus_network(frame, config, out / "gallery_consensus_network.png"),
        plot_transport_geometry(config, out / "gallery_transport_geometry.png"),
    ]
    return paths


def plot_scorecard(frame: pd.DataFrame, path: str | Path) -> Path:
    final = _final_rows(frame)
    metrics = [
        ("objective", "objective", "lower"),
        ("duality_gap", "duality gap", "lower"),
        ("marginal_constraint_error", "marginal error", "lower"),
        ("sparsity", "sparsity", "higher"),
        ("transport_matrix_error", "matrix error", "lower"),
        ("value_error", "value error", "lower"),
        ("wall_time_sec", "wall time", "lower"),
    ]
    lambdas = sorted(final["sweep_relaxation"].unique())
    values = np.zeros((len(metrics), len(lambdas)), dtype=float)
    annotations: list[list[str]] = []
    for row, (metric, _, direction) in enumerate(metrics):
        row_annotations = []
        for col, relaxation in enumerate(lambdas):
            sub = final[final["sweep_relaxation"] == relaxation].set_index("method")
            dfw = float(sub.loc["dfw", metric])
            dbcfw = float(sub.loc["dbcfw", metric])
            if direction == "lower":
                ratio = max(dfw, 1e-16) / max(dbcfw, 1e-16)
            else:
                ratio = max(dbcfw, 1e-16) / max(dfw, 1e-16)
            values[row, col] = np.log10(ratio)
            row_annotations.append(f"{ratio:.1f}x")
        annotations.append(row_annotations)
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.35, vmax=1.35)
    image = ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(lambdas)), [f"{value:g}" for value in lambdas])
    ax.set_yticks(np.arange(len(metrics)), [label for _, label, _ in metrics])
    ax.set_xlabel("relaxation lambda")
    ax.set_title("DBCFW vs DFW scorecard: green means DBCFW wins")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, annotations[row][col], ha="center", va="center", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86)
    colorbar.set_label("log10(improvement factor)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return Path(path)


def plot_gap_landscape(frame: pd.DataFrame, path: str | Path) -> Path:
    lambdas = sorted(frame["sweep_relaxation"].unique())
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(lambdas)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, method in zip(axes, ["dfw", "dbcfw"]):
        method_frame = frame[frame["method"] == method]
        for color, relaxation in zip(colors, lambdas):
            data = method_frame[method_frame["sweep_relaxation"] == relaxation].sort_values("oracle_epochs")
            ax.plot(
                data["oracle_epochs"],
                data["duality_gap"].clip(lower=1e-16),
                color=color,
                linewidth=1.8,
                label=f"lambda={relaxation:g}",
            )
        ax.set_title(method.upper())
        ax.set_xlabel("oracle epochs")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.28)
    axes[0].set_ylabel("duality gap")
    axes[1].legend(frameon=True, fontsize=9, loc="upper right")
    fig.suptitle("Duality-gap landscape across relaxation setups")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return Path(path)


def plot_consensus_network(frame: pd.DataFrame, config: OTPaperConfig, path: str | Path) -> Path:
    dbcfw = frame[frame["method"] == "dbcfw"].copy()
    lambdas = sorted(dbcfw["sweep_relaxation"].unique())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(lambdas)))
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))
    for color, relaxation in zip(colors, lambdas):
        data = dbcfw[dbcfw["sweep_relaxation"] == relaxation].sort_values("oracle_epochs")
        axes[0].plot(
            data["oracle_epochs"],
            data["consensus_error"].clip(lower=1e-16),
            color=color,
            linewidth=1.7,
            label=f"{relaxation:g}",
        )
    axes[0].set_title("DBCFW consensus error")
    axes[0].set_xlabel("oracle epochs")
    axes[0].set_ylabel("mean agent distance")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="lambda", frameon=True, fontsize=8)

    lambda2 = dbcfw["lambda2"].dropna()
    axes[1].hist(lambda2, bins=22, color="#4c78a8", edgecolor="white")
    axes[1].set_title("time-varying graph contraction")
    axes[1].set_xlabel("lambda2_t = ||W_t - J||_2")
    axes[1].set_ylabel("count")
    axes[1].grid(True, axis="y", alpha=0.25)

    graph_seed = config.graph_seed if config.graph_seed is not None else config.seed
    rng = np.random.default_rng(graph_seed)
    graph_config = GraphConfig(graph_name(config.graph), config.edge_prob, config.geometric_radius, graph_seed)
    graph = sample_graph(config.agents, graph_config, rng)
    pos = nx.spring_layout(graph, seed=graph_seed)
    nx.draw_networkx_edges(graph, pos, ax=axes[2], width=1.6, alpha=0.55, edge_color="#4c78a8")
    nx.draw_networkx_nodes(graph, pos, ax=axes[2], node_size=520, node_color="#f58518", edgecolors="white", linewidths=1.4)
    nx.draw_networkx_labels(graph, pos, ax=axes[2], font_size=9, font_color="white")
    axes[2].set_title(f"one sampled communication graph, N={config.agents}")
    axes[2].axis("off")

    fig.suptitle("Decentralized setup diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return Path(path)


def plot_transport_geometry(config: OTPaperConfig, path: str | Path) -> Path:
    run_config = _paper_run_config(config, config.convergence_relaxation)
    problem = make_semirelaxed_ot_problem(run_config)
    reference_plan = solve_balanced_ot_lp(problem)
    _, dfw_plan = run_dfw_ot(problem, run_config, reference_plan)
    _, dbcfw_plan = run_dbcfw_ot(problem, run_config, reference_plan)
    plans = [("balanced LP", reference_plan), ("DFW", dfw_plan), ("DBCFW", dbcfw_plan)]
    path = Path(path)
    np.savez(path.parent / "gallery_transport_plans.npz", reference=reference_plan, dfw=dfw_plan, dbcfw=dbcfw_plan)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True, sharey=True)
    for ax, (label, plan) in zip(axes, plans):
        _draw_transport_map(ax, problem, plan, label)
    fig.suptitle(f"Transport geometry at lambda={config.convergence_relaxation:g}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def _draw_transport_map(ax, problem, plan: np.ndarray, title: str) -> None:
    threshold = np.quantile(plan[plan > 0.0], 0.74) if np.any(plan > 0.0) else 0.0
    max_mass = max(float(plan.max()), 1e-16)
    for row in range(problem.m):
        for col in range(problem.n):
            mass = float(plan[row, col])
            if mass < threshold:
                continue
            x = [problem.source_points[row, 0], problem.target_points[col, 0]]
            y = [problem.source_points[row, 1], problem.target_points[col, 1]]
            ax.plot(x, y, color="#6f4e7c", alpha=0.22 + 0.45 * mass / max_mass, linewidth=0.5 + 4.0 * mass / max_mass)
    ax.scatter(
        problem.source_points[:, 0],
        problem.source_points[:, 1],
        s=900 * problem.source_weights,
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.7,
        label="source",
        zorder=3,
    )
    ax.scatter(
        problem.target_points[:, 0],
        problem.target_points[:, 1],
        s=900 * problem.target_weights,
        color="#f58518",
        marker="s",
        edgecolor="white",
        linewidth=0.7,
        label="target",
        zorder=4,
    )
    ax.set_title(f"{title}\ngap={problem.duality_gap(plan):.2e}, obj={problem.objective(plan):.3g}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.23)
    ax.legend(frameon=True, loc="lower right", fontsize=8)


def _load_paper_config(path: Path) -> OTPaperConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return OTPaperConfig(
        m=int(data.get("m", 28)),
        n=int(data.get("n", 28)),
        agents=int(data.get("agents", 8)),
        epochs=int(data.get("epochs", 80)),
        batch=int(data.get("batch", 1)),
        relaxations=tuple(float(value) for value in data.get("relaxations", [0.02, 0.04, 0.08, 0.16, 0.32])),
        convergence_relaxation=float(data.get("convergence_relaxation", 0.08)),
        transition_relaxations=tuple(float(value) for value in data.get("transition_relaxations", [0.02, 0.32])),
        cost_noise=float(data.get("cost_noise", 0.03)),
        stepsize=str(data.get("stepsize", "line_search")),
        graph=str(data.get("graph", "erdos")),
        edge_prob=float(data.get("edge_prob", 0.45)),
        geometric_radius=float(data.get("geometric_radius", 0.55)),
        seed=int(data.get("seed", 42)),
        graph_seed=data.get("graph_seed"),
        log_every=int(data.get("log_every", 5)),
    )


def _set_style() -> None:
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "figure.facecolor": "white",
        "font.size": 10,
        "legend.framealpha": 0.92,
    })
