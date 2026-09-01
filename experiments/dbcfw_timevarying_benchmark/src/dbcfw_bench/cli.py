from __future__ import annotations

import argparse
from pathlib import Path

from dbcfw_bench.config import RunConfig
from dbcfw_bench.flow_figures import FigureRunConfig, generate_flow_matching_figures
from dbcfw_bench.ot_color_transfer import ColorTransferConfig, run_color_transfer_experiment
from dbcfw_bench.ot_gallery import build_ot_gallery
from dbcfw_bench.ot_experiment import (
    OTPaperConfig,
    OTRunConfig,
    plot_ot_results,
    run_ot_experiment,
    run_ot_paper_suite,
)
from dbcfw_bench.ot_routes import build_ot_route_report
from dbcfw_bench.ot_showcase import OTShowcaseConfig, build_ot_showcase
from dbcfw_bench.plotting.plots import plot_results
from dbcfw_bench.runners.grid import run_grid
from dbcfw_bench.runners.single_run import run_single
from dbcfw_bench.runners.summary import render_benchmark_log, update_readme_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dbcfw_bench")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)
    _add_grid(sub)
    _add_plot(sub)
    _add_ot(sub)
    _add_ot_paper(sub)
    _add_ot_color_transfer(sub)
    _add_ot_gallery(sub)
    _add_ot_plot(sub)
    _add_ot_routes(sub)
    _add_ot_showcase(sub)
    _add_flow_figures(sub)
    _add_summarize(sub)
    args = parser.parse_args(argv)
    if args.command == "run":
        frame = run_single(_run_config(args), args.out)
        print(f"wrote {Path(args.out) / 'results.csv'} with {len(frame)} rows")
    elif args.command == "grid":
        frame = run_grid(args.config, args.out)
        print(f"wrote {Path(args.out) / 'results.csv'} with {len(frame)} rows")
    elif args.command == "plot":
        paths = plot_results(args.results, args.out)
        print("wrote " + ", ".join(str(path) for path in paths))
    elif args.command == "ot":
        frame, paths = run_ot_experiment(_ot_config(args), args.out)
        print(f"wrote {Path(args.out) / 'results.csv'} with {len(frame)} rows")
        print("wrote " + ", ".join(str(path) for path in paths))
    elif args.command == "ot-paper":
        frame, paths = run_ot_paper_suite(_ot_paper_config(args), args.out)
        print(f"wrote {Path(args.out) / 'paper_suite_results.csv'} with {len(frame)} rows")
        print("wrote " + ", ".join(str(path) for path in paths))
    elif args.command == "ot-color-transfer":
        frame, paths = run_color_transfer_experiment(_color_transfer_config(args), args.out)
        print(f"wrote {Path(args.out) / 'color_transfer_results.csv'} with {len(frame)} rows")
        print("wrote " + ", ".join(str(path) for path in paths))
    elif args.command == "ot-gallery":
        paths = build_ot_gallery(args.run_dir, args.out)
        print("wrote " + ", ".join(str(path) for path in paths))
    elif args.command == "ot-plot":
        paths = plot_ot_results(args.results, args.out)
        print("wrote " + ", ".join(str(path) for path in paths))
    elif args.command == "ot-routes":
        paths = build_ot_route_report(args.run_dir, args.out)
        print("wrote " + ", ".join(str(path) for path in paths))
    elif args.command == "ot-showcase":
        paths = build_ot_showcase(args.out, _ot_showcase_config(args))
        print("wrote " + ", ".join(str(path) for path in paths))
    elif args.command == "flow-figures":
        paths = generate_flow_matching_figures(args.out, _figure_config(args))
        print("wrote " + ", ".join(str(path) for path in paths))
    elif args.command == "summarize":
        table = render_benchmark_log(args.runs)
        if args.readme:
            update_readme_summary(args.readme, table)
            print(f"updated {args.readme}")
        else:
            print(table)


def _add_run(sub) -> None:
    run = sub.add_parser("run")
    run.add_argument("--objective", default="quadratic")
    run.add_argument("--method", choices=["fw", "bcfw", "dfw", "dbcfw"], required=True)
    run.add_argument("--agents", type=int, required=True)
    run.add_argument("--dim", type=int, required=True)
    run.add_argument("--blocks", type=int, required=True)
    run.add_argument("--batch", type=int, default=1)
    run.add_argument("--iters", type=int, required=True)
    run.add_argument("--graph", default="erdos")
    run.add_argument("--edge-prob", type=float, default=0.25)
    run.add_argument("--geo-radius", type=float, default=0.4)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--graph-seed", type=int, default=None)
    run.add_argument("--samples-per-agent", type=int, default=20)
    run.add_argument("--reg", type=float, default=1e-3)
    run.add_argument("--box-radius", type=float, default=1.0)
    run.add_argument("--lmo", choices=["box", "l1_block", "l2_block", "simplex"], default="box")
    run.add_argument("--log-every", type=int, default=1)
    run.add_argument("--opt-maxiter", type=int, default=300)
    run.add_argument("--wall-time-budget-sec", type=float, default=None)
    run.add_argument("--data-dir", default=None)
    run.add_argument("--hidden-dim", type=int, default=16)
    run.add_argument("--gamma-offset", type=float, default=2.0)
    run.add_argument("--sequence-length", type=int, default=8)
    run.add_argument("--label-count", type=int, default=3)
    run.add_argument("--out", required=True)


def _add_grid(sub) -> None:
    grid = sub.add_parser("grid")
    grid.add_argument("--config", required=True)
    grid.add_argument("--out", required=True)


def _add_plot(sub) -> None:
    plot = sub.add_parser("plot")
    plot.add_argument("--results", required=True)
    plot.add_argument("--out", required=True)


def _add_ot(sub) -> None:
    ot = sub.add_parser("ot")
    ot.add_argument("--methods", nargs="+", choices=["fw", "bcfw", "dfw", "dbcfw"], default=["dfw", "dbcfw"])
    ot.add_argument("--m", type=int, default=28)
    ot.add_argument("--n", type=int, default=28)
    ot.add_argument("--agents", type=int, default=8)
    ot.add_argument("--epochs", type=int, default=80)
    ot.add_argument("--batch", type=int, default=1)
    ot.add_argument("--relaxation", type=float, default=0.08)
    ot.add_argument("--cost-noise", type=float, default=0.03)
    ot.add_argument("--stepsize", choices=["line_search", "decay"], default="line_search")
    ot.add_argument("--graph", default="erdos")
    ot.add_argument("--edge-prob", type=float, default=0.45)
    ot.add_argument("--geo-radius", type=float, default=0.55)
    ot.add_argument("--seed", type=int, default=42)
    ot.add_argument("--graph-seed", type=int, default=None)
    ot.add_argument("--log-every", type=int, default=5)
    ot.add_argument("--out", required=True)


def _add_ot_paper(sub) -> None:
    paper = sub.add_parser("ot-paper")
    paper.add_argument("--m", type=int, default=28)
    paper.add_argument("--n", type=int, default=28)
    paper.add_argument("--agents", type=int, default=8)
    paper.add_argument("--epochs", type=int, default=80)
    paper.add_argument("--batch", type=int, default=1)
    paper.add_argument("--relaxations", nargs="+", type=float, default=[0.02, 0.04, 0.08, 0.16, 0.32])
    paper.add_argument("--convergence-relaxation", type=float, default=0.08)
    paper.add_argument("--transition-relaxations", nargs=2, type=float, default=[0.02, 0.32])
    paper.add_argument("--cost-noise", type=float, default=0.03)
    paper.add_argument("--stepsize", choices=["line_search", "decay"], default="line_search")
    paper.add_argument("--graph", default="erdos")
    paper.add_argument("--edge-prob", type=float, default=0.45)
    paper.add_argument("--geo-radius", type=float, default=0.55)
    paper.add_argument("--seed", type=int, default=42)
    paper.add_argument("--graph-seed", type=int, default=None)
    paper.add_argument("--log-every", type=int, default=5)
    paper.add_argument("--out", required=True)


def _add_ot_color_transfer(sub) -> None:
    color = sub.add_parser("ot-color-transfer")
    color.add_argument("--source", default="rocket")
    color.add_argument("--target", default="coffee")
    color.add_argument("--colors", type=int, default=32)
    color.add_argument("--agents", type=int, default=8)
    color.add_argument("--epochs", type=int, default=80)
    color.add_argument("--batch", type=int, default=1)
    color.add_argument("--relaxation", type=float, default=0.08)
    color.add_argument("--cost-noise", type=float, default=0.0)
    color.add_argument("--stepsize", choices=["line_search", "decay"], default="line_search")
    color.add_argument("--graph", default="erdos")
    color.add_argument("--edge-prob", type=float, default=0.45)
    color.add_argument("--geo-radius", type=float, default=0.55)
    color.add_argument("--seed", type=int, default=202)
    color.add_argument("--graph-seed", type=int, default=1202)
    color.add_argument("--log-every", type=int, default=5)
    color.add_argument("--image-size", type=int, default=240)
    color.add_argument("--sample-pixels", type=int, default=25000)
    color.add_argument("--reference-epochs", type=int, default=1000)
    color.add_argument("--out", required=True)


def _add_ot_gallery(sub) -> None:
    gallery = sub.add_parser("ot-gallery")
    gallery.add_argument("--run-dir", required=True)
    gallery.add_argument("--out", default=None)


def _add_ot_plot(sub) -> None:
    plot = sub.add_parser("ot-plot")
    plot.add_argument("--results", required=True)
    plot.add_argument("--out", required=True)


def _add_ot_routes(sub) -> None:
    routes = sub.add_parser("ot-routes")
    routes.add_argument("--run-dir", required=True)
    routes.add_argument("--out", default=None)


def _add_ot_showcase(sub) -> None:
    showcase = sub.add_parser("ot-showcase")
    showcase.add_argument("--out", required=True)
    showcase.add_argument("--seed", type=int, default=2026)
    showcase.add_argument("--agents", type=int, default=6)
    showcase.add_argument("--epochs", type=int, default=70)
    showcase.add_argument("--batch", type=int, default=4)
    showcase.add_argument("--relaxation", type=float, default=0.055)
    showcase.add_argument("--cost-noise", type=float, default=0.025)
    showcase.add_argument("--bunny-points", type=int, default=28)
    showcase.add_argument("--maze-points", type=int, default=24)
    showcase.add_argument("--maze-width", type=int, default=35)
    showcase.add_argument("--maze-height", type=int, default=23)


def _add_flow_figures(sub) -> None:
    figures = sub.add_parser("flow-figures")
    figures.add_argument("--out", required=True)
    figures.add_argument("--resolution", type=int, default=76)
    figures.add_argument("--eigen-count", type=int, default=24)
    figures.add_argument("--sample-count", type=int, default=360)
    figures.add_argument("--seed", type=int, default=2302)


def _add_summarize(sub) -> None:
    summary = sub.add_parser("summarize")
    summary.add_argument("--runs", nargs="+", default=["runs"])
    summary.add_argument("--readme", default=None)


def _run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        objective=args.objective, method=args.method, agents=args.agents, dim=args.dim, blocks=args.blocks,
        batch=args.batch, iters=args.iters, samples_per_agent=args.samples_per_agent,
        reg=args.reg, box_radius=args.box_radius, lmo=args.lmo, graph=args.graph,
        edge_prob=args.edge_prob, geometric_radius=args.geo_radius, seed=args.seed,
        graph_seed=args.graph_seed, log_every=args.log_every,
        opt_maxiter=args.opt_maxiter, wall_time_budget_sec=args.wall_time_budget_sec,
        data_dir=args.data_dir, hidden_dim=args.hidden_dim, gamma_offset=args.gamma_offset,
        sequence_length=args.sequence_length, label_count=args.label_count,
    )


def _ot_config(args: argparse.Namespace) -> OTRunConfig:
    return OTRunConfig(
        methods=tuple(args.methods),
        m=args.m,
        n=args.n,
        agents=args.agents,
        epochs=args.epochs,
        batch=args.batch,
        relaxation=args.relaxation,
        cost_noise=args.cost_noise,
        stepsize=args.stepsize,
        graph=args.graph,
        edge_prob=args.edge_prob,
        geometric_radius=args.geo_radius,
        seed=args.seed,
        graph_seed=args.graph_seed,
        log_every=args.log_every,
    )


def _ot_paper_config(args: argparse.Namespace) -> OTPaperConfig:
    return OTPaperConfig(
        m=args.m,
        n=args.n,
        agents=args.agents,
        epochs=args.epochs,
        batch=args.batch,
        relaxations=tuple(args.relaxations),
        convergence_relaxation=args.convergence_relaxation,
        transition_relaxations=tuple(args.transition_relaxations),
        cost_noise=args.cost_noise,
        stepsize=args.stepsize,
        graph=args.graph,
        edge_prob=args.edge_prob,
        geometric_radius=args.geo_radius,
        seed=args.seed,
        graph_seed=args.graph_seed,
        log_every=args.log_every,
    )


def _ot_showcase_config(args: argparse.Namespace) -> OTShowcaseConfig:
    return OTShowcaseConfig(
        seed=args.seed,
        agents=args.agents,
        epochs=args.epochs,
        batch=args.batch,
        relaxation=args.relaxation,
        cost_noise=args.cost_noise,
        bunny_points=args.bunny_points,
        maze_points=args.maze_points,
        maze_width=args.maze_width,
        maze_height=args.maze_height,
    )


def _color_transfer_config(args: argparse.Namespace) -> ColorTransferConfig:
    return ColorTransferConfig(
        source=args.source,
        target=args.target,
        colors=args.colors,
        agents=args.agents,
        epochs=args.epochs,
        batch=args.batch,
        relaxation=args.relaxation,
        cost_noise=args.cost_noise,
        stepsize=args.stepsize,
        graph=args.graph,
        edge_prob=args.edge_prob,
        geometric_radius=args.geo_radius,
        seed=args.seed,
        graph_seed=args.graph_seed,
        log_every=args.log_every,
        image_size=args.image_size,
        sample_pixels=args.sample_pixels,
        reference_epochs=args.reference_epochs,
    )


def _figure_config(args: argparse.Namespace) -> FigureRunConfig:
    return FigureRunConfig(
        resolution=args.resolution,
        eigen_count=args.eigen_count,
        sample_count=args.sample_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
