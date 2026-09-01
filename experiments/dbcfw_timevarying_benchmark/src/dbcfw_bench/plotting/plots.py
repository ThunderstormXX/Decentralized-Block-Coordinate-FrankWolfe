from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dbcfw_bench.plotting.gamma import gamma_plot
from dbcfw_bench.plotting.labels import method_label, setup_title
from dbcfw_bench.plotting.multimetric import multi_metric_plot
from dbcfw_bench.plotting.panels import add_panel


def plot_results(results_csv: str | Path, out_dir: str | Path) -> list[Path]:
    frame = pd.read_csv(results_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        _line(frame, "wall_time_sec", "objective_gap", out / "gap_vs_time.png"),
        _line(frame, "iteration", "objective_gap", out / "gap_vs_iterations.png"),
        _scatter_last(frame, "peak_memory_bytes", "objective_gap", out / "memory_vs_gap.png"),
        _line(frame, "total_oracle_coordinates", "objective_gap", out / "oracle_work_vs_gap.png"),
        _line(frame, "wall_time_sec", "consensus_error", out / "consensus_error_vs_time.png"),
        multi_metric_plot(frame, out / "loss_accuracy_consensus_vs_time.png"),
        gamma_plot(frame, out / "gamma_vs_iterations.png"),
        _lambda_hist(frame, out / "lambda2_hist.png"),
    ]
    return paths


def _line(frame: pd.DataFrame, x_col: str, y_col: str, path: Path) -> Path:
    if y_col == "consensus_error" and "iteration" in frame.columns:
        frame = frame[frame["iteration"] > 0].copy()
    fig, ax = plt.subplots(figsize=(14, 7))
    group_cols = _group_cols(frame)
    for key, group in frame.groupby(group_cols, sort=False):
        data = group.sort_values(x_col)
        y = data[y_col].clip(lower=1e-16)
        ax.plot(data[x_col], y, marker="o", markersize=2, linewidth=1, label=method_label(key))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(setup_title(frame, y_col))
    if "gap" in y_col or "error" in y_col:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    add_panel(ax, frame)
    _legend(ax, 7)
    fig.subplots_adjust(left=0.08, right=0.58, top=0.84, bottom=0.12)
    fig.savefig(path, dpi=160)
    plt.close()
    return path


def _scatter_last(frame: pd.DataFrame, x_col: str, y_col: str, path: Path) -> Path:
    idx = frame.groupby("run_id")["iteration"].idxmax()
    last = frame.loc[idx].copy()
    if x_col.endswith("_bytes"):
        last[x_col] = last[x_col] / (1024 * 1024)
        xlabel = x_col.replace("_bytes", "_mb")
    else:
        xlabel = x_col
    fig, ax = plt.subplots(figsize=(13, 6.5))
    group_cols = _group_cols(last)
    for key, group in last.groupby(group_cols, sort=False):
        ax.scatter(group[x_col], group[y_col].clip(lower=1e-16), label=method_label(key))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_col)
    ax.set_title(setup_title(frame, y_col))
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    add_panel(ax, frame)
    _legend(ax, 7)
    fig.subplots_adjust(left=0.08, right=0.58, top=0.84, bottom=0.12)
    fig.savefig(path, dpi=160)
    plt.close()
    return path


def _lambda_hist(frame: pd.DataFrame, path: Path) -> Path:
    values = frame["lambda2"].dropna()
    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=25, color="#4c78a8", edgecolor="white")
    plt.xlabel("lambda2_t = ||W_t - J||_2")
    plt.ylabel("count")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _legend(ax, size: int) -> None:
    ax.legend(fontsize=size, frameon=True, loc="best")


def _group_cols(frame: pd.DataFrame) -> list[str]:
    cols = ["method", "batch", "blocks", "graph"]
    if "lmo" in frame.columns:
        cols.append("lmo")
    return cols
