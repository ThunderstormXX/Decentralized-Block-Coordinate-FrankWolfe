from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

def multi_metric_plot(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    frame = frame[frame["iteration"] > 0].copy()
    fig, axes = plt.subplots(3, 1, figsize=(17, 13), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    handles = _draw_lines(frame, axes)
    _format_axes(frame, axes)
    _side_text(fig, frame)
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
        fontsize=6.8, frameon=False, ncol=4, columnspacing=1.0, handlelength=2.0
    )
    fig.subplots_adjust(left=0.075, right=0.68, top=0.93, bottom=0.19, hspace=0.08)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

def _draw_lines(frame: pd.DataFrame, axes) -> list[Line2D]:
    batches = sorted(int(x) for x in frame["batch"].dropna().unique())
    blocks = int(frame["blocks"].iloc[0])
    max_batch = max(batches) if batches else 1
    colors = plt.cm.Blues(np.linspace(0.35, 0.95, max_batch))
    method_colors = {"fw": "#111111", "dfw": "#555555", "bcfw": "#1b9e77"}
    group_cols = ["method", "batch"] + (["lmo"] if "lmo" in frame.columns else [])
    handles: list[Line2D] = []
    for key, group in frame.groupby(group_cols, sort=True):
        method, batch, lmo = _group_key(key)
        data = group.sort_values("wall_time_sec")
        color = method_colors.get(method, colors[int(batch) - 1])
        width = 3.0 if method in {"fw", "dfw"} else 1.55
        style = "--" if lmo == "l2_block" else "-"
        axes[0].plot(data["wall_time_sec"], data["objective_gap"], color=color, lw=width, ls=style)
        axes[1].plot(data["wall_time_sec"], data["mean_agent_accuracy"], color=color, lw=width, ls=style)
        axes[2].plot(data["wall_time_sec"], data["consensus_error"].clip(1e-16), color=color, lw=width, ls=style)
        handles.append(Line2D([0], [0], color=color, lw=width, ls=style, label=_label(method, batch, blocks, lmo)))
    return handles


def _group_key(key) -> tuple[str, int, str]:
    if isinstance(key, tuple) and len(key) == 3:
        method, batch, lmo = key
        return str(method), int(batch), str(lmo)
    if isinstance(key, tuple):
        method, batch = key
        return str(method), int(batch), ""
    raise ValueError(f"unexpected group key: {key}")

def _format_axes(frame: pd.DataFrame, axes) -> None:
    linear = _linear_loss(frame)
    if linear:
        post_start = frame.loc[frame["wall_time_sec"] >= 1.0, "objective_gap"]
        top = max(3.0, float(post_start.quantile(0.98)) * 1.05)
        axes[0].set_ylim(0, min(top, 5.0))
        axes[0].set_ylabel("train loss, linear zoom")
    else:
        axes[0].set_yscale("log")
        axes[0].set_ylabel("objective_gap / train loss")
    axes[1].set_ylabel("mean agent accuracy")
    axes[2].set_ylabel("consensus error")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("wall time spent in iterations, sec")
    positive = frame["consensus_error"].clip(lower=1e-16)
    if len(positive):
        axes[2].set_ylim(float(positive.min()) * 0.8, float(positive.max()) * 1.2)
    budget = frame["wall_time_budget_sec"].dropna()
    if len(budget):
        axes[2].set_xlim(float(frame["wall_time_sec"].min()), float(budget.max()) * 1.01)
    for ax in axes:
        ax.grid(True, alpha=0.28)
    axes[1].set_ylim(
        max(0.0, float(frame["mean_agent_accuracy"].min()) - 0.04),
        min(1.0, float(frame["mean_agent_accuracy"].max()) + 0.04),
    )
    axes[0].set_title(f"{_objective(frame)}: loss, accuracy, and point consensus")

def _side_text(fig, frame: pd.DataFrame) -> None:
    blocks = int(frame["blocks"].iloc[0])
    dim = int(frame["dim"].iloc[0])
    gamma = float(frame.get("gamma_offset", pd.Series([2.0])).iloc[0])
    setup = "\n".join([
        "FAIR SETUP", f"objective: {_objective(frame)}", f"N agents: {int(frame['agents'].iloc[0])}",
        f"dim: {dim}", f"n blocks: {blocks}", f"wall-time budget: {_budget(frame)}s per curve",
        "same objective seed, graph seed, W_t sequence", "same initialization across methods",
    ])
    method = "\n".join([
        "METHOD HYPERS", f"FW/DFW: full LMO, oracle coords/iter = d = {dim}",
        "BCFW/DBCFW: block LMO only on B active blocks", "B/n grid as configured",
        f"gamma_t = 2/(t + {gamma:g}); gamma_0={2 / gamma:.3f}",
        "consensus_error = mean_i ||x_i - x_avg||",
    ])
    box = {"facecolor": "white", "edgecolor": "#dddddd", "pad": 8}
    fig.text(0.715, 0.76, setup, fontsize=8.6, va="top", ha="left", bbox=box)
    fig.text(0.715, 0.48, method, fontsize=8.6, va="top", ha="left", bbox=box)

def _objective(frame: pd.DataFrame) -> str:
    return str(frame["objective"].iloc[0]) if "objective" in frame.columns else "benchmark"

def _budget(frame: pd.DataFrame) -> str:
    values = frame["wall_time_budget_sec"].dropna()
    return f"{float(values.max()):.0f}" if len(values) else "NA"

def _linear_loss(frame: pd.DataFrame) -> bool:
    return frame.get("objective", pd.Series([""])).astype(str).str.contains("cnn").any()

def _label(method: str, batch: int, blocks: int, lmo: str = "") -> str:
    lmo_part = f" | {lmo}" if lmo else ""
    if method == "fw":
        return f"FW full LMO | B=n={blocks}{lmo_part}"
    if method == "dfw":
        return f"DFW full LMO | B=n={blocks}{lmo_part}"
    prefix = "BCFW" if method == "bcfw" else "DBCFW"
    return f"{prefix} | B={int(batch):02d}, n={blocks}, B/n={batch / blocks:.0%}{lmo_part}"
