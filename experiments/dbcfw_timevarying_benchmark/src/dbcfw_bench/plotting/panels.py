from __future__ import annotations

import matplotlib.axes
import pandas as pd


def add_panel(ax: matplotlib.axes.Axes, frame: pd.DataFrame) -> None:
    ax.text(
        1.03,
        0.98,
        _panel_text(frame),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        linespacing=1.18,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f7f7f7", "edgecolor": "#555"},
    )


def _panel_text(frame: pd.DataFrame) -> str:
    return "\n".join([
        "FAIR SETUP",
        _setup_line(frame),
        _seed_line(frame),
        "same objective seed",
        "same W_t for method pairs",
        _gamma_line(frame),
        "",
        "METHOD HYPERPARAMETERS",
        _fw_line(frame),
        _bcfw_line(frame),
        _dfw_line(frame),
        _dbcfw_line(frame),
    ])


def _setup_line(frame: pd.DataFrame) -> str:
    bits = []
    for col, label in [("agents", "N"), ("dim", "d"), ("blocks", "n")]:
        if col in frame.columns:
            vals = sorted(frame[col].dropna().unique())
            value = vals[0] if len(vals) == 1 else "{" + ",".join(map(str, vals)) + "}"
            bits.append(f"{label}={value}")
    return ", ".join(bits)


def _seed_line(frame: pd.DataFrame) -> str:
    seed = _values(frame, "seed")
    graph_seed = _values(frame, "graph_seed")
    return f"seed={seed}, graph_seed={graph_seed}"


def _fw_line(frame: pd.DataFrame) -> str:
    if "method" not in frame.columns or not frame["method"].eq("fw").any():
        return "FW:    not run"
    blocks = _values(frame[frame["method"].eq("fw")], "blocks")
    return f"FW:    full LMO, B=n={blocks}, centralized"


def _bcfw_line(frame: pd.DataFrame) -> str:
    if "method" not in frame.columns or not frame["method"].eq("bcfw").any():
        return "BCFW:  not run"
    sub = frame[frame["method"].eq("bcfw")]
    return _batch_line(sub, "BCFW:  block LMO, centralized")


def _dfw_line(frame: pd.DataFrame) -> str:
    if "method" in frame.columns and not frame["method"].eq("dfw").any():
        return "DFW:   not run"
    blocks = _values(frame, "blocks")
    return f"DFW:   full LMO, B=n={blocks}, B/n=100%"


def _dbcfw_line(frame: pd.DataFrame) -> str:
    sub = frame[frame["method"].eq("dbcfw")] if "method" in frame.columns else frame
    return _batch_line(sub, "DBCFW: block LMO")


def _batch_line(frame: pd.DataFrame, prefix: str) -> str:
    if frame.empty:
        return f"{prefix}, not run"
    batches = [int(x) for x in sorted(frame["batch"].dropna().unique())] if "batch" in frame.columns else []
    blocks = [int(x) for x in sorted(frame["blocks"].dropna().unique())] if "blocks" in frame.columns else []
    if len(blocks) == 1 and batches:
        fracs = [f"{100 * b / blocks[0]:.0f}%" for b in batches]
        batch_text = ",".join(str(batch) for batch in batches)
        frac_text = ",".join(fracs)
        return f"{prefix}, B={{{batch_text}}}, B/n={{{frac_text}}}"
    return f"{prefix}, B/n as labeled"


def _gamma_line(frame: pd.DataFrame) -> str:
    if "gamma_offset" not in frame.columns:
        return "gamma_t = 2/(t+2)"
    vals = sorted(frame["gamma_offset"].dropna().unique())
    if len(vals) == 1:
        return f"gamma_t = 2/(t+{vals[0]:g})"
    return "gamma_t = 2/(t+offset)"


def _values(frame: pd.DataFrame, col: str) -> str:
    if col not in frame.columns:
        return "NA"
    vals = sorted(frame[col].dropna().unique())
    if len(vals) == 1:
        return str(vals[0])
    return "{" + ",".join(map(str, vals)) + "}"
