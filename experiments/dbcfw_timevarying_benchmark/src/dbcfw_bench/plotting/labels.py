from __future__ import annotations

import pandas as pd


def method_label(key: tuple) -> str:
    if len(key) == 5:
        method, batch, blocks, graph, lmo = key
    else:
        method, batch, blocks, graph = key
        lmo = ""
    frac = 100 * batch / blocks
    if method in {"dfw", "fw"}:
        oracle = "full LMO"
    elif batch >= blocks:
        oracle = "full/block LMO"
    else:
        oracle = "block LMO"
    graph_part = "" if graph in {"mixed", "centralized"} else f", graph={graph}"
    lmo_part = f", LMO={lmo}" if lmo else ""
    return f"{method.upper()}: {oracle}, B={batch}/n={blocks} ({frac:.0f}%){lmo_part}{graph_part}"


def setup_title(frame: pd.DataFrame, metric: str) -> str:
    parts = _common_parts(frame)
    setup = ", ".join(parts)
    return (
        f"{metric}\nCommon setup: {setup}; "
        "same objective seed; same graph seed for decentralized methods; "
        "gamma_t=2/(t+2)"
    )


def _common_parts(frame: pd.DataFrame) -> list[str]:
    names = {
        "agents": "N",
        "dim": "d",
        "blocks": "n",
        "seed": "seed",
        "graph_seed": "graph_seed",
        "graph": "graph",
    }
    parts: list[str] = []
    for col, label in names.items():
        if col not in frame.columns:
            continue
        values = frame[col].dropna().unique()
        if len(values) == 1:
            parts.append(f"{label}={values[0]}")
    varying = _varying(frame, ["blocks", "batch", "graph", "lmo"])
    if varying:
        parts.append("varying " + "/".join(varying))
    return parts


def _varying(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    labels = {"blocks": "n", "batch": "B", "graph": "graph", "lmo": "LMO"}
    out: list[str] = []
    for col in columns:
        if col in frame.columns and frame[col].nunique(dropna=True) > 1:
            out.append(labels[col])
    return out
