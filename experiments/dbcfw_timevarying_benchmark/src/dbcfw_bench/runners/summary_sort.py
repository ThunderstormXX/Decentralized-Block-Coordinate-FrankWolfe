from __future__ import annotations

import pandas as pd


def quality_sort_key(name: str, frame: pd.DataFrame) -> tuple[int, float, str]:
    score = quality_score(frame)
    if score is None:
        return 3, 0.0, name
    if score > 0:
        return 0, -score, name
    if score == 0:
        return 1, 0.0, name
    return 2, score, name


def quality_score(frame: pd.DataFrame) -> float | None:
    metric, lower, _ = quality_metric(frame)
    return metric_score(frame, metric, lower)


def quality_metric(frame: pd.DataFrame) -> tuple[str, bool, str]:
    if "mean_agent_accuracy" in frame and frame["mean_agent_accuracy"].notna().any():
        return "mean_agent_accuracy", False, "accuracy"
    return "consensus_error", True, "consensus"


def metric_score(frame: pd.DataFrame, metric: str, lower: bool) -> float | None:
    dfw = best(frame, "dfw", metric, lower)
    dbcfw = best(frame, "dbcfw", metric, lower)
    if dfw is None or dbcfw is None:
        return None
    if lower:
        return float(dfw[metric] - dbcfw[metric])
    return float(dbcfw[metric] - dfw[metric])


def best(frame: pd.DataFrame, method: str, metric: str, lower: bool):
    part = frame[(frame.method == method) & frame[metric].notna()]
    if part.empty:
        return None
    idx = part[metric].idxmin() if lower else part[metric].idxmax()
    return part.loc[idx]
