from __future__ import annotations

import pandas as pd


def value(row, metric: str) -> str:
    if row is None:
        return "NA"
    return fmt(float(row[metric]))


def ratio(row) -> str:
    if row is None:
        return "NA"
    return pct(100.0 * float(row.batch) / float(row.blocks))


def step(row) -> str:
    if row is None:
        return "NA"
    return fmt(float(row.last_step_sec))


def budget_s(frame: pd.DataFrame) -> str:
    value = frame.wall_time_budget_sec.dropna()
    return "-" if value.empty else fmt(float(value.iloc[0]))


def uniq(frame: pd.DataFrame, column: str) -> str:
    values = sorted(frame[column].dropna().unique().tolist())
    return str(values[0]) if len(values) == 1 else "{" + ",".join(map(str, values)) + "}"


def fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3g}"


def pct(value: float) -> str:
    return f"{value:.0f}%" if abs(value - round(value)) < 1e-9 else f"{value:.1f}%"
