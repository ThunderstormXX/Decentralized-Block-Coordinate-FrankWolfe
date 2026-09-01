from __future__ import annotations

from pathlib import Path

import pandas as pd

from dbcfw_bench.runners.summary_format import budget_s, ratio, step, uniq, value
from dbcfw_bench.runners.summary_counts import render_counts
from dbcfw_bench.runners.summary_paths import display_name, grouped_sort_key, result_files
from dbcfw_bench.runners.summary_sort import best, quality_metric

HEADER = (
    "| Run | Task | LMO | N | d | n | budget_s | metric | DFW | DBCFW | "
    "B/n | better | DFW step_s | DBCFW step_s |"
)
SEP = "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|"


def render_benchmark_log(runs_dir: str | Path | list[str | Path] = "runs") -> str:
    entries = [_summarize_csv(path) for path in result_files(runs_dir)]
    packed = sorted([entry for entry in entries if entry], key=lambda item: item[1])
    rows = [row for row, _, _ in packed]
    return "\n".join([render_counts([winner for _, _, winner in packed]), "", HEADER, SEP, *rows])


def update_readme_summary(readme: str | Path, table: str) -> None:
    path = Path(readme)
    text = path.read_text(encoding="utf-8")
    start = text.index("## Benchmark Log")
    end = text.index("## Generated Outputs")
    body = "## Benchmark Log\n\n" + table + "\n\n"
    path.write_text(text[:start] + body + text[end:], encoding="utf-8")

def _summarize_csv(path: Path) -> tuple[str, tuple[str, int, str], str] | None:
    frame = _normalize(pd.read_csv(path))
    if frame.empty:
        return None
    final = _final_rows(frame)
    metric, lower, label = quality_metric(final)
    dfw = best(final, "dfw", metric, lower)
    dbcfw = best(final, "dbcfw", metric, lower)
    winner = _winner(dfw, dbcfw, metric, lower)
    lmo = uniq(final, "lmo")
    name = display_name(path.parent.name, lmo)
    cells = [
        name, uniq(final, "objective"), lmo, uniq(final, "agents"),
        uniq(final, "dim"), uniq(final, "blocks"), budget_s(final), label,
        value(dfw, metric), value(dbcfw, metric), ratio(dbcfw),
        winner, step(dfw), step(dbcfw),
    ]
    return "| " + " | ".join(cells) + " |", grouped_sort_key(path.parent.name, lmo), winner


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if "objective" not in frame:
        frame["objective"] = "quadratic"
    if "wall_time_budget_sec" not in frame:
        frame["wall_time_budget_sec"] = float("nan")
    if "mean_agent_accuracy" not in frame:
        frame["mean_agent_accuracy"] = float("nan")
    if "lmo" not in frame:
        frame["lmo"] = "box"
    return frame


def _final_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in frame.sort_values(["run_id", "iteration"]).groupby("run_id"):
        last = group.iloc[-1].copy()
        if len(group) > 1:
            prev = group.iloc[-2]
            steps = max(int(last.iteration - prev.iteration), 1)
            last["last_step_sec"] = (last.wall_time_sec - prev.wall_time_sec) / steps
        else:
            last["last_step_sec"] = float("nan")
        rows.append(last)
    return pd.DataFrame(rows)


def _winner(dfw, dbcfw, metric: str, lower: bool) -> str:
    if dfw is None:
        return "DBCFW"
    if dbcfw is None:
        return "DFW"
    left, right = float(dfw[metric]), float(dbcfw[metric])
    if abs(left - right) <= 1e-12:
        return "tie"
    if lower:
        return "DFW" if left < right else "DBCFW"
    return "DFW" if left > right else "DBCFW"
