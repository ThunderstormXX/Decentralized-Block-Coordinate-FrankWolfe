from __future__ import annotations

import pandas as pd

from dbcfw_bench.runners.summary import render_benchmark_log
from dbcfw_bench.runners.summary_sort import quality_metric


def test_summary_reads_old_and_new_logs(tmp_path):
    old = tmp_path / "old_run"
    new = tmp_path / "new_run"
    artifact = tmp_path / "grid"
    old.mkdir()
    new.mkdir()
    artifact.mkdir()
    _write(old / "results.csv", objective=None, acc=False)
    _write(new / "results.csv", objective="mnist_multiclass_logreg", acc=True)
    _write(artifact / "results.csv", objective=None, acc=False)
    table = render_benchmark_log(tmp_path)
    assert "old_run | quadratic | box" in table
    assert "new_run | mnist_multiclass_logreg" in table
    assert "Summary: DBCFW better: 2/2; DFW better: 0/2; ties: 0/2." in table
    assert "grid | quadratic" not in table
    assert "accuracy | 0.7 | 0.8 | 50% | DBCFW" in table
    assert "consensus | 0.3 | 0.2 | 50% | DBCFW" in table


def test_summary_groups_lmo_prefixed_runs(tmp_path):
    root = tmp_path / "runs_lmo"
    run = root / "l1_block_old_run"
    run.mkdir(parents=True)
    _write(run / "results.csv", objective=None, acc=False, lmo="l1_block")
    table = render_benchmark_log([tmp_path, root])
    assert "old_run | quadratic | l1_block" in table
    assert "l1_block_old_run |" not in table


def test_accuracy_tie_still_uses_accuracy_metric() -> None:
    frame = pd.DataFrame([
        {"method": "dfw", "mean_agent_accuracy": 0.7, "consensus_error": 0.4},
        {"method": "dbcfw", "mean_agent_accuracy": 0.7, "consensus_error": 0.2},
    ])
    assert quality_metric(frame) == ("mean_agent_accuracy", False, "accuracy")


def _write(path, objective: str | None, acc: bool, lmo: str | None = None) -> None:
    rows = []
    for method, batch, gap, cons, accuracy in [
        ("dfw", 10, 1.0, 0.3, 0.7),
        ("dbcfw", 5, 0.5, 0.2, 0.8),
    ]:
        for iteration, wall in [(0, 0.0), (2, 0.4)]:
            row = dict(
                run_id=f"{method}_{path.parent.name}",
                method=method, agents=2, dim=20, blocks=10, batch=batch,
                iteration=iteration, wall_time_sec=wall, objective_gap=gap,
                consensus_error=cons,
            )
            if objective is not None:
                row["objective"] = objective
                row["wall_time_budget_sec"] = 3.0
            if acc:
                row["mean_agent_accuracy"] = accuracy
            if lmo:
                row["lmo"] = lmo
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
