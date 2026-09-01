from __future__ import annotations

import numpy as np

from dbcfw_bench.ot_experiment import (
    OTPaperConfig,
    OTRunConfig,
    make_semirelaxed_ot_problem,
    run_bcfw,
    run_dbcfw_ot,
    run_dfw_ot,
    run_fw,
    run_ot_experiment,
    run_ot_paper_suite,
)
from dbcfw_bench.ot_routes import build_ot_route_report


def test_ot_lmo_preserves_column_marginals() -> None:
    config = OTRunConfig(m=6, n=5, agents=3, epochs=2, seed=7)
    problem = make_semirelaxed_ot_problem(config)
    atom = problem.lmo(problem.gradient(problem.initial_plan()))
    np.testing.assert_allclose(atom.sum(axis=0), problem.target_weights)
    assert np.all(atom >= 0.0)


def test_fw_and_bcfw_reduce_ot_gap() -> None:
    config = OTRunConfig(m=7, n=6, agents=3, epochs=5, batch=1, log_every=1, seed=8)
    problem = make_semirelaxed_ot_problem(config)
    fw_frame, _ = run_fw(problem, config)
    bcfw_frame, _ = run_bcfw(problem, config)
    assert fw_frame["duality_gap"].iloc[-1] < fw_frame["duality_gap"].iloc[0]
    assert bcfw_frame["duality_gap"].iloc[-1] < bcfw_frame["duality_gap"].iloc[0]


def test_dbcfw_ot_smoke() -> None:
    config = OTRunConfig(
        m=6, n=5, agents=4, epochs=3, batch=1, log_every=1,
        edge_prob=0.9, seed=9, graph_seed=11,
    )
    problem = make_semirelaxed_ot_problem(config)
    frame, plan = run_dbcfw_ot(problem, config)
    assert len(frame) == 16
    assert np.isfinite(frame["objective"]).all()
    assert frame["communication_rounds"].iloc[-1] == 15
    assert frame["total_oracle_time_sec"].iloc[-1] >= 0.0
    assert frame["total_communication_time_sec"].iloc[-1] >= 0.0
    np.testing.assert_allclose(plan.sum(axis=0), problem.target_weights, atol=1e-10)


def test_dfw_ot_smoke() -> None:
    config = OTRunConfig(
        m=6, n=5, agents=4, epochs=3, batch=1, log_every=1,
        edge_prob=0.9, seed=10, graph_seed=12,
    )
    problem = make_semirelaxed_ot_problem(config)
    frame, plan = run_dfw_ot(problem, config)
    assert len(frame) == 4
    assert frame["communication_rounds"].iloc[-1] == 3
    assert frame["oracle_columns_per_iter"].iloc[-1] == config.agents * config.n
    assert frame["total_oracle_time_sec"].iloc[-1] >= 0.0
    assert frame["total_communication_time_sec"].iloc[-1] >= 0.0
    np.testing.assert_allclose(plan.sum(axis=0), problem.target_weights, atol=1e-10)


def test_ot_paper_suite_smoke(tmp_path) -> None:
    config = OTPaperConfig(
        m=5, n=4, agents=3, epochs=2, batch=1,
        relaxations=(0.04, 0.08), convergence_relaxation=0.08,
        transition_relaxations=(0.04, 0.08), edge_prob=1.0,
        seed=13, graph_seed=14, log_every=1,
    )
    frame, paths = run_ot_paper_suite(config, tmp_path)
    assert (tmp_path / "paper_suite_results.csv").exists()
    assert {"dfw", "dbcfw"} == set(frame["method"].unique())
    assert len(paths) == 9
    assert all(path.exists() for path in paths)


def test_ot_route_report_smoke(tmp_path) -> None:
    config = OTRunConfig(
        methods=("dfw", "dbcfw"),
        m=4, n=4, agents=2, epochs=1, batch=2,
        edge_prob=1.0, seed=15, graph_seed=16, log_every=1,
    )
    run_ot_experiment(config, tmp_path / "run")
    paths = build_ot_route_report(tmp_path / "run", tmp_path / "report")
    assert (tmp_path / "report" / "route_report_summary.csv").exists()
    assert len(paths) == 7
    assert all(path.exists() for path in paths)
