from __future__ import annotations

import numpy as np

from dbcfw_bench.config import RunConfig
from dbcfw_bench.data import make_problem
from dbcfw_bench.runners.single_run import run_single


def test_flow_matching_problem_gradient_matches_finite_difference() -> None:
    cfg = RunConfig(
        objective="euclidean_flow_matching",
        agents=3,
        dim=15,
        hidden_dim=3,
        blocks=5,
        samples_per_agent=8,
        seed=11,
        reg=1e-3,
    )
    problem = make_problem(cfg)
    rng = np.random.default_rng(13)
    x = rng.normal(0.0, 0.1, problem.dim)
    direction = rng.normal(0.0, 1.0, problem.dim)
    direction /= np.linalg.norm(direction)
    eps = 1e-6

    finite_diff = (
        problem.objective(x + eps * direction)
        - problem.objective(x - eps * direction)
    ) / (2 * eps)
    analytic = float(problem.grad(x) @ direction)

    assert np.isclose(finite_diff, analytic, rtol=1e-5, atol=1e-7)


def test_flow_matching_dbcfw_smoke(tmp_path) -> None:
    cfg = RunConfig(
        objective="euclidean_flow_matching",
        method="dbcfw",
        agents=4,
        dim=24,
        hidden_dim=4,
        blocks=6,
        batch=2,
        iters=4,
        samples_per_agent=12,
        edge_prob=0.8,
        seed=23,
        graph_seed=60,
        log_every=1,
        opt_maxiter=40,
    )

    frame = run_single(cfg, tmp_path)

    assert (tmp_path / "results.csv").exists()
    assert len(frame) == 5
    assert np.isfinite(frame["objective_gap"]).all()
    assert frame["objective"].iloc[-1] == "euclidean_flow_matching"
