from __future__ import annotations

import numpy as np

from dbcfw_bench.config import RunConfig
from dbcfw_bench.data import make_problem


def test_synthetic_nlp_problem_is_reproducible() -> None:
    cfg = RunConfig(
        objective="synthetic_topic_logreg",
        agents=3,
        dim=64,
        blocks=8,
        samples_per_agent=12,
        seed=7,
    )
    left = make_problem(cfg)
    right = make_problem(cfg)
    assert left.dim == 64
    assert len(left.x_parts) == 3
    assert np.allclose(left.x_parts[0], right.x_parts[0])


def test_synthetic_sentiment_has_accuracy() -> None:
    cfg = RunConfig(
        objective="synthetic_sentiment_logreg",
        agents=2,
        dim=80,
        blocks=8,
        samples_per_agent=10,
        seed=9,
    )
    problem = make_problem(cfg)
    points = np.zeros((cfg.agents, cfg.dim))
    acc = problem.mean_agent_accuracy(points)
    assert 0.0 <= acc <= 1.0
