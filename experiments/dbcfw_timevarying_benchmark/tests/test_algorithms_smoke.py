from __future__ import annotations

import numpy as np
import pandas as pd

from dbcfw_bench.config import RunConfig
from dbcfw_bench.runners.single_run import run_single


def test_dbcfw_single_run_smoke(tmp_path) -> None:
    cfg = RunConfig(
        method="dbcfw", agents=4, dim=12, blocks=3, batch=1, iters=3,
        samples_per_agent=5, edge_prob=0.7, seed=3, graph_seed=9,
        log_every=1, opt_maxiter=20,
    )
    frame = run_single(cfg, tmp_path)
    assert (tmp_path / "results.csv").exists()
    assert len(frame) == 4
    assert np.isfinite(frame["objective_gap"]).all()
    assert frame["total_oracle_coordinates"].iloc[-1] == 4 * 4 * 3


def test_dfw_uses_all_coordinates(tmp_path) -> None:
    cfg = RunConfig(
        method="dfw", agents=3, dim=12, blocks=3, batch=1, iters=2,
        samples_per_agent=5, edge_prob=0.8, seed=5, log_every=1,
        opt_maxiter=20,
    )
    frame = run_single(cfg, tmp_path)
    saved = pd.read_csv(tmp_path / "results.csv")
    assert frame["oracle_coordinates_per_iter"].iloc[-1] == 36
    assert saved["batch"].iloc[-1] == 3


def test_central_fw_and_bcfw_smoke(tmp_path) -> None:
    fw_cfg = RunConfig(
        method="fw", agents=3, dim=12, blocks=3, batch=1, iters=2,
        samples_per_agent=5, edge_prob=0.8, seed=15, log_every=1,
        opt_maxiter=20,
    )
    bcfw_cfg = RunConfig(
        method="bcfw", agents=3, dim=12, blocks=3, batch=1, iters=2,
        samples_per_agent=5, edge_prob=0.8, seed=15, log_every=1,
        opt_maxiter=20,
    )
    fw = run_single(fw_cfg, tmp_path / "fw")
    bcfw = run_single(bcfw_cfg, tmp_path / "bcfw")
    assert fw["graph"].iloc[-1] == "centralized"
    assert fw["oracle_coordinates_per_iter"].iloc[-1] == 12
    assert bcfw["oracle_coordinates_per_iter"].iloc[-1] == 4
    assert np.isfinite(bcfw["objective_gap"]).all()


def test_structural_svm_dbcfw_smoke(tmp_path) -> None:
    cfg = RunConfig(
        objective="structural_svm", method="dbcfw", agents=3, dim=30,
        blocks=4, batch=1, iters=2, reg=0.1, edge_prob=0.9,
        seed=11, graph_seed=12, log_every=1, opt_maxiter=0,
        sequence_length=5, label_count=3,
    )
    frame = run_single(cfg, tmp_path)
    assert len(frame) == 3
    assert (frame["objective_gap"] >= 0.0).all()
    assert frame["mean_agent_accuracy"].between(0.0, 1.0).all()
    assert frame["lmo"].iloc[-1] == "simplex"
    assert frame["total_oracle_coordinates"].iloc[-1] > 0


def test_structural_svm_dfw_uses_all_local_blocks(tmp_path) -> None:
    cfg = RunConfig(
        objective="structural_sequence_svm", method="dfw", agents=2, dim=30,
        blocks=5, batch=1, iters=1, reg=0.1, edge_prob=1.0,
        seed=13, log_every=1, opt_maxiter=0, sequence_length=4,
        label_count=3,
    )
    frame = run_single(cfg, tmp_path)
    assert frame["batch"].iloc[-1] == 5
    assert frame["oracle_coordinates_per_iter"].iloc[-1] >= 10
