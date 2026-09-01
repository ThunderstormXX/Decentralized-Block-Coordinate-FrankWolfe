from __future__ import annotations

import numpy as np

from dbcfw_bench.ot_color_transfer import ColorTransferConfig, run_color_transfer_experiment


def test_color_transfer_smoke(tmp_path) -> None:
    config = ColorTransferConfig(
        source="rocket",
        target="coffee",
        colors=4,
        agents=2,
        epochs=1,
        batch=1,
        image_size=48,
        sample_pixels=400,
        reference_epochs=3,
        edge_prob=1.0,
        seed=17,
        graph_seed=18,
        log_every=1,
    )
    frame, paths = run_color_transfer_experiment(config, tmp_path)
    assert (tmp_path / "color_transfer_results.csv").exists()
    assert (tmp_path / "semi_relaxed_reference_results.csv").exists()
    assert {"dfw", "dbcfw"} == set(frame["method"].unique())
    assert np.isfinite(frame["duality_gap"]).all()
    assert len(paths) == 11
    assert all(path.exists() for path in paths)
