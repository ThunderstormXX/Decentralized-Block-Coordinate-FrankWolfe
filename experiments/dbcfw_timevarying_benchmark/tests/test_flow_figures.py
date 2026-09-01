from __future__ import annotations

from dbcfw_bench.flow_figures import FigureRunConfig, generate_flow_matching_figures


def test_flow_figures_smoke(tmp_path) -> None:
    paths = generate_flow_matching_figures(
        tmp_path,
        FigureRunConfig(resolution=30, eigen_count=8, sample_count=40, seed=5),
    )
    assert len(paths) == 4
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
