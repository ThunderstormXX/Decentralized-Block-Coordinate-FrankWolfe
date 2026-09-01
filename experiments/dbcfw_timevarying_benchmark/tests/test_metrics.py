from __future__ import annotations

import numpy as np

from dbcfw_bench.metrics import boundary_activity


def test_box_boundary_activity_counts_coordinates() -> None:
    points = np.array([[1.0, 0.5], [-0.99, 0.0]])
    assert boundary_activity(points, 1.0, "box", 2) == 0.5


def test_l2_boundary_activity_counts_blocks() -> None:
    points = np.array([[3.0, 4.0, 0.1, 0.1]])
    assert boundary_activity(points, 5.0, "l2_block", 2) == 0.5
