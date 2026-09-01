from __future__ import annotations

import numpy as np

from dbcfw_bench.lmo import block_box_lmo, block_lmo, block_slices, full_box_lmo


def test_full_box_lmo_uses_negative_sign() -> None:
    grad = np.array([-2.0, 0.0, 3.0])
    out = full_box_lmo(grad, radius=4.0)
    np.testing.assert_allclose(out, np.array([4.0, 0.0, -4.0]))


def test_block_box_lmo_changes_only_selected_blocks() -> None:
    grad = np.array([1.0, -1.0, 2.0, -2.0])
    base = np.array([0.1, 0.2, 0.3, 0.4])
    out = block_box_lmo(grad, base, 1.0, block_slices(4, 2), np.array([1]))
    np.testing.assert_allclose(out, np.array([0.1, 0.2, -1.0, 1.0]))


def test_l1_block_lmo_selects_sparse_vertex() -> None:
    out = block_lmo(np.array([1.0, -3.0, 2.0]), 2.0, "l1_block")
    np.testing.assert_allclose(out, np.array([0.0, 2.0, 0.0]))


def test_l2_block_lmo_normalizes_gradient() -> None:
    out = block_lmo(np.array([3.0, 4.0]), 10.0, "l2_block")
    np.testing.assert_allclose(out, np.array([-6.0, -8.0]))
