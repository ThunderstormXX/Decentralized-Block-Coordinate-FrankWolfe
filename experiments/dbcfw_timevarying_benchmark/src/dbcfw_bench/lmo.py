from __future__ import annotations

import numpy as np


def block_slices(dim: int, n_blocks: int) -> list[slice]:
    if n_blocks < 1 or n_blocks > dim:
        raise ValueError("n_blocks must be in [1, dim]")
    edges = np.linspace(0, dim, n_blocks + 1, dtype=int)
    return [slice(int(edges[k]), int(edges[k + 1])) for k in range(n_blocks)]


def full_box_lmo(gradient: np.ndarray, radius: float) -> np.ndarray:
    return -radius * np.sign(gradient)


def lmo_name(name: str) -> str:
    aliases = {"l1": "l1_block", "l2": "l2_block", "l1_ball": "l1_block", "l2_ball": "l2_block"}
    return aliases.get(name, name)


def full_lmo(gradient: np.ndarray, radius: float, slices: list[slice], name: str) -> np.ndarray:
    kind = lmo_name(name)
    if kind == "box":
        return full_box_lmo(gradient, radius)
    out = np.zeros_like(gradient)
    for sl in slices:
        out[..., sl] = block_lmo(gradient[..., sl], radius, kind)
    return out


def block_lmo(gradient: np.ndarray, radius: float, name: str) -> np.ndarray:
    kind = lmo_name(name)
    if kind == "box":
        return full_box_lmo(gradient, radius)
    if kind == "l2_block":
        return _l2_lmo(gradient, radius)
    if kind == "l1_block":
        return _l1_lmo(gradient, radius)
    raise ValueError(f"unknown lmo: {name}")


def feasible_start(point: np.ndarray, radius: float, slices: list[slice], name: str) -> np.ndarray:
    kind = lmo_name(name)
    if kind == "box":
        return np.clip(point, -radius, radius)
    out = point.copy()
    for sl in slices:
        out[sl] = _scale_to_ball(out[sl], radius, ord_value=1 if kind == "l1_block" else 2)
    return out


def _l2_lmo(gradient: np.ndarray, radius: float) -> np.ndarray:
    norms = np.linalg.norm(gradient, axis=-1, keepdims=True)
    return np.divide(-radius * gradient, norms, out=np.zeros_like(gradient), where=norms > 0)


def _l1_lmo(gradient: np.ndarray, radius: float) -> np.ndarray:
    out = np.zeros_like(gradient)
    idx = np.argmax(np.abs(gradient), axis=-1)
    rows = np.arange(gradient.shape[0]) if gradient.ndim > 1 else None
    if gradient.ndim == 1:
        out[int(idx)] = -radius * np.sign(gradient[int(idx)])
    else:
        out[rows, idx] = -radius * np.sign(gradient[rows, idx])
    return out


def _scale_to_ball(vector: np.ndarray, radius: float, ord_value: int) -> np.ndarray:
    norm = np.linalg.norm(vector, ord=ord_value)
    if norm <= radius or norm == 0:
        return vector
    return vector * (radius / norm)


def block_box_lmo(
    gradient: np.ndarray,
    base: np.ndarray,
    radius: float,
    slices: list[slice],
    block_ids: np.ndarray,
) -> np.ndarray:
    out = base.copy()
    for block_id in block_ids:
        sl = slices[int(block_id)]
        out[sl] = full_box_lmo(gradient[sl], radius)
    return out


def selected_coordinates(slices: list[slice], block_ids: np.ndarray) -> int:
    total = 0
    for block_id in block_ids:
        sl = slices[int(block_id)]
        total += sl.stop - sl.start
    return total
