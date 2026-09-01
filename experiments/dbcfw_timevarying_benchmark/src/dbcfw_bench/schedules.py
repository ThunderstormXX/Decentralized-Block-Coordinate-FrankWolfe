from __future__ import annotations

import numpy as np


def fw_step_size(iteration: int, offset: float = 2.0) -> float:
    return 2.0 / (iteration + offset)


def choose_blocks(rng: np.random.Generator, n_blocks: int, batch: int) -> np.ndarray:
    if batch >= n_blocks:
        return np.arange(n_blocks)
    return rng.choice(n_blocks, size=batch, replace=False)
