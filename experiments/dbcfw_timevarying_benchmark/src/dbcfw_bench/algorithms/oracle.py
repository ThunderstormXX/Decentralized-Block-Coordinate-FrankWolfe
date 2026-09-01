from __future__ import annotations

import numpy as np

from dbcfw_bench.config import RunConfig
from dbcfw_bench.lmo import block_lmo, full_lmo
from dbcfw_bench.schedules import choose_blocks


def fw_oracle_update(
    trackers: np.ndarray,
    x_tilde: np.ndarray,
    config: RunConfig,
    slices: list[slice],
    rng: np.random.Generator,
    gamma: float,
) -> tuple[np.ndarray, int, int]:
    if config.batch >= config.blocks:
        v = full_lmo(trackers, config.box_radius, slices, config.lmo)
        x_next = (1.0 - gamma) * x_tilde + gamma * v
        return x_next, config.agents * config.dim, v.nbytes
    return _block_update(trackers, x_tilde, config, slices, rng, gamma)


def _block_update(
    trackers: np.ndarray,
    x_tilde: np.ndarray,
    config: RunConfig,
    slices: list[slice],
    rng: np.random.Generator,
    gamma: float,
) -> tuple[np.ndarray, int, int]:
    x_next = x_tilde.copy()
    coords = 0
    for agent in range(config.agents):
        block_ids = choose_blocks(rng, config.blocks, config.batch)
        for idx in block_ids:
            sl = slices[int(idx)]
            target = block_lmo(trackers[agent, sl], config.box_radius, config.lmo)
            x_next[agent, sl] = (1.0 - gamma) * x_tilde[agent, sl] + gamma * target
            coords += sl.stop - sl.start
    return x_next, coords, 0
