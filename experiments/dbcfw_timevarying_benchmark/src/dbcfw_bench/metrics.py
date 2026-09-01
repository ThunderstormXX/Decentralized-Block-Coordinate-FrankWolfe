from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from dbcfw_bench.lmo import block_slices, lmo_name


@dataclass
class MetricRow:
    run_id: str
    method: str
    graph: str
    agents: int
    dim: int
    blocks: int
    batch: int
    seed: int
    graph_seed: int
    iteration: int
    gamma: float
    wall_time_sec: float
    objective_gap: float
    mean_agent_accuracy: float
    consensus_error: float
    gradient_tracker_disagreement: float
    oracle_coordinates_per_iter: int
    total_oracle_coordinates: int
    lambda2: float
    theoretical_state_bytes: int
    boundary_activity: float
    test_error: float = float("nan")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def consensus_error(points: np.ndarray) -> float:
    avg = points.mean(axis=0)
    return float(np.linalg.norm(points - avg, axis=1).mean())


def tracker_disagreement(trackers: np.ndarray) -> float:
    avg = trackers.mean(axis=0)
    return float(np.linalg.norm(trackers - avg, axis=1).mean())


def arrays_nbytes(*arrays: np.ndarray) -> int:
    return int(sum(arr.nbytes for arr in arrays))


def boundary_activity(points: np.ndarray, radius: float, lmo: str, blocks: int) -> float:
    kind = lmo_name(lmo)
    if radius <= 0:
        return 0.0
    if kind == "box":
        return float(np.mean(np.abs(points) >= 0.98 * radius))
    norms = []
    order = 1 if kind == "l1_block" else 2
    for sl in block_slices(points.shape[1], blocks):
        norms.append(np.linalg.norm(points[:, sl], ord=order, axis=1))
    block_norms = np.stack(norms, axis=1)
    return float(np.mean(block_norms >= 0.98 * radius))
