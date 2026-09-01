from __future__ import annotations

import numpy as np


def stack_local_gradients(problem: object, points: np.ndarray) -> np.ndarray:
    grads = [problem.local_grad(i, points[i]) for i in range(points.shape[0])]
    return np.vstack(grads)
