from __future__ import annotations

import numpy as np

from dbcfw_bench.config import RunConfig
from dbcfw_bench.objective_structural_svm import StructuralSequenceSVMProblem


def make_structural_sequence_svm_problem(config: RunConfig) -> StructuralSequenceSVMProblem:
    rng = np.random.default_rng(config.seed)
    classes = int(config.label_count)
    length = int(config.sequence_length)
    token_dim = _token_dim(config.dim, classes)
    prototypes = rng.normal(0.0, 1.0, size=(classes, token_dim))
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1.0)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for agent in range(config.agents):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        skew = (agent - 0.5 * (config.agents - 1)) / max(config.agents - 1, 1)
        x_i, y_i = _draw_agent(
            local_rng,
            config.blocks,
            length,
            token_dim,
            classes,
            prototypes,
            skew,
        )
        x_parts.append(x_i)
        y_parts.append(y_i)
    return StructuralSequenceSVMProblem(x_parts, y_parts, config.reg, classes)


def _token_dim(dim: int, classes: int) -> int:
    transition_dim = classes * classes
    if dim <= transition_dim + classes:
        raise ValueError("structural SVM dim is too small for emissions and transitions")
    return max(1, (dim - transition_dim) // classes)


def _draw_agent(
    rng: np.random.Generator,
    samples: int,
    length: int,
    token_dim: int,
    classes: int,
    prototypes: np.ndarray,
    skew: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((samples, length, token_dim), dtype=float)
    y = np.zeros((samples, length), dtype=int)
    start_probs = _skewed_probs(classes, skew)
    transition = _transition_matrix(classes, skew)
    for sample in range(samples):
        y[sample, 0] = int(rng.choice(classes, p=start_probs))
        for token in range(1, length):
            y[sample, token] = int(rng.choice(classes, p=transition[y[sample, token - 1]]))
        x[sample] = prototypes[y[sample]] + 0.45 * rng.normal(size=(length, token_dim))
    norms = np.linalg.norm(x, axis=2, keepdims=True)
    x /= np.maximum(norms, 1.0)
    return x, y


def _skewed_probs(classes: int, skew: float) -> np.ndarray:
    logits = np.linspace(-0.6, 0.6, classes) * skew
    logits -= np.max(logits)
    probs = np.exp(logits)
    return probs / probs.sum()


def _transition_matrix(classes: int, skew: float) -> np.ndarray:
    matrix = np.full((classes, classes), 0.25 / max(classes - 1, 1), dtype=float)
    np.fill_diagonal(matrix, 0.75)
    if classes > 1:
        bias = _skewed_probs(classes, skew)
        matrix = 0.88 * matrix + 0.12 * bias[None, :]
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix
