from __future__ import annotations

import numpy as np

from dbcfw_bench.config import RunConfig
from dbcfw_bench.objective_logreg import LogisticProblem


def make_synthetic_nlp_problem(config: RunConfig, family: str) -> LogisticProblem:
    rng = np.random.default_rng(config.seed)
    vocab = config.dim - 1
    if vocab < 16:
        raise ValueError("synthetic NLP dim must be at least 17")
    parts_x, parts_y = [], []
    for agent in range(config.agents):
        skew = (agent - 0.5 * (config.agents - 1)) / max(config.agents - 1, 1)
        local = np.random.default_rng(rng.integers(0, 2**32 - 1))
        x_i, y_i = _draw_agent(local, config.samples_per_agent, vocab, family, skew)
        parts_x.append(x_i)
        parts_y.append(y_i)
    return LogisticProblem(parts_x, parts_y, config.reg, config.box_radius)


def _draw_agent(
    rng: np.random.Generator,
    samples: int,
    vocab: int,
    family: str,
    skew: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((samples, vocab + 1), dtype=float)
    y = np.where(rng.random(samples) < 0.5 + 0.28 * skew, 1.0, -1.0)
    widths = _bands(vocab, family)
    for row, label in enumerate(y):
        length = int(rng.integers(18, 45))
        signal = 0.78 if family == "topic" else 0.63
        if rng.random() < 0.18:
            label = -label
        _fill_doc(x[row, :-1], rng, label, length, widths, signal)
    norms = np.linalg.norm(x[:, :-1], axis=1, keepdims=True)
    x[:, :-1] /= np.maximum(norms, 1.0)
    x[:, -1] = 1.0
    return x, y


def _bands(vocab: int, family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = max(4, min(vocab // 5, 256 if family == "topic" else 96))
    pos = np.arange(0, signal)
    neg = np.arange(signal, 2 * signal)
    neutral = np.arange(2 * signal, vocab)
    if len(neutral) == 0:
        neutral = np.arange(vocab)
    return pos, neg, neutral


def _fill_doc(
    row: np.ndarray,
    rng: np.random.Generator,
    label: float,
    length: int,
    bands: tuple[np.ndarray, np.ndarray, np.ndarray],
    signal_prob: float,
) -> None:
    pos, neg, neutral = bands
    good = pos if label > 0 else neg
    bad = neg if label > 0 else pos
    for _ in range(length):
        draw = rng.random()
        if draw < signal_prob:
            token = int(rng.choice(good))
        elif draw < signal_prob + 0.08:
            token = int(rng.choice(bad))
        else:
            token = int(rng.choice(neutral))
        row[token] += 1.0
