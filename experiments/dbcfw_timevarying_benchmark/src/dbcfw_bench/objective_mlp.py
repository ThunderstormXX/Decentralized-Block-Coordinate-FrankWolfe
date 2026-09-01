from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, softmax

from dbcfw_bench.objective import ReferenceSolution


@dataclass
class MLPProblem:
    x_parts: list[np.ndarray]
    y_parts: list[np.ndarray]
    hidden: int
    reg: float
    box_radius: float
    classes: int = 10

    @property
    def agents(self) -> int:
        return len(self.x_parts)

    @property
    def input_dim(self) -> int:
        return int(self.x_parts[0].shape[1])

    @property
    def param_dim(self) -> int:
        return self.input_dim * self.hidden + self.hidden + self.hidden * self.classes + self.classes

    @property
    def dim(self) -> int:
        return int(np.ceil(self.param_dim / 20) * 20)

    def local_value(self, i: int, x: np.ndarray) -> float:
        _, logits = self._forward(self.x_parts[i], x)
        labels = self.y_parts[i]
        loss = (logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]).mean()
        return float(loss) + 0.5 * self.reg * float(x @ x)

    def objective(self, x: np.ndarray) -> float:
        return sum(self.local_value(i, x) for i in range(self.agents)) / self.agents

    def local_grad(self, i: int, x: np.ndarray) -> np.ndarray:
        xdata, labels = self.x_parts[i], self.y_parts[i]
        hidden, logits = self._forward(xdata, x)
        probs = softmax(logits, axis=1)
        probs[np.arange(len(labels)), labels] -= 1.0
        w1, _, w2, _ = self._unpack(x)
        gw2 = hidden.T @ probs / len(labels)
        gb2 = probs.mean(axis=0)
        dh = (probs @ w2.T) * (hidden > 0.0)
        gw1 = xdata.T @ dh / len(labels)
        gb1 = dh.mean(axis=0)
        grad = np.concatenate([gw1.ravel(), gb1, gw2.ravel(), gb2])
        return np.pad(grad, (0, self.dim - self.param_dim)) + self.reg * x

    def mean_agent_accuracy(self, points: np.ndarray) -> float:
        values = []
        for i in range(self.agents):
            _, logits = self._forward(self.x_parts[i], points[i])
            values.append(float(np.mean(np.argmax(logits, axis=1) == self.y_parts[i])))
        return float(np.mean(values))

    def solve_reference(self, maxiter: int = 300) -> ReferenceSolution:
        return ReferenceSolution(np.zeros(self.dim), 0.0, True, "cross_entropy_lower_bound")

    def initial_point(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 7919)
        s1 = np.sqrt(2.0 / max(1, self.input_dim))
        s2 = np.sqrt(2.0 / max(1, self.hidden))
        w1 = rng.normal(0.0, s1, (self.input_dim, self.hidden))
        b1 = np.zeros(self.hidden)
        w2 = rng.normal(0.0, s2, (self.hidden, self.classes))
        b2 = np.zeros(self.classes)
        params = np.concatenate([w1.ravel(), b1, w2.ravel(), b2])
        params = np.pad(params, (0, self.dim - self.param_dim))
        return np.clip(params, -self.box_radius, self.box_radius)

    def _forward(self, xdata: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w1, b1, w2, b2 = self._unpack(params)
        hidden = np.maximum(0.0, xdata @ w1 + b1)
        return hidden, hidden @ w2 + b2

    def _unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p, h, c = self.input_dim, self.hidden, self.classes
        cur = 0
        w1 = params[cur:cur + p * h].reshape(p, h); cur += p * h
        b1 = params[cur:cur + h]; cur += h
        w2 = params[cur:cur + h * c].reshape(h, c); cur += h * c
        return w1, b1, w2, params[cur:cur + c]
