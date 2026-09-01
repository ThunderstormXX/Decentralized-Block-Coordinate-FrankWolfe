from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from dbcfw_bench.objective import ReferenceSolution


@dataclass
class MulticlassLogisticProblem:
    x_parts: list[np.ndarray]
    y_parts: list[np.ndarray]
    reg: float
    box_radius: float
    classes: int = 10

    @property
    def agents(self) -> int:
        return len(self.x_parts)

    @property
    def dim(self) -> int:
        return int(self.x_parts[0].shape[1] * self.classes)

    def local_value(self, i: int, x: np.ndarray) -> float:
        w = x.reshape(self.x_parts[i].shape[1], self.classes)
        logits = self.x_parts[i] @ w
        labels = self.y_parts[i]
        loss = (logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]).mean()
        return float(loss) + 0.5 * self.reg * float(x @ x)

    def objective(self, x: np.ndarray) -> float:
        return sum(self.local_value(i, x) for i in range(self.agents)) / self.agents

    def local_grad(self, i: int, x: np.ndarray) -> np.ndarray:
        p = self.x_parts[i].shape[1]
        w = x.reshape(p, self.classes)
        probs = softmax(self.x_parts[i] @ w, axis=1)
        probs[np.arange(len(self.y_parts[i])), self.y_parts[i]] -= 1.0
        grad = self.x_parts[i].T @ probs / len(self.y_parts[i])
        return grad.ravel() + self.reg * x

    def grad(self, x: np.ndarray) -> np.ndarray:
        return sum(self.local_grad(i, x) for i in range(self.agents)) / self.agents

    def mean_agent_accuracy(self, points: np.ndarray) -> float:
        values = []
        p = self.x_parts[0].shape[1]
        for i in range(self.agents):
            logits = self.x_parts[i] @ points[i].reshape(p, self.classes)
            values.append(float(np.mean(np.argmax(logits, axis=1) == self.y_parts[i])))
        return float(np.mean(values))

    def solve_reference(self, maxiter: int = 300) -> ReferenceSolution:
        if maxiter <= 0:
            return ReferenceSolution(np.zeros(self.dim), 0.0, True, "cross_entropy_lower_bound")
        bounds = [(-self.box_radius, self.box_radius)] * self.dim
        result = minimize(
            self.objective,
            np.zeros(self.dim),
            method="L-BFGS-B",
            jac=self.grad,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )
        return ReferenceSolution(result.x, float(result.fun), result.success, result.message)
