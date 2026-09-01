from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from dbcfw_bench.objective import ReferenceSolution


@dataclass
class LogisticProblem:
    x_parts: list[np.ndarray]
    y_parts: list[np.ndarray]
    reg: float
    box_radius: float

    @property
    def agents(self) -> int:
        return len(self.x_parts)

    @property
    def dim(self) -> int:
        return int(self.x_parts[0].shape[1])

    def local_value(self, i: int, x: np.ndarray) -> float:
        margins = self.y_parts[i] * (self.x_parts[i] @ x)
        loss = float(np.logaddexp(0.0, -margins).mean())
        return loss + 0.5 * self.reg * float(x @ x)

    def objective(self, x: np.ndarray) -> float:
        return sum(self.local_value(i, x) for i in range(self.agents)) / self.agents

    def local_grad(self, i: int, x: np.ndarray) -> np.ndarray:
        margins = self.y_parts[i] * (self.x_parts[i] @ x)
        coeff = -self.y_parts[i] * expit(-margins)
        grad = self.x_parts[i].T @ coeff / len(self.y_parts[i])
        return grad + self.reg * x

    def grad(self, x: np.ndarray) -> np.ndarray:
        return sum(self.local_grad(i, x) for i in range(self.agents)) / self.agents

    def mean_agent_accuracy(self, points: np.ndarray) -> float:
        values = []
        for i in range(self.agents):
            pred = np.where(self.x_parts[i] @ points[i] >= 0.0, 1.0, -1.0)
            values.append(float(np.mean(pred == self.y_parts[i])))
        return float(np.mean(values))

    def solve_reference(self, maxiter: int = 300) -> ReferenceSolution:
        if maxiter <= 0:
            return ReferenceSolution(np.zeros(self.dim), 0.0, True, "logistic_lower_bound")
        bounds = [(-self.box_radius, self.box_radius)] * self.dim
        result = minimize(
            self.objective,
            np.zeros(self.dim),
            method="L-BFGS-B",
            jac=self.grad,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
        )
        return ReferenceSolution(result.x, float(result.fun), result.success, result.message)
