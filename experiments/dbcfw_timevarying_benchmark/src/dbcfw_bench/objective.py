from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class ReferenceSolution:
    x: np.ndarray
    value: float
    success: bool
    message: str


@dataclass
class QuadraticProblem:
    a_parts: list[np.ndarray]
    b_parts: list[np.ndarray]
    reg: float
    box_radius: float

    @property
    def agents(self) -> int:
        return len(self.a_parts)

    @property
    def dim(self) -> int:
        return int(self.a_parts[0].shape[1])

    def local_value(self, i: int, x: np.ndarray) -> float:
        residual = self.a_parts[i] @ x - self.b_parts[i]
        loss = 0.5 * float(residual @ residual) / len(self.b_parts[i])
        return loss + 0.5 * self.reg * float(x @ x)

    def objective(self, x: np.ndarray) -> float:
        return sum(self.local_value(i, x) for i in range(self.agents)) / self.agents

    def local_grad(self, i: int, x: np.ndarray) -> np.ndarray:
        residual = self.a_parts[i] @ x - self.b_parts[i]
        grad = self.a_parts[i].T @ residual / len(self.b_parts[i])
        return grad + self.reg * x

    def grad(self, x: np.ndarray) -> np.ndarray:
        grads = (self.local_grad(i, x) for i in range(self.agents))
        return sum(grads) / self.agents

    def solve_reference(self, maxiter: int = 300) -> ReferenceSolution:
        bounds = [(-self.box_radius, self.box_radius)] * self.dim

        def fun(x: np.ndarray) -> float:
            return self.objective(x)

        def jac(x: np.ndarray) -> np.ndarray:
            return self.grad(x)

        result = minimize(
            fun,
            np.zeros(self.dim),
            method="L-BFGS-B",
            jac=jac,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
        )
        return ReferenceSolution(result.x, float(result.fun), result.success, result.message)
