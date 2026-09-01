from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from dbcfw_bench.objective import ReferenceSolution


@dataclass
class EuclideanFlowMatchingProblem:
    features_parts: list[np.ndarray]
    velocity_parts: list[np.ndarray]
    flow_dim: int
    dim: int
    reg: float
    box_radius: float

    @property
    def agents(self) -> int:
        return len(self.features_parts)

    @property
    def feature_dim(self) -> int:
        return self.flow_dim + 2

    @property
    def param_dim(self) -> int:
        return self.feature_dim * self.flow_dim

    def local_value(self, i: int, x: np.ndarray) -> float:
        residual = self.features_parts[i] @ self._weights(x) - self.velocity_parts[i]
        loss = 0.5 * float(np.mean(np.sum(residual * residual, axis=1)))
        return loss + 0.5 * self.reg * float(x @ x)

    def objective(self, x: np.ndarray) -> float:
        return sum(self.local_value(i, x) for i in range(self.agents)) / self.agents

    def local_grad(self, i: int, x: np.ndarray) -> np.ndarray:
        phi = self.features_parts[i]
        residual = phi @ self._weights(x) - self.velocity_parts[i]
        grad = phi.T @ residual / len(phi)
        out = np.zeros(self.dim, dtype=float)
        out[:self.param_dim] = grad.ravel()
        return out + self.reg * x

    def grad(self, x: np.ndarray) -> np.ndarray:
        return sum(self.local_grad(i, x) for i in range(self.agents)) / self.agents

    def solve_reference(self, maxiter: int = 300) -> ReferenceSolution:
        if maxiter <= 0:
            return ReferenceSolution(
                np.zeros(self.dim), 0.0, True, "flow_matching_reference_skipped"
            )
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

    def _weights(self, x: np.ndarray) -> np.ndarray:
        return x[:self.param_dim].reshape(self.feature_dim, self.flow_dim)


def make_euclidean_flow_matching_problem(config) -> EuclideanFlowMatchingProblem:
    flow_dim = int(config.hidden_dim)
    param_dim = (flow_dim + 2) * flow_dim
    if config.dim < param_dim:
        raise ValueError(
            "euclidean_flow_matching requires dim >= hidden_dim * (hidden_dim + 2)"
        )

    rng = np.random.default_rng(config.seed)
    centers = _target_centers(np.random.default_rng(config.seed + 2302), flow_dim)
    features_parts: list[np.ndarray] = []
    velocity_parts: list[np.ndarray] = []
    for agent in range(config.agents):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        features, velocity = _draw_agent_samples(
            local_rng, config.samples_per_agent, centers, agent, config.agents
        )
        features_parts.append(features)
        velocity_parts.append(velocity)
    return EuclideanFlowMatchingProblem(
        features_parts,
        velocity_parts,
        flow_dim,
        int(config.dim),
        config.reg,
        config.box_radius,
    )


def _draw_agent_samples(
    rng: np.random.Generator,
    samples: int,
    centers: np.ndarray,
    agent: int,
    agents: int,
) -> tuple[np.ndarray, np.ndarray]:
    flow_dim = centers.shape[1]
    weights = np.full(len(centers), 1.0)
    weights[agent % len(centers)] += 1.5
    if agents > 1:
        weights[(agent + 1) % len(centers)] += agent / (agents - 1)
    weights /= weights.sum()

    x0 = rng.normal(0.0, 1.0, size=(samples, flow_dim))
    choices = rng.choice(len(centers), size=samples, p=weights)
    x1 = centers[choices] + rng.normal(0.0, 0.35, size=(samples, flow_dim))
    t = rng.uniform(0.0, 1.0, size=(samples, 1))
    z_t = (1.0 - t) * x0 + t * x1
    velocity = x1 - x0
    features = np.hstack([z_t, t, np.ones((samples, 1))])
    return features.astype(float, copy=False), velocity.astype(float, copy=False)


def _target_centers(rng: np.random.Generator, flow_dim: int) -> np.ndarray:
    centers = rng.normal(0.0, 1.0, size=(4, flow_dim))
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    return 2.0 * centers / np.maximum(norms, 1e-12)
