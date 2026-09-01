from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, softmax

from dbcfw_bench.objective import ReferenceSolution


@dataclass
class ShallowCNNProblem:
    patch_parts: list[np.ndarray]
    y_parts: list[np.ndarray]
    filters: int
    reg: float
    box_radius: float
    classes: int = 10

    @property
    def agents(self) -> int:
        return len(self.patch_parts)

    @property
    def kernel_dim(self) -> int:
        return int(self.patch_parts[0].shape[2])

    @property
    def param_dim(self) -> int:
        return self.kernel_dim * self.filters + self.filters + self.filters * self.classes + self.classes

    @property
    def dim(self) -> int:
        return int(np.ceil(self.param_dim / 20) * 20)

    def local_value(self, i: int, x: np.ndarray) -> float:
        _, logits = self._forward(self.patch_parts[i], x)
        y = self.y_parts[i]
        loss = (logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]).mean()
        return float(loss) + 0.5 * self.reg * float(x @ x)

    def objective(self, x: np.ndarray) -> float:
        return sum(self.local_value(i, x) for i in range(self.agents)) / self.agents

    def local_grad(self, i: int, x: np.ndarray) -> np.ndarray:
        patches, y = self.patch_parts[i], self.y_parts[i]
        pooled, logits, z = self._forward_train(patches, x)
        probs = softmax(logits, axis=1)
        probs[np.arange(len(y)), y] -= 1.0
        conv, _, wc, _ = self._unpack(x)
        scale = 1.0 / len(y)
        gwc = pooled.T @ probs * scale
        gbc = probs.mean(axis=0)
        dpool = probs @ wc.T * scale
        dz = (dpool[:, None, :] / patches.shape[1]) * (z > 0.0)
        gconv = np.einsum("mlp,mlf->pf", patches, dz)
        gbconv = dz.sum(axis=(0, 1))
        grad = np.concatenate([gconv.ravel(), gbconv, gwc.ravel(), gbc])
        return np.pad(grad, (0, self.dim - self.param_dim)) + self.reg * x

    def mean_agent_accuracy(self, points: np.ndarray) -> float:
        acc = []
        for i in range(self.agents):
            _, logits = self._forward(self.patch_parts[i], points[i])
            acc.append(float(np.mean(np.argmax(logits, axis=1) == self.y_parts[i])))
        return float(np.mean(acc))

    def solve_reference(self, maxiter: int = 300) -> ReferenceSolution:
        return ReferenceSolution(np.zeros(self.dim), 0.0, True, "cross_entropy_lower_bound")

    def initial_point(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 104729)
        s1 = np.sqrt(2.0 / max(1, self.kernel_dim))
        s2 = np.sqrt(2.0 / max(1, self.filters))
        conv = rng.normal(0.0, s1, (self.kernel_dim, self.filters))
        bconv = np.zeros(self.filters)
        wc = rng.normal(0.0, s2, (self.filters, self.classes))
        bc = np.zeros(self.classes)
        params = np.concatenate([conv.ravel(), bconv, wc.ravel(), bc])
        return np.clip(np.pad(params, (0, self.dim - self.param_dim)), -self.box_radius, self.box_radius)

    def _forward(self, patches: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pooled, logits, _ = self._forward_train(patches, params)
        return pooled, logits

    def _forward_train(self, patches: np.ndarray, params: np.ndarray):
        conv, bconv, wc, bc = self._unpack(params)
        z = patches @ conv + bconv
        pooled = np.maximum(0.0, z).mean(axis=1)
        return pooled, pooled @ wc + bc, z

    def _unpack(self, params: np.ndarray):
        k, f, c = self.kernel_dim, self.filters, self.classes
        cur = 0
        conv = params[cur:cur + k * f].reshape(k, f); cur += k * f
        bconv = params[cur:cur + f]; cur += f
        wc = params[cur:cur + f * c].reshape(f, c); cur += f * c
        return conv, bconv, wc, params[cur:cur + c]
