from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from dbcfw_bench.objective import ReferenceSolution

@dataclass
class TorchFashionCNNProblem:
    x_parts: list[np.ndarray]
    y_parts: list[np.ndarray]
    filters: int
    reg: float
    box_radius: float
    classes: int = 10

    def __post_init__(self) -> None:
        self.x_parts = [torch.as_tensor(x, dtype=torch.float32) for x in self.x_parts]
        self.y_parts = [torch.as_tensor(y, dtype=torch.long) for y in self.y_parts]
    @property
    def agents(self) -> int:
        return len(self.x_parts)
    @property
    def param_dim(self) -> int:
        f1, f2, h = self.filters, 2 * self.filters, 4 * self.filters
        flat = f2 * 3 * 3
        return f1 * 9 + f1 + f2 * f1 * 9 + f2 + flat * h + h + h * self.classes + self.classes

    @property
    def dim(self) -> int:
        return int(np.ceil(self.param_dim / 20) * 20)
    def local_value(self, i: int, x: np.ndarray) -> float:
        with torch.no_grad():
            params = torch.as_tensor(x[:self.param_dim], dtype=torch.float32)
            loss = F.cross_entropy(self._forward(self.x_parts[i], params), self.y_parts[i])
        return float(loss.item()) + 0.5 * self.reg * float(x @ x)

    def objective(self, x: np.ndarray) -> float:
        return sum(self.local_value(i, x) for i in range(self.agents)) / self.agents

    def local_grad(self, i: int, x: np.ndarray) -> np.ndarray:
        params = torch.tensor(x[:self.param_dim], dtype=torch.float32, requires_grad=True)
        loss = F.cross_entropy(self._forward(self.x_parts[i], params), self.y_parts[i])
        loss.backward()
        grad = params.grad.detach().numpy().astype(float, copy=False)
        return np.pad(grad, (0, self.dim - self.param_dim)) + self.reg * x

    def mean_agent_accuracy(self, points: np.ndarray) -> float:
        values = []
        with torch.no_grad():
            for i in range(self.agents):
                params = torch.as_tensor(points[i, :self.param_dim], dtype=torch.float32)
                pred = self._forward(self.x_parts[i], params).argmax(dim=1)
                values.append(float((pred == self.y_parts[i]).float().mean().item()))
        return float(np.mean(values))

    def solve_reference(self, maxiter: int = 300) -> ReferenceSolution:
        return ReferenceSolution(np.zeros(self.dim), 0.0, True, "cross_entropy_lower_bound")

    def initial_point(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 31337)
        f1, f2, h = self.filters, 2 * self.filters, 4 * self.filters
        flat = f2 * 3 * 3
        w1 = rng.normal(0.0, np.sqrt(2 / 9), (f1, 1, 3, 3))
        b1 = np.zeros(f1)
        w2 = rng.normal(0.0, np.sqrt(2 / (9 * f1)), (f2, f1, 3, 3))
        b2 = np.zeros(f2)
        w3 = rng.normal(0.0, np.sqrt(2 / flat), (h, flat))
        b3 = np.zeros(h)
        w4 = rng.normal(0.0, np.sqrt(2 / h), (self.classes, h))
        b4 = np.zeros(self.classes)
        params = np.concatenate([w1.ravel(), b1, w2.ravel(), b2, w3.ravel(), b3, w4.ravel(), b4])
        params = np.pad(params, (0, self.dim - self.param_dim))
        return np.clip(params, -self.box_radius, self.box_radius)

    def _forward(self, images: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        w1, b1, w2, b2, w3, b3, w4, b4 = self._unpack(params)
        h = F.relu(F.conv2d(images, w1, b1, padding=1))
        h = F.max_pool2d(h, 2)
        h = F.relu(F.conv2d(h, w2, b2, padding=1))
        h = F.max_pool2d(h, 2).flatten(1)
        h = F.relu(F.linear(h, w3, b3))
        return F.linear(h, w4, b4)

    def _unpack(self, params: torch.Tensor):
        f1, f2, h, cur = self.filters, 2 * self.filters, 4 * self.filters, 0
        flat = f2 * 3 * 3
        w1 = params[cur:cur + f1 * 9].reshape(f1, 1, 3, 3); cur += f1 * 9
        b1 = params[cur:cur + f1]; cur += f1
        w2 = params[cur:cur + f2 * f1 * 9].reshape(f2, f1, 3, 3); cur += f2 * f1 * 9
        b2 = params[cur:cur + f2]; cur += f2
        w3 = params[cur:cur + flat * h].reshape(h, flat); cur += flat * h
        b3 = params[cur:cur + h]; cur += h
        w4 = params[cur:cur + h * self.classes].reshape(self.classes, h); cur += h * self.classes
        return w1, b1, w2, b2, w3, b3, w4, params[cur:cur + self.classes]
