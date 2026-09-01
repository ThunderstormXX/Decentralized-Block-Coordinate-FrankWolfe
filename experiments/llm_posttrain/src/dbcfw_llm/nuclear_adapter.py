from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FWAtom:
    coefficient: float
    left: torch.Tensor
    right: torch.Tensor


def leading_singular_pair(matrix: torch.Tensor, iterations: int = 8) -> tuple[torch.Tensor, torch.Tensor, float]:
    work = matrix.detach().float()
    if work.ndim != 2:
        raise ValueError("The nuclear-norm LMO expects a matrix gradient")
    right = torch.ones(work.shape[1], device=work.device, dtype=work.dtype)
    right = right / right.norm().clamp_min(1e-12)
    for _ in range(max(1, iterations)):
        left = work @ right
        left = left / left.norm().clamp_min(1e-12)
        right = work.T @ left
        right = right / right.norm().clamp_min(1e-12)
    left = work @ right
    sigma = float(left.norm().item())
    left = left / left.norm().clamp_min(1e-12)
    return left, right, sigma


class NuclearFWLinear(nn.Module):
    """Frozen linear layer plus a nuclear-ball displacement updated by FW."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.delta = nn.Parameter(torch.zeros_like(base.weight, dtype=torch.float32))
        self.enabled = True
        self.atoms: list[FWAtom] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if not self.enabled:
            return output
        delta = self.delta.to(device=inputs.device, dtype=inputs.dtype)
        return output + F.linear(inputs, delta, None)

    @torch.no_grad()
    def propose_atom(
        self, gradient: torch.Tensor | None = None, lmo_iters: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        gradient = self.delta.grad if gradient is None else gradient
        if gradient is None:
            raise RuntimeError("FW update requested before a gradient was computed")
        return leading_singular_pair(gradient, iterations=lmo_iters)

    @torch.no_grad()
    def apply_atom(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        radius: float,
        gamma: float,
        *,
        record: bool = True,
    ) -> None:
        atom = -float(radius) * torch.outer(left, right)
        self.delta.mul_(1.0 - float(gamma)).add_(atom.to(self.delta), alpha=float(gamma))
        if record:
            for previous in self.atoms:
                previous.coefficient *= 1.0 - float(gamma)
            self.atoms.append(
                FWAtom(
                    coefficient=float(gamma) * float(radius),
                    left=(-left).detach().cpu(),
                    right=right.detach().cpu(),
                )
            )

    @torch.no_grad()
    def fw_update(self, radius: float, gamma: float, lmo_iters: int = 8) -> dict[str, float]:
        left, right, sigma = self.propose_atom(lmo_iters=lmo_iters)
        self.apply_atom(left, right, radius, gamma)
        self.delta.grad = None
        return {"lmo_sigma": sigma, "gamma": float(gamma), "radius": float(radius)}

    @torch.no_grad()
    def nuclear_norm(self) -> float:
        return float(torch.linalg.svdvals(self.delta.float()).sum().item())

    @torch.no_grad()
    def numerical_rank(self, tolerance: float = 1e-5) -> int:
        singular = torch.linalg.svdvals(self.delta.float())
        return int((singular > tolerance).sum().item())

    @torch.no_grad()
    def atom_reconstruction(self) -> torch.Tensor:
        reconstructed = torch.zeros_like(self.delta, device="cpu")
        for atom in self.atoms:
            reconstructed.add_(torch.outer(atom.left, atom.right), alpha=atom.coefficient)
        return reconstructed


def inject_nuclear_adapters(
    model: nn.Module, target_suffixes: Iterable[str], max_modules: int | None = None
) -> dict[str, NuclearFWLinear]:
    suffixes = tuple(target_suffixes)
    candidates: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and full_name.endswith(suffixes):
                candidates.append((full_name, parent, child_name, child))
    if max_modules is not None:
        candidates = candidates[: int(max_modules)]
    if not candidates:
        raise ValueError(f"No nn.Linear module matched suffixes {suffixes}")
    adapters: dict[str, NuclearFWLinear] = {}
    for full_name, parent, child_name, child in candidates:
        wrapped = NuclearFWLinear(child)
        setattr(parent, child_name, wrapped)
        adapters[full_name] = wrapped
    return adapters


@contextmanager
def adapters_disabled(adapters: dict[str, nn.Module]):
    previous = {name: adapter.enabled for name, adapter in adapters.items()}
    try:
        for adapter in adapters.values():
            adapter.enabled = False
        yield
    finally:
        for name, adapter in adapters.items():
            adapter.enabled = previous[name]
