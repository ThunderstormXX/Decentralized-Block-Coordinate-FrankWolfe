from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen linear layer plus a standard trainable low-rank displacement."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        device = base.weight.device
        self.lora_a = nn.Parameter(
            torch.empty(self.rank, base.in_features, dtype=torch.float32, device=device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, self.rank, dtype=torch.float32, device=device)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.enabled = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if not self.enabled:
            return output
        hidden = F.linear(inputs, self.lora_a.to(inputs.dtype))
        update = F.linear(hidden, self.lora_b.to(inputs.dtype))
        return output + update * self.scaling

    @torch.no_grad()
    def delta_matrix(self) -> torch.Tensor:
        return (self.lora_b.float() @ self.lora_a.float()) * self.scaling

    @torch.no_grad()
    def nuclear_norm(self) -> float:
        return float(torch.linalg.svdvals(self.delta_matrix()).sum().item())

    @torch.no_grad()
    def numerical_rank(self, tolerance: float = 1e-5) -> int:
        return int((torch.linalg.svdvals(self.delta_matrix()) > tolerance).sum().item())


def inject_lora_adapters(
    model: nn.Module,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    max_modules: int | None = None,
) -> dict[str, LoRALinear]:
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
    adapters = {}
    for full_name, parent, child_name, child in candidates:
        wrapped = LoRALinear(child, rank=rank, alpha=alpha)
        setattr(parent, child_name, wrapped)
        adapters[full_name] = wrapped
    return adapters
