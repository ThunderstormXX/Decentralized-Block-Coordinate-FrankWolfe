from __future__ import annotations

import torch
from torch import nn

from dbcfw_llm.lora_adapter import LoRALinear


def test_lora_starts_as_zero_displacement_and_has_bounded_rank() -> None:
    layer = LoRALinear(nn.Linear(5, 4, bias=False), rank=2, alpha=4.0)
    assert layer.lora_a.device == layer.base.weight.device
    assert layer.lora_b.device == layer.base.weight.device
    assert torch.count_nonzero(layer.delta_matrix()) == 0
    with torch.no_grad():
        layer.lora_b.normal_()
    assert layer.numerical_rank() <= 2
