from __future__ import annotations

import torch
from torch import nn

from dbcfw_llm.nuclear_adapter import NuclearFWLinear, leading_singular_pair


def test_lmo_returns_leading_pair() -> None:
    matrix = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
    left, right, sigma = leading_singular_pair(matrix, iterations=20)
    assert abs(sigma - 4.0) < 1e-4
    assert torch.allclose(torch.abs(left), torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)
    assert torch.allclose(torch.abs(right), torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)


def test_fw_atoms_reconstruct_and_respect_nuclear_ball() -> None:
    layer = NuclearFWLinear(nn.Linear(3, 2, bias=False))
    radius = 0.7
    for gamma in (1.0, 0.5, 0.25):
        layer.delta.grad = torch.randn_like(layer.delta)
        layer.fw_update(radius=radius, gamma=gamma, lmo_iters=20)
    assert torch.allclose(layer.delta.cpu(), layer.atom_reconstruction(), atol=1e-5)
    assert layer.nuclear_norm() <= radius + 1e-5
    assert layer.numerical_rank() <= len(layer.atoms)


def test_unrecorded_candidate_does_not_pollute_atom_history() -> None:
    layer = NuclearFWLinear(nn.Linear(3, 2, bias=False))
    gradient = torch.randn_like(layer.delta)
    left, right, _ = layer.propose_atom(gradient, lmo_iters=10)
    layer.apply_atom(left, right, radius=0.7, gamma=0.5, record=False)
    assert layer.atoms == []
