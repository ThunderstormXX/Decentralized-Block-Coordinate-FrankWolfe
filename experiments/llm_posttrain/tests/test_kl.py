from __future__ import annotations

import torch

from dbcfw_llm.kl import exact_forward_token_kl, sampled_reverse_kl_per_token


def test_exact_forward_kl_is_zero_for_equal_logits() -> None:
    logits = torch.randn(2, 3, 5)
    result = exact_forward_token_kl(logits, logits)
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


def test_sampled_reverse_kl_is_log_ratio() -> None:
    policy = torch.tensor([[-1.0, -2.0]])
    reference = torch.tensor([[-1.5, -1.5]])
    assert torch.equal(sampled_reverse_kl_per_token(policy, reference), policy - reference)
