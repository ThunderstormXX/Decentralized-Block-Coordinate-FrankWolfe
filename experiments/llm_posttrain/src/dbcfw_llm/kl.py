from __future__ import annotations

import torch
from torch.nn import functional as F


def exact_forward_token_kl(reference_logits: torch.Tensor, policy_logits: torch.Tensor) -> torch.Tensor:
    """KL(p_ref || p_policy) for every batch/token position."""
    ref_log = F.log_softmax(reference_logits.float(), dim=-1)
    policy_log = F.log_softmax(policy_logits.float(), dim=-1)
    ref_prob = ref_log.exp()
    return (ref_prob * (ref_log - policy_log)).sum(dim=-1)


def sampled_reverse_kl_per_token(policy_logps: torch.Tensor, reference_logps: torch.Tensor) -> torch.Tensor:
    """Monte Carlo estimator of KL(pi_policy || pi_ref) on policy samples."""
    if policy_logps.shape != reference_logps.shape:
        raise ValueError("Policy and reference log-prob tensors must have equal shape")
    return policy_logps - reference_logps


def kl_quantiles(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    return {
        "mean": float(flat.mean().item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
    }
