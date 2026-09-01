from __future__ import annotations

import torch

from dbcfw_llm.rlvr import exact_answer_reward, extract_final_number, group_advantages


def test_exact_answer_reward_uses_final_answer() -> None:
    assert extract_final_number("work 12 then #### 7") == "7"
    assert exact_answer_reward("reasoning... #### 1,024", "answer #### 1024") == 1.0
    assert exact_answer_reward("#### 7.0", "#### 7") == 1.0
    assert exact_answer_reward("#### 12", "#### 13") == 0.0


def test_group_advantages_are_centered_without_std_scaling() -> None:
    advantages = group_advantages(torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert torch.allclose(advantages, torch.tensor([-0.5, 0.5, 0.5, -0.5]))
    assert torch.isclose(advantages.mean(), torch.tensor(0.0))
