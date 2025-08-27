from typing import Literal
import torch

from cs336_alignment.compute_group_normalized_rewards import compute_group_normalized_rewards
from cs336_alignment.compute_grpo_clip_loss import compute_grpo_clip_loss
from cs336_alignment.compute_naive_policy_gradient_loss import compute_naive_policy_gradient_loss


def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None= None,
        advantages: torch.Tensor | None= None,
        old_log_probs: torch.Tensor | None= None,
        cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    else:
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)