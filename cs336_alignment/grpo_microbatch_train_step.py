from typing import Literal
import torch

from cs336_alignment.compute_policy_gradient_loss import compute_policy_gradient_loss
from cs336_alignment.masked_mean import masked_mean


def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None= None,
        advantages: torch.Tensor | None= None,
        old_log_probs: torch.Tensor | None= None,
        cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_sum, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    loss = masked_mean(loss_sum, response_mask) / gradient_accumulation_steps
    loss.backward()

    return loss.detach(), metadata