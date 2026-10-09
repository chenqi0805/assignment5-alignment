"""GRPO policy-gradient losses and the group-normalized reward primitive.

Equation numbers refer to the CS336 Assignment 5 handout. The loss functions
are pure (they return per-token losses; only the microbatch train step calls
``backward``).
"""

from typing import Literal

import torch

from .common import masked_mean

GRPO_LOSS_TYPES = ("no_baseline", "reinforce_with_baseline", "grpo_clip")


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Score rollouts and normalize rewards within each consecutive group.

    Groups are consecutive slices of the rollout batch: the ground truths are
    repeated ``group_size`` times, so reshaping rewards to
    ``(-1, group_size)`` lines each response up with its group.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], scores
            (response, ground_truth) pairs.
        rollout_responses: list[str], rollouts from the policy, length
            ``rollout_batch_size``.
        repeated_ground_truths: list[str], ground truths repeated
            ``group_size`` times each.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon inside the denominator (only applied
            when normalizing by std).
        normalize_by_std: bool, whether to divide by the group std (GRPO,
            Eq. 28) or by nothing (Dr. GRPO, Eq. 31).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            advantages of shape (rollout_batch_size,);
            raw rewards of shape (rollout_batch_size,);
            free-form metadata for logging.
    """
    rewards = torch.tensor(
        [
            float(reward_fn(response, ground_truth)["reward"])
            for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
        ],
        dtype=torch.float32,
    )
    grouped = rewards.view(-1, group_size)
    group_mean = grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        group_std = grouped.std(dim=1, keepdim=True)
        advantages = (grouped - group_mean) / (group_std + advantage_eps)
    else:
        advantages = grouped - group_mean

    metadata = {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
    }
    return advantages.flatten(), rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Per-token REINFORCE loss (handout Eq. 32).

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1),
            broadcast over the sequence dimension.
        policy_log_probs: torch.Tensor of shape (batch_size, seq_len).

    Returns:
        torch.Tensor of shape (batch_size, seq_len): per-token loss
            -A(i) * log p_theta(o_t | q, o_<t). Not masked; callers mask.
    """
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GRPO-Clip per-token loss (handout Eq. 33), negated objective.

    The importance ratio is computed by differencing in log space first and
    exponentiating once — never exp(policy) * exp(old), which overflows.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1).
        policy_log_probs: torch.Tensor of shape (batch_size, seq_len).
        old_log_probs: torch.Tensor of shape (batch_size, seq_len).
        cliprange: float, epsilon for clipping the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            per-token loss of shape (batch_size, seq_len);
            metadata with the per-token ``was_clipped`` flag (the min chose
            the clipped branch), from which the clip fraction is derived.
    """
    adv = advantages.view(-1, 1)
    log_ratio = policy_log_probs - old_log_probs
    ratio = log_ratio.exp()
    lhs = ratio * adv
    rhs = ratio.clamp(1.0 - cliprange, 1.0 + cliprange) * adv
    per_token_loss = -torch.minimum(lhs, rhs)
    metadata = {"was_clipped": rhs < lhs}
    return per_token_loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the requested policy-gradient loss, merging metadata.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, seq_len).
        loss_type: one of GRPO_LOSS_TYPES.
        raw_rewards: torch.Tensor of shape (batch_size, 1); required for
            "no_baseline".
        advantages: torch.Tensor of shape (batch_size, 1); required for
            "reinforce_with_baseline" and "grpo_clip".
        old_log_probs: torch.Tensor of shape (batch_size, seq_len); required
            for "grpo_clip".
        cliprange: float; required for "grpo_clip".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: per-token loss and
            metadata.
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("loss_type='no_baseline' requires raw_rewards")
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    if loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError(
                "loss_type='reinforce_with_baseline' requires advantages"
            )
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    if loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError(
                "loss_type='grpo_clip' requires advantages, old_log_probs, and cliprange"
            )
        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
    raise ValueError(f"Unknown loss_type: {loss_type!r}; expected one of {GRPO_LOSS_TYPES}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one microbatch's GRPO loss and backprop it (no grad zeroing).

    The loss is meaned over response tokens per example, then over the
    microbatch, then divided by the gradient accumulation factor so that
    accumulated microbatch gradients average to the full-batch gradient.
    The caller zeroes grads between optimizer steps; backward is called
    exactly once here, so gradients accumulate across calls.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: scalar loss and
            metadata from the underlying loss.
    """
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, metadata
