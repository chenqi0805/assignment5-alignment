from typing import Callable

from cv2 import norm
import torch


def compute_group_normalized_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    raw_rewards = torch.tensor(
        [reward_fn(response, ground_truth)["reward"] for response, ground_truth in zip(rollout_responses, repeated_ground_truths)],
        dtype=torch.float32)
    raw_rewards = raw_rewards.view(-1, group_size)
    rewards = raw_rewards - raw_rewards.mean(dim=-1, keepdim=True)
    if normalize_by_std:
        reward_stds = raw_rewards.std(dim=-1, keepdim=True)
        rewards = rewards / (reward_stds + advantage_eps)
    return (rewards.flatten(), raw_rewards.flatten(), {})