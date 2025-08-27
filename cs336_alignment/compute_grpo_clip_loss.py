import torch


def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    naive_loss = advantages * pi_ratio
    clipped_loss = advantages * torch.clamp(pi_ratio, 1 - cliprange, 1 + cliprange)
    return - torch.minimum(naive_loss, clipped_loss), {}
