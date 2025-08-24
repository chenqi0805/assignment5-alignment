import torch

from cs336_alignment.masked_normalize import masked_normalize

def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, seq_length = policy_log_probs.shape
    # Cross-entropy loss (neg log likelihood), per token and masked
    ce_loss = - policy_log_probs                                         # (batch_size, seq_length)
    # Sum over tokens and batch, only response tokens count
    loss_sum = masked_normalize(ce_loss, response_mask, normalize_constant=normalize_constant)

    loss = loss_sum / batch_size / gradient_accumulation_steps
    loss.backward()

    # For logging
    n_tokens = response_mask.sum()
    avg_token_ce = loss_sum / (n_tokens + 1e-8)
    metadata = {
        "loss_sum": loss_sum.detach(),
        "n_tokens": n_tokens.detach(),
        "avg_ce_per_token": avg_token_ce.detach(),
        "mean_log_prob": (policy_log_probs * response_mask).sum() / (n_tokens + 1e-8)
    }

    return loss.detach(), metadata