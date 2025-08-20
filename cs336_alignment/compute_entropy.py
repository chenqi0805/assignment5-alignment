from einops import einsum
import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logsumexp = torch.logsumexp(logits, dim=-1)
    logits = logits - logsumexp.unsqueeze(-1)
    return - einsum(torch.exp(logits), logits, "... seq_len vocab_size, ... seq_len vocab_size -> ... seq_len")