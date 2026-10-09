"""Shared tensor primitives: masked reducers and entropy.

These are the leaves consumed by the SFT and GRPO losses; every function is
pure (no model calls, no in-place ops) so the snapshot tests pin their
semantics exactly.
"""

import torch


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Mean of `tensor` over unmasked elements.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, 1/True where an element participates.
        dim: int | None, dimension to reduce. If None, reduce over all
            unmasked elements (scalar result), matching ``tensor.mean(dim)``
            shape semantics.

    Returns:
        torch.Tensor: sum(tensor * mask, dim) / sum(mask, dim).
    """
    mask = mask.to(tensor.dtype)
    total = (tensor * mask).sum(dim=dim)
    count = mask.sum(dim=dim)
    return total / count


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum of `tensor` over unmasked elements, divided by a constant.

    The "Dr. GRPO" no-mean-normalization primitive: masked-out elements
    contribute nothing to the sum.

    Args:
        tensor: torch.Tensor, the tensor to sum.
        mask: torch.Tensor, 1/True where an element participates.
        dim: int | None, dimension to sum along. If None, sum over all dims.
        normalize_constant: float, the constant to divide the sum by.

    Returns:
        torch.Tensor: sum(tensor * mask, dim) / normalize_constant.
    """
    mask = mask.to(tensor.dtype)
    return (tensor * mask).sum(dim=dim) / normalize_constant


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of the distribution given by `logits` over the final dimension.

    H(p) = -sum_v p_v log p_v (handout Eq. 1), computed via log_softmax for
    numerical stability; zero-probability entries contribute exactly zero
    rather than 0 * -inf = NaN.

    Args:
        logits: torch.Tensor of shape (..., vocab_size).

    Returns:
        torch.Tensor of shape (...): per-position entropy.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    p_log_p = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
    return -p_log_p.sum(dim=-1)
