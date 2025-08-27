import torch


def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None= None,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)