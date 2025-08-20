import torch

from cs336_alignment.compute_entropy import compute_entropy

def get_response_log_probs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    log_probs_for_labels = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    result = {
        "log_probs": log_probs_for_labels,
    }
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result