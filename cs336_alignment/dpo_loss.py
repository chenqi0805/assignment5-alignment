import torch

from transformers import PreTrainedTokenizerBase

from cs336_alignment.data_loading import ALPACA_TEMPLATE

def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
):
    good = ALPACA_TEMPLATE.format(instruction=prompt, response=response_chosen) + tokenizer.eos_token
    bad = ALPACA_TEMPLATE.format(instruction=prompt, response=response_rejected) + tokenizer.eos_token
    good_ids = tokenizer.encode(good, return_tensors="pt")
    bad_ids = tokenizer.encode(bad, return_tensors="pt")

    with torch.no_grad():
        good_logits = lm(good_ids).logits
        bad_logits = lm(bad_ids).logits

    good_ref_logits = lm_ref(good_ids).logits
    bad_ref_logits = lm_ref(bad_ids).logits

    good_log_probs = torch.log_softmax(good_logits, dim=-1)
    bad_log_probs = torch.log_softmax(bad_logits, dim=-1)
    good_ref_log_probs = torch.log_softmax(good_ref_logits, dim=-1)
    bad_ref_log_probs = torch.log_softmax(bad_ref_logits, dim=-1)

    good_log_prob = good_log_probs[:, :-1, :].gather(-1, good_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    bad_log_prob = bad_log_probs[:, :-1, :].gather(-1, bad_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    good_ref_log_prob = good_ref_log_probs[:, :-1, :].gather(-1, good_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    bad_ref_log_prob = bad_ref_log_probs[:, :-1, :].gather(-1, bad_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)

    good_ratios, bad_ratios = good_log_prob - good_ref_log_prob, bad_log_prob - bad_ref_log_prob

    logits = beta * (good_ratios - bad_ratios)
    
    return - torch.nn.functional.logsigmoid(logits)