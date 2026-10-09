"""Per-instance DPO loss (CS336 Assignment 5 supplement, Eq. 3)."""

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

PROMPTS_DIR = Path(__file__).parent / "prompts"
ALPACA_SFT_PROMPT_PATH = PROMPTS_DIR / "alpaca_sft.prompt"


def _sequence_log_prob(lm: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, text: str) -> torch.Tensor:
    """Sum of log p(token_t | token_<t>) over the full tokenized sequence.

    The prompt log-probs cancel in the DPO difference, so summing the
    unconditional sequence log-prob for chosen and rejected (which share the
    prompt prefix) is equivalent to differencing response-only log-probs.
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(lm.device)
    logits = lm(input_ids).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # position t's logits predict token t+1
    return log_probs[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """DPO loss for one preference pair (supplement Eq. 3).

    ell = -log sigma( beta [ (log pi(y_w|x) - log pi_ref(y_w|x))
                            - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

    Each full sequence is formatted with the Alpaca instruction template plus
    the EOS token appended after the response. The reference model may live
    on a different device; its log-probs are moved to ``lm``'s device and it
    runs under no_grad (it is frozen by definition).

    Args:
        lm: torch.nn.Module, language model being trained.
        lm_ref: torch.nn.Module, reference language model.
        tokenizer: PreTrainedTokenizerBase, tokenizer for both models.
        beta: float, DPO beta hyperparameter.
        prompt: str, prompt for this instance of preference pair.
        response_chosen: str, preferred response to the prompt.
        response_rejected: str, rejected response to the prompt.

    Returns:
        torch.Tensor scalar with the DPO loss, on ``lm``'s device.
    """
    template = ALPACA_SFT_PROMPT_PATH.read_text()

    def formatted(response: str) -> str:
        # The template's trailing newline is stripped so the EOS token
        # directly follows the response (matches the numeric target).
        return template.format(instruction=prompt, response=response).rstrip() + tokenizer.eos_token

    with torch.no_grad():
        ref_chosen = _sequence_log_prob(lm_ref, tokenizer, formatted(response_chosen))
        ref_rejected = _sequence_log_prob(lm_ref, tokenizer, formatted(response_rejected))
        ref_chosen = ref_chosen.to(lm.device)
        ref_rejected = ref_rejected.to(lm.device)

    policy_chosen = _sequence_log_prob(lm, tokenizer, formatted(response_chosen))
    policy_rejected = _sequence_log_prob(lm, tokenizer, formatted(response_rejected))

    dpo_logits = beta * (
        (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    )
    return -F.logsigmoid(dpo_logits)
