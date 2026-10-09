"""SFT primitives: tokenization, response log-probs, and the train step."""

import logging

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .common import compute_entropy, masked_normalize

logger = logging.getLogger(__name__)


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build the shifted response mask.

    Each prompt and each output is tokenized separately and concatenated per
    pair. The result is shifted: position t of ``labels`` predicts token t+1,
    so ``input_ids`` drops the final token and ``labels`` drops the first.

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizerBase, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": (batch_size, max(prompt_and_output_lens) - 1) int64.
            "labels": (batch_size, max(prompt_and_output_lens) - 1) int64,
                shifted input_ids.
            "response_mask": (batch_size, max(prompt_and_output_lens) - 1)
                bool, True at label positions that correspond to response
                tokens. Rows shorter than the max are right-padded with
                ``tokenizer.pad_token_id`` (or 0) and masked False.
    """
    concats: list[list[int]] = []
    flags: list[list[bool]] = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt)["input_ids"]
        output_ids = tokenizer(output)["input_ids"]
        concats.append(prompt_ids + output_ids)
        flags.append([False] * len(prompt_ids) + [True] * len(output_ids))

    max_len = max(len(concat) for concat in concats)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    input_ids_rows, labels_rows, mask_rows = [], [], []
    for concat, flag in zip(concats, flags):
        pad = max_len - len(concat)
        padded = concat + [pad_id] * pad
        padded_flags = flag + [False] * pad
        # Pad rows to max length BEFORE shifting/slicing: the dropped final
        # token is a pad token for short rows (reference fixture semantics).
        input_ids_rows.append(padded[:-1])
        labels_rows.append(padded[1:])
        mask_rows.append(padded_flags[1:])

    return {
        "input_ids": torch.tensor(input_ids_rows, dtype=torch.long),
        "labels": torch.tensor(labels_rows, dtype=torch.long),
        "response_mask": torch.tensor(mask_rows, dtype=torch.bool),
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Conditional log-probs of the labels under the model.

    ``labels`` are already shifted input_ids; no further shifting happens
    here. ``log_probs[:, t] = log_softmax(logits[:, t])[labels[:, t]]``.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, seq_len).
        labels: torch.Tensor of shape (batch_size, seq_len), shifted input_ids.
        return_token_entropy: bool, also return per-position entropy of the
            next-token predictive distribution.

    Returns:
        dict[str, torch.Tensor] with key "log_probs" and, if
        ``return_token_entropy``, key "token_entropy" (both (batch_size, seq_len)).
    """
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    output: dict[str, Tensor] = {"log_probs": target_log_probs}
    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)
    return output


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute one microbatch's SFT loss and backprop it (no grad zeroing).

    The SFT loss sums each example's response-token log-probs, normalizes
    per example by ``normalize_constant`` (Dr. GRPO-style constant
    normalization, no token-count division), averages over the batch, and
    divides by the gradient accumulation factor so accumulated microbatch
    gradients average to the full-batch gradient.
    The caller zeroes grads between optimizer steps; backward is called
    exactly once here, so gradients accumulate across calls.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, seq_len).
        response_mask: torch.Tensor of shape (batch_size, seq_len), bool.
        gradient_accumulation_steps: int, number of accumulation steps.
        normalize_constant: float | None, constant to normalize the summed
            log-probs by (None is treated as 1.0).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: scalar loss and empty
            metadata (nothing to log for plain SFT).
    """
    constant = 1.0 if normalize_constant is None else normalize_constant
    per_example = masked_normalize(
        policy_log_probs,
        response_mask,
        dim=1,
        normalize_constant=constant,
    )
    loss = -per_example.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {}


def log_generations(
    generations: list[str],
    rewards: list[float] | None = None,
    max_examples: int = 4,
    max_chars: int = 300,
) -> None:
    """Log sample decoded generations for eyeballing training progress.

    Generation quality is the ground truth of alignment training: loss curves
    hide reward hacking and format collapse, a handful of raw samples reveal
    both. Truncated so logs stay readable.

    Args:
        generations: list[str], decoded generation strings.
        rewards: list[float] | None, optional reward per generation, logged
            alongside the text.
        max_examples: int, at most this many examples are logged.
        max_chars: int, each generation is truncated to this length.
    """
    if not generations:
        logger.warning("log_generations called with no generations to log")
        return
    if rewards is not None and len(rewards) != len(generations):
        raise ValueError(
            f"rewards length {len(rewards)} does not match generations length {len(generations)}"
        )
    count = min(max_examples, len(generations))
    logger.info("Sampled %d of %d generations:", count, len(generations))
    for i in range(count):
        sample = generations[i][:max_chars]
        if rewards is not None:
            logger.info("  [%d] reward=%.4f | %r", i, rewards[i], sample)
        else:
            logger.info("  [%d] %r", i, sample)
