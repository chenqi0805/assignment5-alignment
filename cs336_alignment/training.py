"""Training-loop helpers shared by the experiment scripts (additive module).

Nothing here touches the loss primitives in ``common``/``sft``/``grpo``/
``dpo`` — this is the boring glue: cluster path constants, JSONL loading,
learning-rate schedules, and optional wandb setup.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
from xopen import xopen

logger = logging.getLogger(__name__)

# Cluster mounts (Together cluster paths from the handout). Scripts accept
# overrides for every one of these; the defaults make the README's cluster
# commands copy-pasteable.
CLUSTER_ROOT = "/data/a5-alignment"
QWEN_2_5_MATH_1_5B_PATH = f"{CLUSTER_ROOT}/models/Qwen2.5-Math-1.5B"
LLAMA_3_1_8B_PATH = f"{CLUSTER_ROOT}/models/Llama-3.1-8B"
LLAMA_3_3_70B_INSTRUCT_PATH = f"{CLUSTER_ROOT}/models/Llama-3.3-70B-Instruct"
MATH_TRAIN_PATH = f"{CLUSTER_ROOT}/MATH/train.jsonl"
MATH_VALIDATION_PATH = f"{CLUSTER_ROOT}/MATH/validation.jsonl"
MATH_SFT_PATH = f"{CLUSTER_ROOT}/MATH/sft.jsonl"
ULTRACHAT_TRAIN_PATH = f"{CLUSTER_ROOT}/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz"
ULTRACHAT_TEST_PATH = f"{CLUSTER_ROOT}/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz"
HH_DIR = f"{CLUSTER_ROOT}/hh"
HH_FILES = (
    "harmless-base.jsonl.gz",
    "helpful-base.jsonl.gz",
    "helpful-online.jsonl.gz",
    "helpful-rejection-sampled.jsonl.gz",
)


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL (or gzipped JSONL) file into a list of dicts.

    Uses ``xopen`` so ``.jsonl.gz`` and plain ``.jsonl`` share one code path.
    """
    rows = []
    with xopen(str(path)) as f:
        for line_number, line in enumerate(f):
            if line.strip():
                rows.append(json.loads(line))
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


def field(row: dict, names: tuple[str, ...], context: str) -> str:
    """Return the first present key among ``names`` from ``row``.

    The cluster's MATH/HH/ultrachat files are not readable here, and public
    mirrors of the assignment data disagree on key names (``problem`` vs
    ``question``, ``answer`` vs ``expected_answer``); accepting aliases makes
    the scripts robust to that schema choice instead of guessing one.
    """
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    raise KeyError(f"None of {names} found in {context} row with keys {sorted(row)}")


def linear_warmup_cosine(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0) -> float:
    """LR multiplier for linear warmup followed by cosine decay.

    Args:
        step: int, current optimizer step (0-indexed).
        warmup_steps: int, number of linear warmup steps from 0 to 1.
        total_steps: int, total number of optimizer steps.
        min_lr_ratio: float, final LR as a fraction of the base LR.

    Returns:
        float multiplier in [min_lr_ratio, 1.0].
    """
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """LambdaLR over ``linear_warmup_cosine`` (evaluated per optimizer step)."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: linear_warmup_cosine(step, warmup_steps, total_steps, min_lr_ratio),
    )


def sft_backward_microbatch(
    model: torch.nn.Module,
    prompts: list[str],
    responses: list[str],
    tokenizer,
    gradient_accumulation_steps: int,
    loss_normalization: str = "token_mean",
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """One response-only SFT microbatch: forward + backward (no grad zeroing).

    Tokenization, shifting, and the response mask come from the tested
    ``tokenize_prompt_and_output`` primitive; log-probs from
    ``get_response_log_probs``.

    Args:
        model: torch.nn.Module, the policy being trained.
        prompts: list[str], fully formatted prompt strings for the microbatch.
        responses: list[str], target response strings (EOS appended here).
        tokenizer: PreTrainedTokenizerBase.
        gradient_accumulation_steps: int, divides the loss so accumulated
            microbatch gradients average to the full-batch gradient.
        loss_normalization: str, "token_mean" (mean response-token log-prob,
            standard SFT) or "per_example_sum" (the handout's
            ``sft_microbatch_train_step`` construction with a constant).
        normalize_constant: float, constant for "per_example_sum".

    Returns:
        torch.Tensor scalar microbatch loss (already divided by the
        gradient-accumulation factor), for logging.
    """
    from .sft import get_response_log_probs, sft_microbatch_train_step, tokenize_prompt_and_output
    from .common import masked_mean

    device = next(model.parameters()).device
    batch = tokenize_prompt_and_output(prompts, [r + tokenizer.eos_token for r in responses], tokenizer)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)

    policy_log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
    if loss_normalization == "token_mean":
        loss = -masked_mean(policy_log_probs, response_mask, dim=None) / gradient_accumulation_steps
        loss.backward()
        return loss
    if loss_normalization == "per_example_sum":
        loss, _ = sft_microbatch_train_step(
            policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant
        )
        return loss
    raise ValueError(f"Unknown loss_normalization: {loss_normalization!r}")


def sft_optimization_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    microbatches: list[tuple[list[str], list[str]]],
    tokenizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float = 1.0,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    loss_normalization: str = "token_mean",
    normalize_constant: float = 1.0,
) -> dict[str, float]:
    """One optimizer step over accumulated SFT microbatches.

    Args:
        microbatches: list of (prompts, responses) pairs; together they cover
            the full train batch, and their count equals
            ``gradient_accumulation_steps``.

    Returns:
        dict with the mean microbatch loss and the gradient norm.
    """
    optimizer.zero_grad()
    loss_sum = 0.0
    for prompts, responses in microbatches:
        loss = sft_backward_microbatch(
            model,
            prompts,
            responses,
            tokenizer,
            gradient_accumulation_steps,
            loss_normalization=loss_normalization,
            normalize_constant=normalize_constant,
        )
        loss_sum += float(loss.detach())
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {"loss": loss_sum / len(microbatches), "grad_norm": float(grad_norm)}


def init_wandb(mode: str, project: str, config: dict) -> None:
    """Initialize wandb unless disabled; offline mode never touches the network.

    Args:
        mode: str, one of "off", "offline", "online".
        project: str, wandb project name.
        config: dict, run config to log.
    """
    if mode == "off":
        return
    import wandb  # lazy: wandb is never required for import

    wandb.init(project=project, mode=mode, config=config)


def wandb_log(step: int, values: dict[str, float], mode: str) -> None:
    """Log metrics to wandb if enabled; silently skip otherwise."""
    if mode == "off":
        return
    import wandb

    if wandb.run is not None:
        wandb.log(values, step=step)
