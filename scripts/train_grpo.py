"""GRPO train loop on MATH (handout Problem: grpo_train_loop + Section 8 experiments).

Implements Algorithm 3: vLLM rollouts at ``--group-size`` per prompt, rewards
from the r1_zero (or question_only) grader, group-normalized advantages, and
gradient updates via the tested loss primitives. The default hyperparameters
are the handout's on-policy block (pp. 27-28); flags cover every graded
ablation:

- ``--loss-type``: no_baseline | reinforce_with_baseline | grpo_clip |
  grpo_no_clip (handout Eq. 34: off-policy importance-weighted PG without
  clipping).
- ``--no-std-normalization``: Dr. GRPO advantages (Eq. 31).
- ``--length-norm``: masked_mean vs masked_normalize (constant = max
  generation length, handout p. 30).
- ``--epochs-per-rollout-batch`` / ``--train-batch-size``: on- vs off-policy
  updates; off-policy consumes old_log_probs computed once per rollout batch
  under ``torch.inference_mode``.
- ``--prompt-template``: r1_zero vs question_only (rewards follow the
  template, handout p. 33).
- ``--leaderboard``: enforces the p. 33-34 evaluation constraints (full
  validation set, r1_zero prompt + r1_zero_reward_fn, temperature 1.0,
  max tokens 1024) and logs wall-clock time for the accuracy-vs-time plot.

Example (cluster, 2 GPUs — vLLM on cuda:1, policy on cuda:0):
    uv run python scripts/train_grpo.py \
        --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
        --train-path /data/a5-alignment/MATH/train.jsonl \
        --output-dir outputs/grpo \
        --vllm-device cuda:1
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

import torch
import typer

from cs336_alignment.common import masked_mean, masked_normalize
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
    grpo_microbatch_train_step,
)
from cs336_alignment.sft import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.training import (
    MATH_TRAIN_PATH,
    MATH_VALIDATION_PATH,
    QWEN_2_5_MATH_1_5B_PATH,
    field,
    init_wandb,
    load_jsonl,
    wandb_log,
)
from cs336_alignment.vllm_utils import evaluate_vllm, init_vllm, load_policy_into_vllm_instance

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts"
# Handout p. 33: the question_only prompt ablation also swaps the reward fn,
# for both training and validation.
PROMPT_SPECS = {
    "r1_zero": ("r1_zero.prompt", r1_zero_reward_fn),
    "question_only": ("question_only.prompt", question_only_reward_fn),
}

app = typer.Typer(add_completion=False)


def compute_grpo_no_clip_loss(
    policy_log_probs,
    old_log_probs,
    advantages,
):
    """Unclipped off-policy GRPO loss (handout Eq. 34, negated objective).

    per-token loss = -exp(log p_theta(o_t) - log p_theta_old(o_t)) * A.
    Requires old_log_probs, hence the off-policy setting.
    """
    adv = advantages.view(-1, 1)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    return -ratio * adv


def validate_config(
    train_batch_size: int,
    gradient_accumulation_steps: int,
    rollout_batch_size: int,
    group_size: int,
    loss_type: str,
    length_norm: str,
    epochs_per_rollout_batch: int,
) -> dict[str, int]:
    """Enforce the handout pp. 27-28 sanity asserts and derive loop quantities."""
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    if loss_type not in ("no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"):
        raise ValueError(f"Unknown loss_type: {loss_type!r}")
    if length_norm not in ("masked_mean", "masked_normalize"):
        raise ValueError(f"Unknown length_norm: {length_norm!r}")
    if epochs_per_rollout_batch < 1:
        raise ValueError("epochs_per_rollout_batch must be >= 1")
    if loss_type in ("grpo_clip", "grpo_no_clip") and epochs_per_rollout_batch == 1 and train_batch_size == rollout_batch_size:
        logger.warning(
            "%s on a fully on-policy setting: the ratio is 1 and clipping never "
            "activates (handout p. 28 recommends clipping only off-policy)",
            loss_type,
        )
    return {
        "micro_train_batch_size": micro_train_batch_size,
        "n_prompts_per_rollout_batch": n_prompts_per_rollout_batch,
        "n_microbatches_per_rollout_batch": n_microbatches_per_rollout_batch,
    }


def _rollout_tensors(
    tokenizer,
    prompts: list[str],
    responses: list[str],
    device: str,
):
    """Featurize (prompt, rollout) pairs with the tested SFT tokenization.

    Returns padded (input_ids, labels, response_mask) aligned so that
    response tokens are predicted at masked positions.
    """
    batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
    return (
        batch["input_ids"].to(device),
        batch["labels"].to(device),
        batch["response_mask"].to(device),
    )


def _forward_log_probs(
    policy,
    input_ids,
    labels,
    response_mask,
    microbatch_size: int,
    need_old_log_probs: bool,
) -> dict[str, object]:
    """inference-mode pass over the rollout batch for old_log_probs + entropy.

    Old log-probabilities are computed once per rollout batch and reused for
    every epoch of gradient steps (handout p. 29); they are never
    differentiated through.
    """
    chunk_entropy: list = []
    old_chunks: list = []
    with torch.inference_mode():
        for start in range(0, input_ids.shape[0], microbatch_size):
            sl = slice(start, start + microbatch_size)
            out = get_response_log_probs(
                policy, input_ids[sl], labels[sl], return_token_entropy=True
            )
            old_chunks.append(out["log_probs"])
            chunk_entropy.append(
                float(masked_mean(out["token_entropy"], response_mask[sl], dim=None))
            )
    result: dict = {"token_entropy": sum(chunk_entropy) / len(chunk_entropy)}
    if need_old_log_probs:
        result["old_log_probs"] = torch.cat(old_chunks, dim=0)
    return result


def _grpo_microbatch_backward(
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards,
    advantages,
    old_log_probs,
    cliprange: float,
    length_norm: str,
    max_gen_len: int,
):
    """One GRPO microbatch forward loss + backward, honoring the ablations."""
    if loss_type == "grpo_no_clip":
        per_token, metadata = compute_grpo_no_clip_loss(
            policy_log_probs, old_log_probs, advantages
        ), {}
    elif length_norm == "masked_mean":
        # The tested composition (core grpo_microbatch_train_step) already
        # masks, means, scales, and backprops.
        loss, metadata = grpo_microbatch_train_step(
            policy_log_probs,
            response_mask,
            gradient_accumulation_steps,
            loss_type,
            raw_rewards=raw_rewards,
            advantages=advantages,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        return loss, metadata
    else:
        per_token, metadata = compute_policy_gradient_loss(
            policy_log_probs,
            loss_type,
            raw_rewards=raw_rewards,
            advantages=advantages,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    # Length normalization ablation (handout p. 30): per-example mean over
    # response tokens, or token-sum divided by the max generation length.
    if length_norm == "masked_mean":
        per_example = masked_mean(per_token, response_mask, dim=1)
    else:
        per_example = masked_normalize(
            per_token, response_mask, dim=1, normalize_constant=float(max_gen_len)
        )
    loss = per_example.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, metadata


@app.command()
def main(
    model_path: str = typer.Option(QWEN_2_5_MATH_1_5B_PATH, help="Initial policy checkpoint."),
    train_path: str = typer.Option(MATH_TRAIN_PATH, help="MATH train JSONL."),
    val_path: str = typer.Option(MATH_VALIDATION_PATH, help="MATH validation JSONL."),
    output_dir: str = typer.Option("outputs/grpo", help="Directory for checkpoints and metrics."),
    n_grpo_steps: int = typer.Option(200, help="Number of rollout batches (handout: 200)."),
    learning_rate: float = typer.Option(1e-5, help="AdamW learning rate (constant; handout: 1e-5)."),
    advantage_eps: float = typer.Option(1e-6, help="Epsilon inside the std denominator (handout: 1e-6)."),
    rollout_batch_size: int = typer.Option(256, help="Rollouts per training step (handout: 256)."),
    group_size: int = typer.Option(8, help="Rollouts per question (handout: 8)."),
    sampling_temperature: float = typer.Option(1.0, help="Rollout sampling temperature (handout: 1.0)."),
    sampling_min_tokens: int = typer.Option(4, help="Min generation length (handout tip)."),
    sampling_max_tokens: int = typer.Option(1024, help="Max generation length (handout: 1024)."),
    epochs_per_rollout_batch: int = typer.Option(1, help="Gradient epochs per rollout batch (1 = on-policy)."),
    train_batch_size: int = typer.Option(256, help="Examples per optimizer step (handout: 256 on-policy)."),
    gradient_accumulation_steps: int = typer.Option(128, help="Microbatches per optimizer step (handout: 128)."),
    loss_type: str = typer.Option(
        "reinforce_with_baseline",
        help="no_baseline | reinforce_with_baseline | grpo_clip | grpo_no_clip (Eq. 34).",
    ),
    use_std_normalization: bool = typer.Option(
        True, help="--no-std-normalization gives Dr. GRPO advantages (Eq. 31)."
    ),
    length_norm: str = typer.Option(
        "masked_mean", help="masked_mean or masked_normalize (constant = max generation length)."
    ),
    cliprange: float = typer.Option(0.2, help="Clip epsilon for grpo_clip."),
    max_grad_norm: float = typer.Option(1.0, help="Gradient clipping value (handout: 1.0)."),
    prompt_template: str = typer.Option(
        "r1_zero", help="Training/validation prompt: r1_zero or question_only (rewards follow)."
    ),
    leaderboard: bool = typer.Option(
        False, help="Leaderboard constraints: full validation set, r1_zero eval prompt/reward, "
        "temperature 1.0, max tokens 1024 (handout pp. 33-34)."
    ),
    eval_every: int = typer.Option(10, help="Validation evals every N GRPO steps (handout: 5-10)."),
    val_size: Optional[int] = typer.Option(
        1024, help="Validation examples per eval (handout: >=1024; leaderboard uses all)."
    ),
    attn_implementation: Optional[str] = typer.Option(
        None, help="HF attention implementation; pass 'flash_attention_2' to opt in (requires flash-attn)."
    ),
    device: str = typer.Option("cuda:0", help="Device for the training policy."),
    vllm_device: str = typer.Option("cuda:0", help="Device for the vLLM rollout engine."),
    gpu_memory_utilization: float = typer.Option(0.85, help="Fraction of GPU memory for vLLM."),
    enforce_eager: bool = typer.Option(True, help="Skip vLLM CUDA graph capture."),
    seed: int = typer.Option(0, help="Random seed."),
    wandb_mode: str = typer.Option("off", help="wandb mode: off | offline | online."),
    wandb_project: str = typer.Option("a5-grpo", help="wandb project name."),
) -> None:
    """Run the GRPO train loop (Algorithm 3) with all graded ablations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import SamplingParams  # lazy: vLLM only loads when the script runs

    torch.manual_seed(seed)
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    derived = validate_config(
        train_batch_size,
        gradient_accumulation_steps,
        rollout_batch_size,
        group_size,
        loss_type,
        length_norm,
        epochs_per_rollout_batch,
    )
    micro_train_batch_size = derived["micro_train_batch_size"]
    n_prompts_per_rollout_batch = derived["n_prompts_per_rollout_batch"]
    if prompt_template not in PROMPT_SPECS:
        raise ValueError(f"Unknown prompt_template: {prompt_template!r}")

    # Prompt choice ablation (handout p. 33): template and reward fn pair up.
    template_name, train_reward_fn = PROMPT_SPECS[prompt_template]
    template_text = (PROMPTS_DIR / template_name).read_text()

    rows = load_jsonl(train_path)
    questions = [field(r, ("problem", "question"), "MATH train") for r in rows]
    ground_truths = [str(field(r, ("answer", "expected_answer", "ground_truth"), "MATH train")) for r in rows]

    val_rows = load_jsonl(val_path)
    if leaderboard:
        # Leaderboard constraints (handout pp. 33-34): all 5K validation
        # examples, r1_zero prompt + r1_zero_reward_fn, temp 1.0, max 1024.
        val_template_text = (PROMPTS_DIR / "r1_zero.prompt").read_text()
        val_reward_fn = r1_zero_reward_fn
        val_max_tokens = 1024
        val_temperature = 1.0
    else:
        val_template_text = template_text
        val_reward_fn = train_reward_fn
        val_max_tokens = sampling_max_tokens
        val_temperature = 1.0
    val_rows = val_rows if val_size is None else val_rows[:val_size]
    val_prompts = [
        val_template_text.format(question=field(r, ("problem", "question"), "MATH validation")) for r in val_rows
    ]
    val_ground_truths = [
        str(field(r, ("answer", "expected_answer", "ground_truth"), "MATH validation")) for r in val_rows
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="bfloat16",
        attn_implementation=attn_implementation,
    ).to(device)
    policy.train()

    init_wandb(wandb_mode, wandb_project, {
        "model_path": model_path, "n_grpo_steps": n_grpo_steps, "learning_rate": learning_rate,
        "rollout_batch_size": rollout_batch_size, "group_size": group_size,
        "epochs_per_rollout_batch": epochs_per_rollout_batch, "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps, "loss_type": loss_type,
        "use_std_normalization": use_std_normalization, "length_norm": length_norm,
        "cliprange": cliprange, "prompt_template": prompt_template, "leaderboard": leaderboard, "seed": seed,
    })

    vllm_model = init_vllm(
        model_path,
        device=vllm_device,
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        n=group_size,
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    eval_sampling_params = SamplingParams(
        temperature=val_temperature,
        top_p=1.0,
        max_tokens=val_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95)
    )
    metrics_path = out / "metrics.jsonl"
    need_old_log_probs = loss_type in ("grpo_clip", "grpo_no_clip")
    start_time = time.time()
    optimizer_updates = 0

    for grpo_step in range(1, n_grpo_steps + 1):
        step_wall = time.time()

        # Sample a batch of questions (Algorithm 3, line 3).
        prompt_indices = random.sample(
            range(len(questions)),
            min(n_prompts_per_rollout_batch, len(questions)),
        )
        batch_ground_truths = [ground_truths[i] for i in prompt_indices]
        prompts = [template_text.format(question=questions[i]) for i in prompt_indices]

        # Roll G outputs per question from the current policy (lines 4-5).
        outputs = vllm_model.generate(prompts, sampling_params)
        rollout_responses: list[str] = []
        repeated_ground_truths: list[str] = []
        for output, ground_truth in zip(outputs, batch_ground_truths):
            for completion in output.outputs:
                rollout_responses.append(completion.text)
                repeated_ground_truths.append(ground_truth)

        # Rewards and group-normalized advantages (lines 6-7).
        reward_rows = [train_reward_fn(r, gt) for r, gt in zip(rollout_responses, repeated_ground_truths)]
        raw_reward_stats = {
            f"reward_{k}_mean": sum(row[k] for row in reward_rows) / len(reward_rows)
            for k in ("reward", "format_reward", "answer_reward")
        }
        advantages, raw_rewards, advantage_meta = compute_group_normalized_rewards(
            train_reward_fn,
            rollout_responses,
            repeated_ground_truths,
            group_size,
            advantage_eps,
            use_std_normalization,
        )
        advantages = advantages.to(device)
        raw_rewards = raw_rewards.to(device)

        # Featurize the rollout batch once; old log-probs once per batch (p. 29).
        input_ids, labels, response_mask = _rollout_tensors(
            tokenizer, [p for p in prompts for _ in range(group_size)], rollout_responses, device
        )
        forward_stats = _forward_log_probs(
            policy, input_ids, labels, response_mask, micro_train_batch_size, need_old_log_probs
        )
        old_log_probs = forward_stats.get("old_log_probs")
        token_entropy = forward_stats["token_entropy"]

        # Gradient epochs over this rollout batch (line 8; 1 = on-policy).
        rollout_batch_size_actual = input_ids.shape[0]
        for epoch in range(epochs_per_rollout_batch):
            order = torch.randperm(rollout_batch_size_actual) if epochs_per_rollout_batch > 1 else torch.arange(rollout_batch_size_actual)
            for batch_start in range(0, rollout_batch_size_actual, train_batch_size):
                batch_idx = order[batch_start : batch_start + train_batch_size]
                optimizer.zero_grad()
                for mb_start in range(0, len(batch_idx), micro_train_batch_size):
                    mb_idx = batch_idx[mb_start : mb_start + micro_train_batch_size]
                    policy_log_probs = get_response_log_probs(
                        policy, input_ids[mb_idx], labels[mb_idx]
                    )["log_probs"]
                    loss, meta = _grpo_microbatch_backward(
                        policy_log_probs,
                        response_mask[mb_idx],
                        gradient_accumulation_steps,
                        loss_type,
                        raw_rewards=raw_rewards[mb_idx].view(-1, 1),
                        advantages=advantages[mb_idx].view(-1, 1),
                        old_log_probs=old_log_probs[mb_idx] if old_log_probs is not None else None,
                        cliprange=cliprange,
                        length_norm=length_norm,
                        max_gen_len=sampling_max_tokens,
                    )
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                optimizer_updates += 1
                clip_fraction = float("nan")
                was_clipped = meta.get("was_clipped") if isinstance(meta, dict) else None
                if was_clipped is not None and response_mask[mb_idx].any():
                    clip_fraction = float(masked_mean(
                        was_clipped.float(), response_mask[mb_idx], dim=None
                    ))
                logger.info(
                    "grpo step %d epoch %d update %d | loss %.5f | grad_norm %.3f | entropy %.4f | clip_frac %.3f",
                    grpo_step, epoch, optimizer_updates, float(loss.detach()),
                    float(grad_norm), token_entropy, clip_fraction,
                )
                wandb_log(optimizer_updates, {
                    "train/loss": float(loss.detach()),
                    "train/grad_norm": float(grad_norm),
                    "train/token_entropy": token_entropy,
                    "train/clip_fraction": clip_fraction,
                    **{f"train/{k}": v for k, v in raw_reward_stats.items()},
                    **{f"train/{k}": v for k, v in advantage_meta.items()},
                }, wandb_mode)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({
                        "grpo_step": grpo_step, "epoch": epoch, "update": optimizer_updates,
                        "split": "train", "loss": float(loss.detach()),
                        "grad_norm": float(grad_norm), "token_entropy": token_entropy,
                        "clip_fraction": clip_fraction, "wall_clock_s": time.time() - start_time,
                        **raw_reward_stats, **advantage_meta,
                    }) + "\n")

        # Routine validation (handout p. 28: every 5-10 steps, >=1024 examples).
        if eval_every and (grpo_step % eval_every == 0 or grpo_step == n_grpo_steps):
            policy.eval()
            load_policy_into_vllm_instance(policy, vllm_model)
            results = evaluate_vllm(
                vllm_model, val_reward_fn, val_prompts, eval_sampling_params,
                ground_truths=val_ground_truths,
                output_path=str(out / f"val_generations_step{grpo_step}.jsonl"),
            )
            policy.train()
            logger.info(
                "grpo step %d | validation accuracy %.4f | wall clock %.0fs",
                grpo_step, results["metrics"]["accuracy"], time.time() - start_time,
            )
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "grpo_step": grpo_step, "split": "validation", "wall_clock_s": time.time() - start_time,
                    **results["metrics"],
                }) + "\n")
            wandb_log(
                grpo_step,
                {f"validation/{k}": v for k, v in results["metrics"].items()},
                wandb_mode,
            )
            checkpoint_dir = out / f"step_{grpo_step}"
            policy.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

        logger.info(
            "grpo step %d done in %.1fs (rollouts %d, prompts %d)",
            grpo_step, time.time() - step_wall, rollout_batch_size_actual, len(prompt_indices),
        )

    final_dir = out / "final"
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    typer.echo(f"GRPO training complete: {optimizer_updates} updates, final checkpoint at {final_dir}")


if __name__ == "__main__":
    app()
