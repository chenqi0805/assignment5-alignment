"""Expert iteration on MATH (handout Algorithm 2, Problem: expert_iteration_experiment).

Each EI step: sample a batch of questions, draw G rollouts per question from
the current policy with vLLM, keep generations whose r1_zero reward is 1.0 as
a synthetic SFT dataset, run SFT (Algorithm 1) on it from the current policy,
and log response entropy + validation accuracy. vLLM terminates generations
at ``</answer>`` and uses ``min_tokens`` to disallow empty responses (handout
tips).

Example (cluster):
    uv run python scripts/expert_iteration.py \
        --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
        --train-path /data/a5-alignment/MATH/train.jsonl \
        --output-dir outputs/expert_iteration \
        --vllm-device cuda:1
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import typer

from cs336_alignment.common import masked_mean
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.training import (
    MATH_TRAIN_PATH,
    MATH_VALIDATION_PATH,
    QWEN_2_5_MATH_1_5B_PATH,
    build_lr_scheduler,
    field,
    init_wandb,
    load_jsonl,
    sft_optimization_step,
    wandb_log,
)
from cs336_alignment.vllm_utils import evaluate_vllm, init_vllm, load_policy_into_vllm_instance

logger = logging.getLogger(__name__)

R1_ZERO_PROMPT_PATH = Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"

app = typer.Typer(add_completion=False)


def _mean_response_entropy(
    policy,
    tokenizer,
    pairs: list[tuple[str, str]],
    max_pairs: int,
) -> float:
    """Mean token entropy of generated responses under the current policy."""
    if not pairs:
        return 0.0
    pairs = pairs[:max_pairs]
    batch = tokenize_prompt_and_output([p for p, _ in pairs], [r for _, r in pairs], tokenizer)
    device = next(policy.parameters()).device
    with torch.inference_mode():
        result = get_response_log_probs(
            policy,
            batch["input_ids"].to(device),
            batch["labels"].to(device),
            return_token_entropy=True,
        )
        entropy = masked_mean(result["token_entropy"], batch["response_mask"].to(device), dim=None)
    return float(entropy)


@app.command()
def main(
    model_path: str = typer.Option(QWEN_2_5_MATH_1_5B_PATH, help="Initial policy checkpoint."),
    train_path: str = typer.Option(MATH_TRAIN_PATH, help="MATH train JSONL (questions to sample)."),
    val_path: str = typer.Option(MATH_VALIDATION_PATH, help="MATH validation JSONL for accuracy evals."),
    output_dir: str = typer.Option("outputs/expert_iteration", help="Directory for checkpoints and metrics."),
    n_ei_steps: int = typer.Option(5, help="Number of expert-iteration steps (handout: 5)."),
    batch_size: int = typer.Option(512, help="Questions per EI step (handout ablation: 512/1024/2048)."),
    num_rollouts: int = typer.Option(4, help="Rollouts G per question (handout: try >= 2 values)."),
    sft_epochs: int = typer.Option(1, help="SFT epochs over the filtered dataset per EI step (ablation)."),
    sft_batch_size: int = typer.Option(64, help="Examples per SFT optimizer step."),
    sft_gradient_accumulation_steps: int = typer.Option(4, help="SFT microbatches per optimizer step."),
    learning_rate: float = typer.Option(2e-5, help="SFT AdamW learning rate (fresh optimizer per EI step)."),
    warmup_ratio: float = typer.Option(0.03, help="SFT linear warmup fraction."),
    min_lr_ratio: float = typer.Option(0.0, help="SFT final LR ratio (cosine decay)."),
    max_grad_norm: float = typer.Option(1.0, help="Gradient clipping value (handout: 1.0)."),
    sampling_temperature: float = typer.Option(1.0, help="Rollout sampling temperature."),
    sampling_min_tokens: int = typer.Option(4, help="Min generation length (handout tip)."),
    sampling_max_tokens: int = typer.Option(1024, help="Max generation length."),
    val_size: int = typer.Option(1024, help="Validation examples per eval (handout: >=1024)."),
    entropy_examples: int = typer.Option(64, help="Pairs subsampled for the entropy metric."),
    attn_implementation: Optional[str] = typer.Option(
        None, help="HF attention implementation; pass 'flash_attention_2' to opt in (requires flash-attn)."
    ),
    device: str = typer.Option("cuda:0", help="Device for the training policy."),
    vllm_device: str = typer.Option("cuda:0", help="Device for the vLLM rollout engine."),
    gpu_memory_utilization: float = typer.Option(0.85, help="Fraction of GPU memory for vLLM."),
    enforce_eager: bool = typer.Option(True, help="Skip vLLM CUDA graph capture."),
    seed: int = typer.Option(0, help="Random seed."),
    wandb_mode: str = typer.Option("off", help="wandb mode: off | offline | online."),
    wandb_project: str = typer.Option("a5-expert-iteration", help="wandb project name."),
) -> None:
    """Run Algorithm 2 (expert iteration) on MATH."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import SamplingParams  # lazy: vLLM only loads when the script runs

    torch.manual_seed(seed)
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(train_path)
    questions = [field(r, ("problem", "question"), "MATH train") for r in rows]
    ground_truths = [str(field(r, ("answer", "expected_answer", "ground_truth"), "MATH train")) for r in rows]
    template_text = R1_ZERO_PROMPT_PATH.read_text()

    val_rows = load_jsonl(val_path)[:val_size]
    val_prompts = [template_text.format(question=field(r, ("problem", "question"), "MATH validation")) for r in val_rows]
    val_ground_truths = [
        str(field(r, ("answer", "expected_answer", "ground_truth"), "MATH validation")) for r in val_rows
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="bfloat16",
        attn_implementation=attn_implementation,
    ).to(device)

    init_wandb(wandb_mode, wandb_project, {
        "model_path": model_path, "n_ei_steps": n_ei_steps, "batch_size": batch_size,
        "num_rollouts": num_rollouts, "sft_epochs": sft_epochs, "learning_rate": learning_rate, "seed": seed,
    })

    vllm_model = init_vllm(
        model_path,
        device=vllm_device,
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        n=num_rollouts,
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    metrics_path = out / "metrics.jsonl"
    microbatch_size = sft_batch_size // sft_gradient_accumulation_steps
    if sft_batch_size % sft_gradient_accumulation_steps != 0:
        raise ValueError("sft_batch_size must be divisible by sft_gradient_accumulation_steps")

    for ei_step in range(1, n_ei_steps + 1):
        # Sample a fresh batch of questions for this EI step (Algorithm 2, line 3).
        batch_indices = random.sample(range(len(questions)), min(batch_size, len(questions)))
        batch_questions = [questions[i] for i in batch_indices]
        batch_ground_truths = [ground_truths[i] for i in batch_indices]
        prompts = [template_text.format(question=q) for q in batch_questions]

        # Roll G generations per question from the current policy (lines 4-5).
        outputs = vllm_model.generate(prompts, sampling_params)
        rollout_responses: list[str] = []
        rollout_ground_truths: list[str] = []
        for output, ground_truth in zip(outputs, batch_ground_truths):
            for completion in output.outputs:
                rollout_responses.append(completion.text)
                rollout_ground_truths.append(ground_truth)

        # Keep only correct generations as the synthetic SFT set (lines 6-7).
        expert_pairs: list[tuple[str, str]] = []
        total_reward, num_generated = 0.0, len(rollout_responses)
        for prompt, response, ground_truth in zip(
            [p for p in prompts for _ in range(num_rollouts)], rollout_responses, rollout_ground_truths
        ):
            score = r1_zero_reward_fn(response, ground_truth)
            total_reward += score["reward"]
            if score["reward"] == 1.0:
                expert_pairs.append((prompt, response))
        if not expert_pairs:
            logger.warning("EI step %d: no correct generations; skipping SFT", ei_step)

        entropy = _mean_response_entropy(policy, tokenizer, expert_pairs, entropy_examples)
        logger.info(
            "EI step %d | %d/%d rollouts correct (reward mean %.4f) | response entropy %.4f",
            ei_step, len(expert_pairs), num_generated, total_reward / max(1, num_generated), entropy,
        )
        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "step": ei_step,
                "num_expert_pairs": len(expert_pairs),
                "num_rollouts": num_generated,
                "rollout_reward_mean": total_reward / max(1, num_generated),
                "response_entropy": entropy,
            }) + "\n")
        wandb_log(ei_step, {
            "ei/num_expert_pairs": len(expert_pairs),
            "ei/rollout_reward_mean": total_reward / max(1, num_generated),
            "ei/response_entropy": entropy,
        }, wandb_mode)

        # SFT from the current policy on the filtered data (line 8).
        if expert_pairs:
            policy.train()
            num_updates = sft_epochs * -(-len(expert_pairs) // sft_batch_size)
            optimizer = torch.optim.AdamW(
                policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95)
            )
            scheduler = build_lr_scheduler(optimizer, int(warmup_ratio * num_updates), int(num_updates), min_lr_ratio)
            for epoch in range(sft_epochs):
                random.shuffle(expert_pairs)
                for batch_start in range(0, len(expert_pairs), sft_batch_size):
                    batch = expert_pairs[batch_start : batch_start + sft_batch_size]
                    microbatches = [
                        ([p for p, _ in mb], [r for _, r in mb])
                        for mb in (batch[mb : mb + microbatch_size] for mb in range(0, len(batch), microbatch_size))
                    ]
                    stats = sft_optimization_step(
                        policy, optimizer, microbatches, tokenizer, sft_gradient_accumulation_steps,
                        max_grad_norm=max_grad_norm, scheduler=scheduler,
                    )
                    wandb_log(
                        ei_step * 1000 + epoch * max(1, num_updates // max(1, sft_epochs)) + batch_start // sft_batch_size,
                        {"ei_sft/loss": stats["loss"], "ei_sft/grad_norm": stats["grad_norm"]},
                        wandb_mode,
                    )
            policy.eval()

        # Validation accuracy after this EI step's retraining.
        load_policy_into_vllm_instance(policy, vllm_model)
        results = evaluate_vllm(
            vllm_model, r1_zero_reward_fn, val_prompts, eval_sampling_params,
            ground_truths=val_ground_truths,
            output_path=str(out / f"val_generations_ei_step{ei_step}.jsonl"),
        )
        logger.info("EI step %d | validation accuracy %.4f", ei_step, results["metrics"]["accuracy"])
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": ei_step, "split": "validation", **results["metrics"]}) + "\n")
        wandb_log(ei_step, {f"validation/{k}": v for k, v in results["metrics"].items()}, wandb_mode)

        checkpoint_dir = out / f"step_{ei_step}"
        policy.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info("Saved EI step %d checkpoint to %s", ei_step, checkpoint_dir)

    typer.echo(f"Expert iteration complete: {n_ei_steps} steps, checkpoints under {out}")


if __name__ == "__main__":
    app()
