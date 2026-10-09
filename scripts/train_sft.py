"""Supervised finetuning on MATH reasoning traces (handout Algorithm 1, Problem: sft_experiment).

Trains the Qwen 2.5 Math 1.5B base model on the r1-formatted reasoning SFT
data with a response-only cross-entropy loss (``get_response_log_probs``),
AdamW + linear warmup + cosine decay, and gradient clipping at 1.0. Flags
cover the graded ablations:

- ``--max-examples {128,256,512,1024}``: dataset-size ablation.
- ``--correct-only``: keep only traces whose answer grades correct against
  the gold answer (report the filtered size in the logs).

Validation accuracy (r1_zero prompt + ``r1_zero_reward_fn`` via vLLM) is
logged every ``--eval-every`` optimizer steps to wandb (optional) and a
metrics JSONL; the model and tokenizer are saved to ``--output-dir``.

Example (cluster):
    uv run python scripts/train_sft.py \
        --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
        --data-path /data/a5-alignment/MATH/sft.jsonl \
        --output-dir outputs/sft_full \
        --vllm-device cuda:1
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import typer

from cs336_alignment.drgrpo_grader import grade, extract_answer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.training import (
    MATH_SFT_PATH,
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


def _load_sft_pairs(
    data_path: str,
    max_examples: Optional[int],
    correct_only: bool,
) -> list[tuple[str, str]]:
    """Load (r1_zero prompt, response) pairs, with the graded ablation filters."""
    rows = load_jsonl(data_path)
    questions = [field(row, ("problem", "question"), "MATH SFT") for row in rows]
    responses = [field(row, ("response", "reasoning_trace", "output"), "MATH SFT") for row in rows]

    if correct_only:
        template = R1_ZERO_PROMPT_PATH.read_text()
        kept: list[tuple[str, str]] = []
        for row, question, response in zip(rows, questions, responses):
            gold = str(field(row, ("answer", "expected_answer", "ground_truth"), "MATH SFT"))
            predicted = extract_answer(response)
            if predicted is not None and grade(predicted, gold):
                kept.append((template.format(question=question), response))
        logger.info(
            "Correct-only filter kept %d of %d traces", len(kept), len(questions)
        )
        pairs = kept
    else:
        template = R1_ZERO_PROMPT_PATH.read_text()
        pairs = [(template.format(question=q), r) for q, r in zip(questions, responses)]

    if max_examples is not None:
        pairs = pairs[:max_examples]
    if not pairs:
        raise ValueError(
            f"No SFT pairs after filtering (correct_only={correct_only}, max_examples={max_examples})"
        )
    logger.info("Training on %d SFT pairs", len(pairs))
    return pairs


@app.command()
def main(
    model_path: str = typer.Option(QWEN_2_5_MATH_1_5B_PATH, help="Initial policy checkpoint."),
    data_path: str = typer.Option(MATH_SFT_PATH, help="Reasoning SFT JSONL (question + trace)."),
    val_path: str = typer.Option(MATH_VALIDATION_PATH, help="MATH validation JSONL for accuracy evals."),
    output_dir: str = typer.Option("outputs/sft", help="Directory for checkpoints, metrics, and eval dumps."),
    max_examples: Optional[int] = typer.Option(
        None, help="Cap unique SFT examples (dataset-size ablation: 128/256/512/1024; default: full dataset)."
    ),
    correct_only: bool = typer.Option(
        False, help="Keep only traces whose answer grades correct (handout sft_experiment part 2)."
    ),
    epochs: int = typer.Option(2, help="Passes over the SFT dataset."),
    train_batch_size: int = typer.Option(64, help="Examples per optimizer step."),
    gradient_accumulation_steps: int = typer.Option(4, help="Microbatches per optimizer step."),
    learning_rate: float = typer.Option(2e-5, help="Peak AdamW learning rate."),
    warmup_ratio: float = typer.Option(0.03, help="Fraction of total steps used for linear warmup."),
    min_lr_ratio: float = typer.Option(0.0, help="Final LR as a fraction of the peak (cosine decay)."),
    max_grad_norm: float = typer.Option(1.0, help="Gradient clipping value (handout: 1.0)."),
    loss_normalization: str = typer.Option(
        "token_mean", help="'token_mean' (mean response-token log-prob) or 'per_example_sum' (handout constant form)."
    ),
    normalize_constant: float = typer.Option(1.0, help="Constant for 'per_example_sum' normalization."),
    eval_every: int = typer.Option(50, help="Run validation accuracy every N optimizer steps (0 disables)."),
    val_size: int = typer.Option(1024, help="Number of validation examples per eval (handout: >=1024)."),
    val_max_tokens: int = typer.Option(1024, help="Max generation length for validation."),
    val_temperature: float = typer.Option(1.0, help="Sampling temperature for validation."),
    attn_implementation: Optional[str] = typer.Option(
        None, help="HF attention implementation; pass 'flash_attention_2' to opt in (requires flash-attn)."
    ),
    device: str = typer.Option("cuda:0", help="Device for the training policy."),
    vllm_device: str = typer.Option("cuda:0", help="Device for the vLLM validation engine."),
    gpu_memory_utilization: float = typer.Option(0.85, help="Fraction of GPU memory for vLLM."),
    enforce_eager: bool = typer.Option(True, help="Skip vLLM CUDA graph capture."),
    seed: int = typer.Option(0, help="Random seed for shuffling and sampling."),
    wandb_mode: str = typer.Option("off", help="wandb mode: off | offline | online."),
    wandb_project: str = typer.Option("a5-sft", help="wandb project name."),
) -> None:
    """Run Algorithm 1 (SFT) on MATH reasoning traces."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import SamplingParams  # lazy: vLLM only loads when the script runs

    torch.manual_seed(seed)
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs = _load_sft_pairs(data_path, max_examples, correct_only)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="bfloat16",
        attn_implementation=attn_implementation,
    ).to(device)
    policy.train()

    num_updates = epochs * -(-len(pairs) // train_batch_size)
    warmup_steps = int(warmup_ratio * num_updates)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95)
    )
    scheduler = build_lr_scheduler(optimizer, warmup_steps, int(num_updates), min_lr_ratio)

    config = {k: v for k, v in {
        "model_path": model_path, "data_path": data_path, "num_pairs": len(pairs),
        "epochs": epochs, "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate, "warmup_steps": warmup_steps,
        "loss_normalization": loss_normalization, "correct_only": correct_only,
        "max_examples": max_examples, "seed": seed,
    }.items()}
    init_wandb(wandb_mode, wandb_project, config)

    microbatch_size = train_batch_size // gradient_accumulation_steps
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps")

    metrics_path = out / "metrics.jsonl"
    best_accuracy = -1.0

    # Fixed validation subset so accuracy is comparable across evals.
    template_text = R1_ZERO_PROMPT_PATH.read_text()
    val_rows = load_jsonl(val_path)[:val_size]
    val_prompts = [
        template_text.format(question=field(r, ("problem", "question"), "MATH validation")) for r in val_rows
    ]
    val_ground_truths = [
        str(field(r, ("answer", "expected_answer", "ground_truth"), "MATH validation")) for r in val_rows
    ]

    def run_validation(step: int) -> float:
        """vLLM validation accuracy on the fixed MATH validation subset."""
        nonlocal best_accuracy
        load_policy_into_vllm_instance(policy, vllm_model)
        sampling_params = SamplingParams(
            temperature=val_temperature,
            top_p=1.0,
            max_tokens=val_max_tokens,
            min_tokens=4,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        results = evaluate_vllm(
            vllm_model, r1_zero_reward_fn, val_prompts, sampling_params,
            ground_truths=val_ground_truths, output_path=str(out / f"val_generations_step{step}.jsonl"),
        )
        accuracy = results["metrics"]["accuracy"]
        logger.info("step %d | validation accuracy %.4f | %s", step, accuracy, results["metrics"])
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, "split": "validation", **results["metrics"]}) + "\n")
        wandb_log(step, {f"validation/{k}": v for k, v in results["metrics"].items()}, wandb_mode)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_dir = out / "best"
            policy.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.info("Saved new best checkpoint (accuracy %.4f) to %s", accuracy, best_dir)
        return accuracy

    # The vLLM engine is created before training so validation can reuse it.
    vllm_model = init_vllm(
        model_path,
        device=vllm_device,
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if eval_every:
        run_validation(0)

    step = 0
    for epoch in range(int(epochs)):
        order = list(range(len(pairs)))
        random.shuffle(order)
        for batch_start in range(0, len(order), train_batch_size):
            batch = [pairs[i] for i in order[batch_start : batch_start + train_batch_size]]
            microbatches = [
                ([p for p, _ in mb], [r for _, r in mb])
                for mb in (batch[mb_start : mb_start + microbatch_size] for mb_start in range(0, len(batch), microbatch_size))
            ]
            stats = sft_optimization_step(
                policy,
                optimizer,
                microbatches,
                tokenizer,
                gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                scheduler=scheduler,
                loss_normalization=loss_normalization,
                normalize_constant=normalize_constant,
            )
            step += 1
            logger.info(
                "epoch %d step %d/%d | loss %.4f | grad_norm %.3f | lr %.2e",
                epoch, step, int(num_updates), stats["loss"], stats["grad_norm"], scheduler.get_last_lr()[0],
            )
            wandb_log(step, {"train/loss": stats["loss"], "train/grad_norm": stats["grad_norm"], "train/lr": scheduler.get_last_lr()[0]}, wandb_mode)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": step, "split": "train", **stats}) + "\n")
            if eval_every and step % eval_every == 0:
                policy.eval()
                run_validation(step)
                policy.train()

    final_dir = out / "final"
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if eval_every:
        policy.eval()
        run_validation(step)
    typer.echo(f"Training complete: final checkpoint at {final_dir}, best validation accuracy {best_accuracy:.4f}")


if __name__ == "__main__":
    app()
