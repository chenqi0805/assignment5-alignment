"""DPO training on Anthropic HH (supplement Problems dpo_training, pp. 17-19).

Per-instance DPO (no batching through the LMs; the handout recommends this
for memory) with gradient accumulation for an effective batch of 64, RMSprop
at lr 1e-6 with beta = 0.1, one epoch over the combined single-turn HH pairs,
and validation "classification accuracy" of the implicit reward model on a
held-out slice (an example is correct when the chosen completion has the
higher implicit reward, i.e. a positive DPO logit). The checkpoint with the
best validation accuracy is saved.

Example (cluster, 2 GPUs):
    uv run python scripts/train_dpo.py \
        --policy-path outputs/instruction_sft/final \
        --ref-path outputs/instruction_sft/final \
        --data-dir /data/a5-alignment/hh \
        --output-dir outputs/dpo \
        --policy-device cuda:0 --ref-device cuda:1
"""

from __future__ import annotations

import gzip
import json
import logging
import random
from pathlib import Path
from typing import Optional

import torch
import typer

from cs336_alignment.data import _split_hh_dialogue
from cs336_alignment.dpo import (
    ALPACA_SFT_PROMPT_PATH,
    _sequence_log_prob,
    compute_per_instance_dpo_loss,
)
from cs336_alignment.training import init_wandb, wandb_log

logger = logging.getLogger(__name__)

HH_DATA_DIR = "/data/a5-alignment/hh"

app = typer.Typer(add_completion=False)


def _load_hh_pairs(data_dir: str) -> list[dict]:
    """Load the four HH jsonl.gz files, keeping single-turn pairs only.

    Supplement p. 17 processing: ignore multi-turn conversations (the human
    sent more than one message), separate the instruction (first human
    message) from the chosen/rejected assistant responses, and remember the
    source file. Parsing reuses the core dialogue splitter so the format
    contract lives in one place.
    """
    pairs: list[dict] = []
    source_counts: dict[str, int] = {}
    for path in sorted(Path(data_dir).glob("*.jsonl.gz")):
        kept = 0
        with gzip.open(path, "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                chosen_turns = _split_hh_dialogue(row["chosen"])
                rejected_turns = _split_hh_dialogue(row["rejected"])
                if len(chosen_turns) != 2 or len(rejected_turns) != 2:
                    continue  # multi-turn: the human sent more than one message
                if chosen_turns[0]["content"] != rejected_turns[0]["content"]:
                    continue  # the sides diverge before the assistant response
                pairs.append({
                    "prompt": chosen_turns[0]["content"],
                    "response_chosen": chosen_turns[1]["content"],
                    "response_rejected": rejected_turns[1]["content"],
                    "source": path.name,
                })
                kept += 1
        source_counts[path.name] = kept
        logger.info("Loaded %d single-turn pairs from %s", kept, path.name)
    logger.info("HH pairs loaded per source: %s (total %d)", source_counts, len(pairs))
    return pairs


def _dpo_logits(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> float:
    """beta * [(log pi(y_w|x) - log pi_ref(y_w|x)) - (log pi(y_l|x) - log pi_ref(y_l|x))].

    Same formatting (Alpaca template + EOS) and unconditional-sequence
    log-prob primitives as the core loss, so validation cannot drift from
    training. Positive logits mean the implicit reward prefers the chosen
    response.
    """
    template = ALPACA_SFT_PROMPT_PATH.read_text()

    def formatted(response: str) -> str:
        return template.format(instruction=prompt, response=response).rstrip() + tokenizer.eos_token

    with torch.inference_mode():
        policy_chosen = _sequence_log_prob(lm, tokenizer, formatted(response_chosen)).to(lm.device)
        policy_rejected = _sequence_log_prob(lm, tokenizer, formatted(response_rejected)).to(lm.device)
        ref_chosen = _sequence_log_prob(lm_ref, tokenizer, formatted(response_chosen)).to(lm.device)
        ref_rejected = _sequence_log_prob(lm_ref, tokenizer, formatted(response_rejected)).to(lm.device)
    logits = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return float(logits)


def validation_accuracy(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    val_pairs: list[dict],
) -> dict[str, float]:
    """Implicit-reward classification accuracy over the validation pairs."""
    correct = 0
    for pair in val_pairs:
        logits = _dpo_logits(
            lm, lm_ref, tokenizer, beta,
            pair["prompt"], pair["response_chosen"], pair["response_rejected"],
        )
        if logits > 0:
            correct += 1
    return {"classification_accuracy": correct / max(len(val_pairs), 1), "num_examples": len(val_pairs)}


@app.command()
def main(
    policy_path: str = typer.Option(
        "outputs/instruction_sft/final", help="Policy init: your instruction-tuned model."
    ),
    ref_path: str = typer.Option(
        "outputs/instruction_sft/final", help="Reference init (handout: a second copy of the SFT model)."
    ),
    tokenizer_path: str = typer.Option(
        "outputs/instruction_sft/final", help="Tokenizer path (defaults to the policy checkpoint)."
    ),
    data_dir: str = typer.Option(HH_DATA_DIR, help="Directory with the four HH jsonl.gz files."),
    output_dir: str = typer.Option("outputs/dpo", help="Directory for checkpoints and metrics."),
    val_size: int = typer.Option(200, help="Validation pairs held out from training (handout: ~200)."),
    max_train_pairs: Optional[int] = typer.Option(None, help="Cap training pairs (smoke runs)."),
    beta: float = typer.Option(0.1, help="DPO beta (handout start: 0.1)."),
    learning_rate: float = typer.Option(1e-6, help="RMSprop learning rate (handout start: 1e-6)."),
    batch_size: int = typer.Option(
        64, help="Effective pairs per optimizer step via gradient accumulation (handout start: 64)."
    ),
    epochs: int = typer.Option(1, help="Epochs over the HH pairs (handout: 1)."),
    eval_every: int = typer.Option(50, help="Validation accuracy every N optimizer steps."),
    max_grad_norm: float = typer.Option(
        0.0, help="Gradient clipping value; 0 disables clipping (original DPO used none)."
    ),
    attn_implementation: Optional[str] = typer.Option(
        None, help="HF attention implementation; pass 'flash_attention_2' to opt in (requires flash-attn)."
    ),
    policy_device: str = typer.Option("cuda:0", help="Device for the trained model."),
    ref_device: str = typer.Option("cuda:1", help="Device for the reference model."),
    seed: int = typer.Option(0, help="Random seed."),
    wandb_mode: str = typer.Option("off", help="wandb mode: off | offline | online."),
    wandb_project: str = typer.Option("a5-dpo", help="wandb project name."),
) -> None:
    """Train the instruction-tuned model with DPO on the HH preference data."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs = _load_hh_pairs(data_dir)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    if max_train_pairs is not None:
        train_pairs = train_pairs[:max_train_pairs]
    logger.info(
        "DPO split: %d training pairs, %d validation pairs", len(train_pairs), len(val_pairs)
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    policy = AutoModelForCausalLM.from_pretrained(
        policy_path, dtype="bfloat16", attn_implementation=attn_implementation,
    ).to(policy_device)
    policy.train()
    # The reference model is frozen by definition; the core loss runs it
    # under no_grad.
    ref = AutoModelForCausalLM.from_pretrained(
        ref_path, dtype="bfloat16", attn_implementation=attn_implementation,
    ).to(ref_device)
    ref.eval()

    optimizer = torch.optim.RMSprop(policy.parameters(), lr=learning_rate)

    init_wandb(wandb_mode, wandb_project, {
        "policy_path": policy_path, "ref_path": ref_path, "data_dir": data_dir,
        "beta": beta, "learning_rate": learning_rate, "batch_size": batch_size,
        "epochs": epochs, "val_size": val_size, "train_pairs": len(train_pairs), "seed": seed,
    })

    metrics_path = out / "metrics.jsonl"
    optimizer_step = 0
    best_accuracy = float("-inf")
    running_loss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        for index, pair in enumerate(train_pairs):
            # Per-instance loss (no batching through the LMs, supplement
            # p. 18), divided so accumulated gradients average over the
            # effective batch.
            loss = compute_per_instance_dpo_loss(
                policy, ref, tokenizer, beta,
                pair["prompt"], pair["response_chosen"], pair["response_rejected"],
            )
            scaled = loss / batch_size
            scaled.backward()
            running_loss += float(loss.detach())

            if (index + 1) % batch_size != 0:
                continue

            if max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            else:
                grad_norm = float("nan")
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step += 1
            avg_loss = running_loss / batch_size
            running_loss = 0.0
            logger.info(
                "epoch %d step %d | dpo loss %.5f | grad_norm %s",
                epoch, optimizer_step, avg_loss, grad_norm,
            )
            wandb_log(optimizer_step, {"train/loss": avg_loss}, wandb_mode)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "step": optimizer_step, "epoch": epoch, "split": "train",
                    "loss": avg_loss,
                    "grad_norm": None if grad_norm != grad_norm else float(grad_norm),
                }) + "\n")

            if eval_every and optimizer_step % eval_every == 0:
                policy.eval()
                metrics = validation_accuracy(policy, ref, tokenizer, beta, val_pairs)
                policy.train()
                logger.info(
                    "epoch %d step %d | validation accuracy %.4f (%d pairs)",
                    epoch, optimizer_step, metrics["classification_accuracy"], metrics["num_examples"],
                )
                wandb_log(optimizer_step, {
                    "validation/classification_accuracy": metrics["classification_accuracy"],
                }, wandb_mode)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({
                        "step": optimizer_step, "split": "validation", **metrics,
                    }) + "\n")
                if metrics["classification_accuracy"] > best_accuracy:
                    best_accuracy = metrics["classification_accuracy"]
                    best_dir = out / "best"
                    policy.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    logger.info(
                        "New best validation accuracy %.4f; checkpoint saved to %s",
                        best_accuracy, best_dir,
                    )

    if val_pairs and eval_every:
        policy.eval()
        final_metrics = validation_accuracy(policy, ref, tokenizer, beta, val_pairs)
        policy.train()
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": optimizer_step, "split": "validation_final", **final_metrics}) + "\n")
        logger.info("Final validation accuracy %.4f", final_metrics["classification_accuracy"])

    final_dir = out / "final"
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    typer.echo(
        f"DPO training complete after {optimizer_step} optimizer steps; "
        f"best validation accuracy {best_accuracy:.4f}; final model at {final_dir}"
    )


if __name__ == "__main__":
    app()
