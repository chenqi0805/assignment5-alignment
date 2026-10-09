"""Instruction fine-tuning of Llama 3.1 8B (supplement Problems sft_script + sft).

Trains on the packed single-turn instruction data (UltraChat-200K +
SafetyTunedLlamas mix) with the Alpaca template, full-sequence LM loss over
packed 512-token documents, gradient accumulation (handout: microbatch 2 in
bf16/FA-2, effective batch 32), and AdamW with a linear 3% warmup followed by
cosine decay to 0 at lr 2e-5.

Example (cluster):
    uv run python scripts/train_instruction_sft.py \
        --model-path /data/a5-alignment/models/Llama-3.1-8B \
        --train-path /data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz \
        --val-path /data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz \
        --output-dir outputs/instruction_sft \
        --attn-implementation flash_attention_2
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

from cs336_alignment.data import get_packed_sft_dataset, iterate_batches
from cs336_alignment.training import (
    LLAMA_3_1_8B_PATH,
    ULTRACHAT_TEST_PATH,
    ULTRACHAT_TRAIN_PATH,
    build_lr_scheduler,
    init_wandb,
    wandb_log,
)

logger = logging.getLogger(__name__)

INSTRUCTION_TRAIN_PATH = ULTRACHAT_TRAIN_PATH
INSTRUCTION_VAL_PATH = ULTRACHAT_TEST_PATH

app = typer.Typer(add_completion=False)


def _prepare_plain_jsonl(source: str, prepared_path: Path) -> Path:
    """Materialize a (possibly gzipped) jsonl as plain text with prompt/response keys.

    The core PackedSFTDataset reads plain-text jsonl rows with "prompt" and
    "response" keys; the cluster instruction data ships gzipped with its own
    key names. Decompress once into the output directory and normalize the
    common aliases so the tested dataset class can consume it unchanged.
    """
    if prepared_path.exists():
        return prepared_path
    opener = gzip.open if source.endswith(".gz") else open
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    with opener(source, "rt") as src, open(prepared_path, "w") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt", row.get("instruction"))
            response = row.get("response", row.get("output"))
            if prompt is None or response is None:
                raise ValueError(
                    f"Row in {source} has neither prompt/instruction nor "
                    f"response/output keys: {sorted(row)[:8]}"
                )
            dst.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
            n_docs += 1
    logger.info("Prepared %d documents from %s -> %s", n_docs, source, prepared_path)
    return prepared_path


@app.command()
def main(
    model_path: str = typer.Option(LLAMA_3_1_8B_PATH, help="Base model checkpoint (Llama 3.1 8B)."),
    train_path: str = typer.Option(INSTRUCTION_TRAIN_PATH, help="Instruction training jsonl(.gz)."),
    val_path: str = typer.Option(INSTRUCTION_VAL_PATH, help="Instruction validation jsonl(.gz)."),
    output_dir: str = typer.Option("outputs/instruction_sft", help="Directory for checkpoints and metrics."),
    seq_length: int = typer.Option(512, help="Packed sequence length m (handout: 512)."),
    batch_size: int = typer.Option(32, help="Effective sequences per gradient step (handout: 32)."),
    gradient_accumulation_steps: int = typer.Option(
        16, help="Microbatches per optimizer step (handout: 2 per device microbatch x 8)."
    ),
    learning_rate: float = typer.Option(2e-5, help="Peak AdamW learning rate (handout: 2e-5)."),
    min_learning_rate: float = typer.Option(0.0, help="Final cosine-decay learning rate."),
    warmup_ratio: float = typer.Option(0.03, help="Linear warmup fraction of total steps (handout: 3%)."),
    epochs: int = typer.Option(1, help="Epochs over the packed dataset (handout: 1)."),
    max_grad_norm: float = typer.Option(1.0, help="Gradient clipping value."),
    weight_decay: float = typer.Option(0.0, help="AdamW weight decay."),
    max_train_documents: Optional[int] = typer.Option(
        None, help="Cap the number of training documents (smoke runs)."
    ),
    val_batches: int = typer.Option(50, help="Validation microbatches per eval (0 = skip validation)."),
    eval_every: int = typer.Option(500, help="Log validation loss every N optimizer steps."),
    attn_implementation: Optional[str] = typer.Option(
        None, help="HF attention implementation; pass 'flash_attention_2' to opt in (requires flash-attn)."
    ),
    device: str = typer.Option("cuda:0", help="Training device."),
    seed: int = typer.Option(0, help="Random seed."),
    wandb_mode: str = typer.Option("off", help="wandb mode: off | offline | online."),
    wandb_project: str = typer.Option("a5-instruction-sft", help="wandb project name."),
) -> None:
    """Fine-tune Llama 3.1 8B base on the packed instruction data."""
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    assert batch_size % gradient_accumulation_steps == 0, (
        "batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_batch_size = batch_size // gradient_accumulation_steps

    prepared_train = _prepare_plain_jsonl(train_path, out / "prepared_train.jsonl")
    prepared_val = _prepare_plain_jsonl(val_path, out / "prepared_val.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="bfloat16",
        attn_implementation=attn_implementation,
    ).to(device)
    model.train()

    train_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer, dataset_path=str(prepared_train), seq_length=seq_length, shuffle=True
    )
    train_batches = iterate_batches(train_dataset, micro_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    microbatches_per_epoch = len(train_dataset) // micro_batch_size
    total_optimizer_steps = max(1, microbatches_per_epoch // gradient_accumulation_steps) * epochs
    warmup_steps = max(1, int(warmup_ratio * total_optimizer_steps))
    scheduler = build_lr_scheduler(
        optimizer, warmup_steps, total_optimizer_steps,
        min_lr_ratio=(min_learning_rate / learning_rate if learning_rate else 0.0),
    )
    logger.info(
        "Packed dataset: %d sequences of %d tokens | %d microbatches/epoch | "
        "%d optimizer steps total (warmup %d, microbatch %d, effective batch %d)",
        len(train_dataset), seq_length, microbatches_per_epoch,
        total_optimizer_steps, warmup_steps, micro_batch_size, batch_size,
    )

    init_wandb(wandb_mode, wandb_project, {
        "model_path": model_path, "seq_length": seq_length, "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps, "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio, "epochs": epochs, "max_train_documents": max_train_documents,
        "train_sequences": len(train_dataset), "seed": seed,
    })

    metrics_path = out / "metrics.jsonl"
    microbatch_index = 0
    optimizer_step = 0
    accumulation_loss = 0.0
    document_cap_reached = max_train_documents is not None and max_train_documents <= 0

    def run_validation() -> float:
        """Mean full-sequence LM loss over a capped slice of the validation pack."""
        if val_batches <= 0:
            return float("nan")
        val_dataset = get_packed_sft_dataset(
            tokenizer=tokenizer, dataset_path=str(prepared_val), seq_length=seq_length, shuffle=False
        )
        total, count = 0.0, 0
        model.eval()
        with torch.inference_mode():
            for val_index, val_batch in enumerate(iterate_batches(val_dataset, micro_batch_size, shuffle=False)):
                if val_index >= val_batches:
                    break
                val_ids = val_batch["input_ids"].to(device)
                val_labels = val_batch["labels"].to(device)
                val_logits = model(val_ids).logits
                loss = F.cross_entropy(
                    val_logits.view(-1, val_logits.shape[-1]).float(), val_labels.view(-1)
                )
                total += float(loss)
                count += 1
        model.train()
        return total / max(count, 1)

    for epoch in range(epochs):
        if document_cap_reached:
            break
        for batch in train_batches:
            lr = scheduler.get_last_lr()[0]

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids).logits
            # Full-sequence LM loss over the packed documents (supplement
            # p. 10: the packed strings are plain language-modeling targets).
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]).float(), labels.view(-1))
            scaled = loss / gradient_accumulation_steps
            scaled.backward()
            accumulation_loss += float(loss.detach())

            microbatch_index += 1
            if (
                max_train_documents is not None
                and microbatch_index * micro_batch_size >= max_train_documents
            ):
                document_cap_reached = True
            if microbatch_index % gradient_accumulation_steps != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            optimizer_step += 1

            avg_loss = accumulation_loss / gradient_accumulation_steps
            accumulation_loss = 0.0
            logger.info(
                "epoch %d step %d/%d | loss %.5f | lr %.3e | grad_norm %.3f",
                epoch, optimizer_step, total_optimizer_steps, avg_loss, lr, float(grad_norm),
            )
            wandb_log(optimizer_step, {
                "train/loss": avg_loss, "train/lr": lr, "train/grad_norm": float(grad_norm),
            }, wandb_mode)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "step": optimizer_step, "split": "train", "loss": avg_loss,
                    "lr": lr, "grad_norm": float(grad_norm),
                }) + "\n")

            if eval_every and optimizer_step % eval_every == 0:
                val_loss = run_validation()
                logger.info("epoch %d step %d | validation loss %.5f", epoch, optimizer_step, val_loss)
                wandb_log(optimizer_step, {"validation/loss": val_loss}, wandb_mode)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({
                        "step": optimizer_step, "split": "validation", "loss": val_loss,
                    }) + "\n")

    val_loss = run_validation()
    logger.info("final validation loss %.5f", val_loss)
    with open(metrics_path, "a") as f:
        f.write(json.dumps({"step": optimizer_step, "split": "validation_final", "loss": val_loss}) + "\n")

    final_dir = out / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    typer.echo(f"Instruction SFT complete after {optimizer_step} optimizer steps; model saved to {final_dir}")


if __name__ == "__main__":
    app()
