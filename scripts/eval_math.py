"""Zero-shot MATH baseline (handout Problem: math_baseline).

Loads MATH validation examples, prompts the model with the r1_zero template,
generates with vLLM (temperature 1.0, top-p 1.0, max 1024 tokens, stopping at
the second answer tag), grades with ``r1_zero_reward_fn``, prints accuracy
with the math_baseline (b) category counts, and serializes everything to
JSONL for later analysis.

Example (cluster):
    uv run python scripts/eval_math.py \
        --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
        --data-path /data/a5-alignment/MATH/validation.jsonl \
        --output-path outputs/eval_math_baseline.jsonl
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import typer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.training import MATH_VALIDATION_PATH, QWEN_2_5_MATH_1_5B_PATH, field, load_jsonl
from cs336_alignment.vllm_utils import evaluate_vllm, init_vllm

logger = logging.getLogger(__name__)

R1_ZERO_PROMPT_PATH = Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_path: str = typer.Option(QWEN_2_5_MATH_1_5B_PATH, help="Model checkpoint for generation."),
    data_path: str = typer.Option(MATH_VALIDATION_PATH, help="MATH validation JSONL (question + gold answer)."),
    output_path: str = typer.Option(
        "outputs/eval_math_baseline.jsonl", help="Where to write prompts, generations, and scores."
    ),
    limit: Optional[int] = typer.Option(None, help="Evaluate only the first N examples (smoke testing)."),
    temperature: float = typer.Option(1.0, help="Sampling temperature (handout: 1.0)."),
    top_p: float = typer.Option(1.0, help="Top-p (handout: 1.0)."),
    max_tokens: int = typer.Option(1024, help="Max generation length (handout: 1024)."),
    min_tokens: int = typer.Option(4, help="Min generation length (disallows empty responses)."),
    seed: Optional[int] = typer.Option(None, help="vLLM sampling seed."),
    device: str = typer.Option("cuda:0", help="Device for the vLLM engine."),
    dtype: str = typer.Option("bfloat16", help="vLLM weight dtype."),
    gpu_memory_utilization: float = typer.Option(0.85, help="Fraction of GPU memory for vLLM."),
    enforce_eager: bool = typer.Option(True, help="Skip CUDA graph capture (faster startup)."),
) -> None:
    """Evaluate zero-shot MATH performance with the r1_zero prompt."""
    from vllm import SamplingParams  # lazy: vLLM only loads when the script runs

    examples = load_jsonl(data_path)
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"No examples loaded from {data_path}")

    questions = [field(row, ("problem", "question"), "MATH validation") for row in examples]
    ground_truths = [str(field(row, ("answer", "expected_answer", "ground_truth"), "MATH validation")) for row in examples]

    template = R1_ZERO_PROMPT_PATH.read_text()
    prompts = [template.format(question=question) for question in questions]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )

    vllm_model = init_vllm(
        model_path,
        device=device,
        dtype=dtype,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    start = time.perf_counter()
    results = evaluate_vllm(
        vllm_model,
        r1_zero_reward_fn,
        prompts,
        sampling_params,
        ground_truths=ground_truths,
        output_path=output_path,
    )
    elapsed = time.perf_counter() - start

    metrics = results["metrics"]
    metrics = {**metrics, "generation_seconds": elapsed, "examples_per_second": metrics["num_examples"] / elapsed if elapsed else 0.0}
    print(json.dumps(metrics, indent=2))
    typer.echo(f"Wrote {metrics['num_examples']} evaluated generations to {output_path}")


if __name__ == "__main__":
    app()
