"""Zero-shot benchmark evaluations (handout Problems: mmlu/gsm8k/alpaca_eval/sst baselines).

One script, four benchmarks, base-model or SFT-checkpoint inputs:

- ``mmlu``: all 57 subject CSVs under ``data/mmlu/<split>/``, system-prompt +
  multiple-choice template, greedy decoding, ``parse_mmlu_response`` accuracy.
- ``gsm8k``: ``data/gsm8k/<split>.jsonl``, ``{question}\nAnswer:`` prompt,
  ``parse_gsm8k_response`` accuracy against the ``####`` gold.
- ``alpaca_eval``: generates predictions serialized as the JSON array the
  alpaca-eval annotator consumes (cluster-side 70B judge command printed).
- ``simple_safety_tests``: predictions as JSONL (``prompts_final``/``output``)
  ready for ``scripts/evaluate_safety.py`` (cluster-side).

SFT checkpoints are evaluated with ``--use-alpaca-template`` so inputs match
the instruction-tuning prompt format (handout Problems mmlu_sft/gsm8k_sft/...).

Example (cluster):
    uv run python scripts/eval_benchmark.py --benchmark mmlu \
        --model-path /data/a5-alignment/models/Llama-3.1-8B \
        --output-dir outputs/mmlu_baseline
"""

from __future__ import annotations

import csv
import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from xopen import xopen

from cs336_alignment.data import parse_gsm8k_response, parse_mmlu_response
from cs336_alignment.training import LLAMA_3_1_8B_PATH
from cs336_alignment.vllm_utils import init_vllm

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = REPO_ROOT / "cs336_alignment" / "prompts" / "zero_shot_system_prompt.prompt"
ALPACA_TEMPLATE_PATH = REPO_ROOT / "cs336_alignment" / "prompts" / "alpaca_sft.prompt"

MMLU_PROMPT_TEMPLATE = (
    "Answer the following multiple choice question about {subject}. "
    'Respond with a single sentence of the form "The correct answer is _", '
    "filling the blank with the letter corresponding to the correct answer "
    "(i.e., A, B, C or D).\n\n"
    "Question: {question}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Answer:"
)

MMLU_LETTERS = ("A", "B", "C", "D")

app = typer.Typer(add_completion=False)


class Benchmark(str, Enum):
    """The four evaluated benchmarks (typer renders the choices in --help)."""

    mmlu = "mmlu"
    gsm8k = "gsm8k"
    alpaca_eval = "alpaca_eval"
    simple_safety_tests = "simple_safety_tests"


def _format_instruction(instruction: str, use_alpaca_template: bool) -> str:
    """Wrap the task instruction in the Alpaca template for SFT checkpoints."""
    if not use_alpaca_template:
        return instruction
    template = ALPACA_TEMPLATE_PATH.read_text()
    return template.format(instruction=instruction, response="").rstrip()


def _mmlu_examples(data_dir: Path, split: str, subjects: Optional[str], limit: Optional[int]) -> list[dict]:
    subject_dirs = sorted((data_dir / "mmlu" / split).glob("*_" + split + ".csv"))
    if subjects:
        wanted = {s.strip() for s in subjects.split(",")}
        subject_dirs = [p for p in subject_dirs if p.stem.replace("_" + split, "") in wanted]
    examples = []
    for path in subject_dirs:
        subject = path.stem.replace("_" + split, "").replace("_", " ")
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if len(row) != 6:
                    continue
                question, options, gold = row[0], row[1:5], row[5].strip().upper()
                examples.append(
                    {
                        "subject": subject,
                        "question": question,
                        "options": options,
                        "gold_letter": gold,
                        "instruction": MMLU_PROMPT_TEMPLATE.format(
                            subject=subject, question=question, a=options[0], b=options[1], c=options[2], d=options[3]
                        ),
                    }
                )
                if limit is not None and len(examples) >= limit:
                    return examples
    return examples


def _gsm8k_examples(data_dir: Path, split: str, limit: Optional[int]) -> list[dict]:
    examples = []
    with xopen(str(data_dir / "gsm8k" / f"{split}.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            gold = row["answer"].split("####")[-1].strip()
            examples.append(
                {
                    "question": row["question"],
                    "gold_answer": gold,
                    "instruction": f"{row['question']}\nAnswer:",
                }
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def _alpaca_examples(data_dir: Path, limit: Optional[int]) -> list[dict]:
    examples = []
    with xopen(str(data_dir / "alpaca_eval" / "alpaca_eval.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                {
                    "instruction": row["instruction"],
                    "dataset": row.get("dataset", "alpaca_eval"),
                    "reference_output": row.get("output"),
                }
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def _sst_examples(data_dir: Path, limit: Optional[int]) -> list[dict]:
    examples = []
    with open(data_dir / "simple_safety_tests" / "simple_safety_tests.csv", newline="") as f:
        for row in csv.DictReader(f):
            examples.append({"prompts_final": row["prompts_final"], "instruction": row["prompts_final"]})
            if limit is not None and len(examples) >= limit:
                break
    return examples


@app.command()
def main(
    benchmark: Benchmark = typer.Option(..., help="Which benchmark to evaluate."),
    model_path: str = typer.Option(LLAMA_3_1_8B_PATH, help="Model checkpoint (base model or SFT checkpoint)."),
    data_dir: Path = typer.Option(REPO_ROOT / "data", help="Repo data directory (MMLU/GSM8K ship with the repo)."),
    output_dir: Path = typer.Option(..., help="Directory for predictions and metrics."),
    split: str = typer.Option("test", help="Split for MMLU (test/val/dev) and GSM8K (test/train)."),
    subjects: Optional[str] = typer.Option(None, help="Comma-separated MMLU subject filter (default: all 57)."),
    limit: Optional[int] = typer.Option(None, help="Cap the number of examples (smoke testing)."),
    use_alpaca_template: bool = typer.Option(
        False, help="Format inputs with the Alpaca SFT template (for instruction-tuned checkpoints)."
    ),
    use_system_prompt: bool = typer.Option(
        True, help="Prepend the handout's zero-shot system prompt (supplement section 2)."
    ),
    temperature: float = typer.Option(0.0, help="Sampling temperature (handout: greedy, 0.0)."),
    top_p: float = typer.Option(1.0, help="Top-p (handout: 1.0)."),
    max_tokens: int = typer.Option(512, help="Max generation length."),
    device: str = typer.Option("cuda:0", help="Device for the vLLM engine."),
    dtype: str = typer.Option("bfloat16", help="vLLM weight dtype."),
    gpu_memory_utilization: float = typer.Option(0.85, help="Fraction of GPU memory for vLLM."),
    enforce_eager: bool = typer.Option(True, help="Skip CUDA graph capture (faster startup)."),
) -> None:
    """Evaluate a model on MMLU, GSM8K, AlpacaEval, or SimpleSafetyTests."""
    from vllm import SamplingParams  # lazy: vLLM only loads when the script runs

    output_dir.mkdir(parents=True, exist_ok=True)
    examples = {
        "mmlu": lambda: _mmlu_examples(data_dir, split, subjects, limit),
        "gsm8k": lambda: _gsm8k_examples(data_dir, split, limit),
        "alpaca_eval": lambda: _alpaca_examples(data_dir, limit),
        "simple_safety_tests": lambda: _sst_examples(data_dir, limit),
    }[benchmark.value]()
    if not examples:
        raise ValueError(f"No examples found for benchmark={benchmark} split={split} in {data_dir}")

    system_template = SYSTEM_PROMPT_PATH.read_text() if use_system_prompt else None
    prompts = []
    for example in examples:
        body = _format_instruction(example["instruction"], use_alpaca_template)
        prompts.append(system_template.format(instruction=body) if system_template else body)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        # The system prompt casts the answer as one turn of a conversation:
        # stop when the model starts the next "# Query:" turn.
        stop=["# Query:"] if system_template else None,
    )

    vllm_model = init_vllm(
        model_path,
        device=device,
        dtype=dtype,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    start = time.perf_counter()
    outputs = vllm_model.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    generations = [output.outputs[0].text for output in outputs]
    throughput = len(generations) / elapsed if elapsed else 0.0
    logger.info("Generated %d responses in %.1fs (%.2f examples/second)", len(generations), elapsed, throughput)

    metrics: dict = {
        "benchmark": benchmark,
        "model_path": model_path,
        "num_examples": len(generations),
        "generation_seconds": elapsed,
        "examples_per_second": throughput,
        "use_alpaca_template": use_alpaca_template,
        "use_system_prompt": use_system_prompt,
    }

    if benchmark == "mmlu":
        predictions_path = output_dir / "mmlu_predictions.jsonl"
        correct, unparsed = 0, 0
        with open(predictions_path, "w") as f:
            for example, prompt, generation in zip(examples, prompts, generations):
                predicted = parse_mmlu_response(example, generation)
                if predicted is None:
                    unparsed += 1
                correct += int(predicted == example["gold_letter"])
                f.write(
                    json.dumps(
                        {
                            "subject": example["subject"],
                            "question": example["question"],
                            "gold_letter": example["gold_letter"],
                            "predicted_letter": predicted,
                            "generation": generation,
                            "prompt": prompt,
                        }
                    )
                    + "\n"
                )
        metrics.update(
            {
                "accuracy": correct / len(generations),
                "num_unparsed": unparsed,
                "predictions_path": str(predictions_path),
            }
        )

    elif benchmark == "gsm8k":
        predictions_path = output_dir / "gsm8k_predictions.jsonl"
        correct, unparsed = 0, 0
        with open(predictions_path, "w") as f:
            for example, prompt, generation in zip(examples, prompts, generations):
                predicted = parse_gsm8k_response(generation)
                if predicted is None:
                    unparsed += 1
                correct += int(predicted == example["gold_answer"])
                f.write(
                    json.dumps(
                        {
                            "question": example["question"],
                            "gold_answer": example["gold_answer"],
                            "predicted_answer": predicted,
                            "generation": generation,
                            "prompt": prompt,
                        }
                    )
                    + "\n"
                )
        metrics.update(
            {
                "accuracy": correct / len(generations),
                "num_unparsed": unparsed,
                "predictions_path": str(predictions_path),
            }
        )

    elif benchmark == "alpaca_eval":
        # AlpacaEval's evaluator consumes a JSON array with
        # instruction/output/generator/dataset keys (supplement alpaca_eval_baseline).
        predictions_path = output_dir / "alpaca_eval_predictions.json"
        generator = Path(model_path).name.lower().replace("/", "-")
        payload = [
            {
                "instruction": example["instruction"],
                "output": generation,
                "generator": generator,
                "dataset": example["dataset"],
            }
            for example, generation in zip(examples, generations)
        ]
        with open(predictions_path, "w") as f:
            json.dump(payload, f)
        metrics["predictions_path"] = str(predictions_path)
        metrics["annotator_command"] = (
            "uv run alpaca_eval --model_outputs "
            f"{predictions_path} --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn --base-dir ."
        )

    else:  # simple_safety_tests
        predictions_path = output_dir / "simple_safety_tests_predictions.jsonl"
        with open(predictions_path, "w") as f:
            for example, generation in zip(examples, generations):
                f.write(json.dumps({"prompts_final": example["prompts_final"], "output": generation}) + "\n")
        metrics["predictions_path"] = str(predictions_path)
        metrics["annotator_command"] = (
            "uv run python scripts/evaluate_safety.py "
            f"--input-path {predictions_path} "
            "--model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct "
            "--num-gpus 2 --output-path "
            f"{output_dir / 'simple_safety_tests_annotations.jsonl'}"
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()
