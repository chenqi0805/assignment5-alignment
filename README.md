# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Cluster runs

Every experiment and evaluation in the handout runs from one command. The
scripts are `typer` CLIs that **import on any machine** (vLLM is loaded
lazily inside each entry point, so `--help` works without CUDA); actually
running them needs the cluster GPUs and the `/data/a5-alignment/` mounts.
`flash-attn` is never required — pass `--attn-implementation
flash_attention_2` only on GPU nodes where it is installed.

Conventions:

- Every training script writes `metrics.jsonl` (one JSON object per training
  step and per validation eval, including wall-clock time) plus checkpoints
  under `--output-dir`.
- `--wandb-mode` defaults to `off` (set `offline` or `online` for
  Weights & Biases); wandb is never imported unless enabled.
- Validation/generation dumps land next to the metrics (e.g.
  `val_generations_step<N>.jsonl`), so failed runs are inspectable.

### Zero-shot baselines

```sh
# math_baseline (a): Qwen2.5-Math-1.5B on MATH validation
uv run python scripts/eval_math.py \
    --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
    --output-path outputs/math_baseline/generations.jsonl

# Benchmark baselines with Llama 3.1 8B (same script for mmlu/gsm8k/
# alpaca_eval/simple_safety_tests; data/mmlu and data/gsm8k ship with the repo)
uv run python scripts/eval_benchmark.py --benchmark mmlu \
    --model-path /data/a5-alignment/models/Llama-3.1-8B \
    --output-dir outputs/mmlu_baseline
uv run python scripts/eval_benchmark.py --benchmark gsm8k \
    --model-path /data/a5-alignment/models/Llama-3.1-8B \
    --output-dir outputs/gsm8k_baseline
uv run python scripts/eval_benchmark.py --benchmark alpaca_eval \
    --model-path /data/a5-alignment/models/Llama-3.1-8B \
    --output-dir outputs/alpaca_eval_baseline
uv run python scripts/eval_benchmark.py --benchmark simple_safety_tests \
    --model-path /data/a5-alignment/models/Llama-3.1-8B \
    --output-dir outputs/sst_baseline
```

`eval_benchmark.py` prints a JSON metrics block (accuracy, parse failures,
generation seconds, examples/second) and writes predictions to
`<output_dir>/<benchmark>_predictions.{jsonl,json}`. The AlpacaEval and
SimpleSafetyTests metrics include the exact 70B annotator command to run
afterwards (see "Annotators" below). To evaluate an instruction-tuned
checkpoint instead, point `--model-path` at the SFT output directory and add
`--use-alpaca-template` so inputs match the training format.

### SFT on MATH reasoning traces (sft_experiment, Algorithm 1)

```sh
# Full run: response-only SFT of Qwen2.5-Math-1.5B (AdamW, warmup + cosine LR)
uv run python scripts/train_sft.py \
    --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
    --data-path /data/a5-alignment/MATH/sft.jsonl \
    --output-dir outputs/sft

# Dataset-size ablation: train on a 25% subset
uv run python scripts/train_sft.py ... --max-examples $(( $(wc -l < /data/a5-alignment/MATH/sft.jsonl) / 4 ))

# Correct-only filtering variant: keep only traces graded correct
uv run python scripts/train_sft.py ... --correct-only
```

Checkpoints: `outputs/sft/final` (+ `step_<N>` snapshots at validation
cadence); learning curves: `outputs/sft/metrics.jsonl`.

### Expert iteration (expert_iteration_experiment, Algorithm 2)

```sh
uv run python scripts/expert_iteration.py \
    --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
    --output-dir outputs/expert_iteration
```

Rollouts with reward 1.0 join the SFT buffer each iteration; per-iteration
checkpoints land in `outputs/expert_iteration/step_<N>/`, SFT-buffer stats
and validation accuracy in `metrics.jsonl`.

### GRPO (grpo_train_loop + Section 8 experiments)

```sh
# Default run: handout on-policy block (rollout batch 256, group 8,
# reinforce_with_baseline, 200 steps, AdamW lr 1e-5, grad accum 128)
uv run python scripts/train_grpo.py \
    --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
    --output-dir outputs/grpo \
    --vllm-device cuda:1

# Ablations (Section 8):
uv run python scripts/train_grpo.py ... --loss-type no_baseline
uv run python scripts/train_grpo.py ... --loss-type grpo_clip --epochs-per-rollout-batch 2
uv run python scripts/train_grpo.py ... --no-std-normalization        # Dr. GRPO advantages (Eq. 31)
uv run python scripts/train_grpo.py ... --loss-type grpo_no_clip --epochs-per-rollout-batch 2  # Eq. 34
uv run python scripts/train_grpo.py ... --length-norm masked_normalize
uv run python scripts/train_grpo.py ... --prompt-template question_only

# Leaderboard: full 5K validation, r1_zero prompt + reward, temp 1.0, 1024 tokens
uv run python scripts/train_grpo.py ... --leaderboard
```

Config validation enforces the handout pp. 27–28 sanity asserts (microbatch
divides train batch, group size divides rollout batch, etc.); per-update
logs cover loss, grad norm, token entropy, clip fraction, and reward means.

### Instruction tuning + DPO (supplement)

```sh
# sft_script + sft: Llama 3.1 8B on the packed single-turn
# UltraChat/SafetyTunedLlamas data (512 tokens, effective batch 32 via
# gradient accumulation, lr 2e-5, 3% warmup + cosine decay)
uv run python scripts/train_instruction_sft.py \
    --output-dir outputs/instruction_sft \
    --attn-implementation flash_attention_2

# dpo_training: RMSprop lr 1e-6, beta 0.1, effective batch 64, policy and
# reference on separate GPUs; best-val-accuracy checkpoint kept
uv run python scripts/train_dpo.py \
    --policy-path outputs/instruction_sft/final \
    --ref-path outputs/instruction_sft/final \
    --output-dir outputs/dpo \
    --policy-device cuda:0 --ref-device cuda:1
```

### Annotators (cluster-side judges)

Both commands below require 80GB+ GPUs; the model generations they consume
are produced by `eval_benchmark.py` above.

```sh
# AlpacaEval winrate vs GPT-4 Turbo with Llama 3.3 70B Instruct judge
uv run alpaca_eval --model_outputs outputs/alpaca_eval_baseline/alpaca_eval_predictions.json \
    --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn --base-dir .

# SimpleSafetyTests: proportion of outputs judged safe
uv run python scripts/evaluate_safety.py \
    --input-path outputs/sst_baseline/simple_safety_tests_predictions.jsonl \
    --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \
    --num-gpus 2 \
    --output-path outputs/sst_baseline/simple_safety_tests_annotations.jsonl
```

The same two commands serve the `_sft` (instruction-tuned) and DPO variants —
swap the predictions path for the checkpoint you are scoring.


