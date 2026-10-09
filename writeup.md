# CS336 Spring 2025 · Assignment 5 (Alignment) — Written Report

This is the written report for CS336 Assignment 5 (core assignment) and the optional
safety/RLHF supplement, organized problem-by-problem in the order the handouts present
them. It accompanies the implementation in `cs336_alignment/` (all 16 adapter primitives),
the experiment CLIs in `scripts/`, and the runbook in [`README.md`](./README.md#cluster-runs)
(every command below is the README's; this document links to it rather than restating its
conventions).

**Status legend**

- ✅ **Verified here** — implemented and exercised by the test suite in this environment
  (`uv run --no-sync pytest -q`, 31/31 passing at the time of writing).
- 🔍 **Best-effort from public data** — answered by inspecting public downloads in this
  environment; the cluster copies are authoritative.
- 📊 **[CLUSTER RUN PENDING]** — requires the course H100 cluster and its `/data/a5-alignment/`
  mounts, which do not exist in this environment. Each such section carries the exact
  command that produces the numbers (consistent with the README runbook) and a placeholder
  table. No experimental numbers below are invented; every missing value is a marked placeholder.

**How this report was produced.** Both handout PDFs in the repo root were text-extracted with
`pypdf` (the assignment handout, 38 pages; the safety/RLHF supplement, 19 pages); every
"Deliverable:" line in both was enumerated and mapped to a section below (§0). The
data-inspection answers (§9.1, §12.2) come from actually downloading and inspecting the public
datasets cited by the supplement.

---

## 0. Deliverable coverage map

Every `Deliverable:` line in both handout PDFs, mapped to the section of this report that
answers it and its status.

### Main handout (`cs336_spring2025_assignment5_alignment.pdf`)

| Handout problem | Deliverable | Section | Status |
|---|---|---|---|
| `math_baseline` (a) | Script to evaluate baseline zero-shot MATH | §1.1 | ✅ script verified importable; 📊 run pending |
| `math_baseline` (b) | Commentary on model/reward-function performance incl. examples | §1.1 | 📊 [CLUSTER RUN PENDING] |
| `math_baseline` (c) | 1–2 sentences with evaluation metrics | §1.1 | 📊 [CLUSTER RUN PENDING] |
| `tokenize_prompt_and_output` | Prompt/output tokenization + response mask | §2.1 | ✅ verified (test suite) |
| `compute_entropy` | Per-token entropy | §2.2 | ✅ verified |
| `get_response_log_probs` | Response log-probs (+ entropy) | §2.3 | ✅ verified |
| `masked_normalize` | Masked normalize | §2.4 | ✅ verified |
| `sft_microbatch_train_step` | Microbatch train step (SFT) | §2.5 | ✅ verified |
| `log_generations` | Function to log generations | §2.6 | ✅ verified + design rationale |
| `sft_experiment` 1 | Validation accuracy curves vs dataset size | §2.7 | 📊 [CLUSTER RUN PENDING] |
| `sft_experiment` 2 | Filtered dataset size + validation accuracy curve | §2.7 | 📊 [CLUSTER RUN PENDING] |
| `expert_iteration_experiment` | Validation accuracy curves for rollout configs (≥2 rollout counts, ≥2 epoch counts) | §3 | 📊 [CLUSTER RUN PENDING] |
| `expert_iteration_experiment` | Model with ≥15% MATH validation accuracy | §3 | 📊 [CLUSTER RUN PENDING] |
| `expert_iteration_experiment` | 2-sentence comparison vs SFT and across EI steps | §3 | 📊 [CLUSTER RUN PENDING] |
| `expert_iteration_experiment` | Entropy plot over training | §3 | 📊 [CLUSTER RUN PENDING] |
| `compute_group_normalized_rewards` | Group normalization (raw rewards + advantages) | §4.1 | ✅ verified |
| `compute_naive_policy_gradient_loss` | Naive policy-gradient loss | §4.2 | ✅ verified |
| `compute_grpo_clip_loss` | GRPO-Clip per-token loss | §4.3 | ✅ verified |
| `compute_policy_gradient_loss` | Policy-gradient wrapper | §4.4 | ✅ verified |
| `masked_mean` | Masked mean | §4.5 | ✅ verified |
| `grpo_microbatch_train_step` | Microbatch train step (GRPO) | §4.6 | ✅ verified |
| `grpo_train_loop` | Complete GRPO train loop + validation reward plot + example rollouts | §4.7 | ✅ implemented; 📊 run pending |
| `grpo_learning_rate` | Validation reward curves vs learning rate | §5.1 | 📊 [CLUSTER RUN PENDING] |
| `grpo_learning_rate` | Model with ≥25% MATH validation accuracy | §5.1 | 📊 [CLUSTER RUN PENDING] |
| `grpo_learning_rate` | 2-sentence discussion of other logged metrics | §5.1 | 📊 [CLUSTER RUN PENDING] |
| `grpo_baselines` | Validation reward curves per loss type | §5.2 | 📊 [CLUSTER RUN PENDING] |
| `grpo_baselines` | 2-sentence discussion of other logged metrics | §5.2 | 📊 [CLUSTER RUN PENDING] |
| `think_about_length_normalization` | Pros/cons of masked_mean vs masked_normalize | §5.3 | ✅ answered in full (conceptual) |
| `grpo_length_normalization` | End-to-end masked_mean vs masked_normalize comparison | §5.4 | 📊 [CLUSTER RUN PENDING] |
| `grpo_group_standard_deviation` | use_std_normalization True vs False comparison | §5.5 | 📊 [CLUSTER RUN PENDING] |
| `grpo_off_policy` | Implement off-policy GRPO | §5.6 | ✅ implemented (epochs/batching + old log-probs + GRPO-Clip) |
| `grpo_off_policy_sweep` | Off-policy sweep + experiment log + wall-clock comparison | §5.7 | 📊 [CLUSTER RUN PENDING] |
| `grpo_off_policy_clip_ablation` | Implement "GRPO-No-Clip" loss type + off-policy ablation | §5.8 | ✅ loss type implemented; 📊 run pending |
| `grpo_prompt_ablation` | R1-Zero vs question-only prompt curves + explanation | §5.9 | explanation pre-answered; 📊 run pending |
| `leaderboard` | Validation accuracy within 4 h on 2×H100 + screenshot | §6 | 📊 [CLUSTER RUN PENDING] |
| (handout) "On KL divergence" | Discussion of omitting the KL term | §7 | ✅ answered in full (conceptual) |

### Supplement (`cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`)

| Handout problem | Deliverable | Section | Status |
|---|---|---|---|
| `mmlu_baseline` (a) | MMLU answer-letter parser | §8.1 | ✅ verified |
| `mmlu_baseline` (b) | Script to evaluate zero-shot MMLU | §8 | ✅ script; 📊 run pending |
| `mmlu_baseline` (c) | Parse-failure count + examples | §8.2 | 📊 [CLUSTER RUN PENDING] |
| `mmlu_baseline` (d) | Throughput (examples/second) | §8.2 | 📊 [CLUSTER RUN PENDING] |
| `mmlu_baseline` (e) | 1–2 sentences with evaluation metrics | §8.2 | 📊 [CLUSTER RUN PENDING] |
| `mmlu_baseline` (f) | 2–4 sentence error analysis (10 wrong examples) | §8.2 | 📊 [CLUSTER RUN PENDING] |
| `gsm8k_baseline` (a) | GSM8K numeric-answer parser | §8.1 | ✅ verified |
| `gsm8k_baseline` (b) | Script to evaluate zero-shot GSM8K | §8 | ✅ script; 📊 run pending |
| `gsm8k_baseline` (c) | Parse-failure count + examples | §8.3 | 📊 [CLUSTER RUN PENDING] |
| `gsm8k_baseline` (d) | Throughput (examples/second) | §8.3 | 📊 [CLUSTER RUN PENDING] |
| `gsm8k_baseline` (e) | 1–2 sentences with evaluation metrics | §8.3 | 📊 [CLUSTER RUN PENDING] |
| `gsm8k_baseline` (f) | 2–4 sentence error analysis | §8.3 | 📊 [CLUSTER RUN PENDING] |
| `alpaca_eval_baseline` (a) | Script generating zero-shot AlpacaEval outputs | §8 | ✅ script; 📊 run pending |
| `alpaca_eval_baseline` (b) | Throughput (examples/second) | §8.4 | 📊 [CLUSTER RUN PENDING] |
| `alpaca_eval_baseline` (c) | Winrate + length-controlled winrate (70B annotator) | §8.4 | 📊 [CLUSTER RUN PENDING] |
| `alpaca_eval_baseline` (d) | 2–4 sentence error analysis of dispreferred outputs | §8.4 | 📊 [CLUSTER RUN PENDING] |
| `sst_baseline` (a) | Script generating zero-shot SimpleSafetyTests outputs | §8 | ✅ script; 📊 run pending |
| `sst_baseline` (b) | Throughput (examples/second) | §8.5 | 📊 [CLUSTER RUN PENDING] |
| `sst_baseline` (c) | Proportion of safe outputs (Llama 3.3 70B judge) | §8.5 | 📊 [CLUSTER RUN PENDING] |
| `sst_baseline` (d) | 2–4 sentence error analysis of unsafe outputs | §8.5 | 📊 [CLUSTER RUN PENDING] |
| `data_loading` (a) | Packed instruction-tuning `Dataset` subclass | §9.2 | ✅ verified |
| `data_loading` (b) | Batch iterator over the Dataset | §9.2 | ✅ verified |
| `look_at_sft` | 2–4 sentences on tasks + data quality in the instruction-tuning set | §9.1 | 🔍 answered from the public download |
| `sft_script` | Instruction-tuning training script | §9.3 | ✅ implemented (`scripts/train_instruction_sft.py`) |
| `sft` | Training-setup description + final validation loss + learning curve | §9.3 | 📊 [CLUSTER RUN PENDING] |
| `mmlu_sft` | Throughput, metrics vs baseline, error analysis on the SFT model | §10.1 | 📊 [CLUSTER RUN PENDING] |
| `gsm8k_sft` | Throughput, metrics vs baseline, error analysis on the SFT model | §10.2 | 📊 [CLUSTER RUN PENDING] |
| `alpaca_eval_sft` | Throughput, winrate + LC winrate vs baseline, error analysis | §10.3 | 📊 [CLUSTER RUN PENDING] |
| `sst_sft` | Throughput, safe proportion vs baseline, error analysis | §10.4 | 📊 [CLUSTER RUN PENDING] |
| `red_teaming` (a) | 1–3 sentences with three potential misuses | §11.1 | ✅ answered in full (conceptual) |
| `red_teaming` (b) | Red-teaming procedure + results for three malicious applications | §11.2 | 📊 [CLUSTER RUN PENDING] (needs the instruction-tuned checkpoint) |
| `look_at_hh` 1 | Function to load the Anthropic HH dataset | §12.1 | ✅ implemented (`cs336_alignment.data.load_anthropic_hh` + `train_dpo.py` gzip wrapper) |
| `look_at_hh` 2 | Commentary on 3 helpful + 3 harmless chosen/rejected pairs | §12.2 | 🔍 answered from the public download |
| `dpo_loss` | Per-instance DPO loss function | §12.3 | ✅ verified (numeric target) |
| `dpo_training` 1 | DPO training script + validation accuracy curve screenshot | §12.4 | ✅ script; 📊 run pending |
| `dpo_training` 2 | AlpacaEval winrates of the DPO-trained model | §12.4 | 📊 [CLUSTER RUN PENDING] |
| `dpo_training` 3 | SimpleSafetyTests evaluation of the DPO-trained model | §12.4 | 📊 [CLUSTER RUN PENDING] |
| `dpo_training` 4 | GSM8K/MMLU evaluations (alignment tax) | §12.4 | 📊 [CLUSTER RUN PENDING] |

---

# Part I — Core assignment: reasoning (MATH)

## 1. `math_baseline` — zero-shot MATH performance (4 points)

**Script (a).** Implemented at `scripts/eval_math.py` (typer CLI; vLLM loaded lazily inside
`main` so the module imports on any machine). It loads MATH validation examples from
`/data/a5-alignment/MATH/validation.jsonl`, formats them with the R1-Zero prompt
(`cs336_alignment/prompts/r1_zero.prompt`), generates with vLLM (temperature 1.0, top-p 1.0,
max tokens 1024, stopping at `</answer>` and including it in the output), scores each
generation with `cs336_alignment.drgrpo_grader.r1_zero_reward_fn`, and serializes examples,
generations, and per-example rewards to disk. The reusable evaluation core
(`init_vllm`, `load_policy_into_vllm_instance`, `evaluate_vllm`) lives in
`cs336_alignment/vllm_utils.py`, shared by every later experiment.

Cluster run (from the [README runbook](./README.md#zero-shot-baselines)):

```sh
uv run python scripts/eval_math.py \
    --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
    --output-path outputs/math_baseline/generations.jsonl
```

Output lands in `outputs/math_baseline/generations.jsonl` (per-example prompt, generation,
format/answer/total rewards); the printed metrics block summarizes the table below.

**(b) Reward-function commentary.** 📊 **[CLUSTER RUN PENDING]** — the analysis categories are
fixed by the handout: (1) format reward 1 and answer reward 1 (correct); (2) format reward 1,
answer reward 0 (well-formatted but wrong); (3) format reward 0 (did not produce the
`</think> <answer>…</answer>` structure the strict `r1_zero_reward_fn` requires, hence
automatically answer reward 0). For each bucket the run dumps every generation with its
rewards, so the commentary will quote at least 10 format-reward-0 cases (judging whether the
base model's output or the parser is at fault — e.g., a model that answers correctly but never
closes the `</think>` tag is a parser/prompt mismatch, not a wrong model) and at least 10
format-reward-1/answer-reward-0 cases (graded-but-wrong answers, e.g. arithmetic slips inside
a well-formed `\boxed{}`).

| Category (per `r1_zero_reward_fn`) | Count (of 5,000) |
|---|---|
| Format 1, answer 1 (correct) | [CLUSTER RUN PENDING] |
| Format 1, answer 0 | [CLUSTER RUN PENDING] |
| Format 0, answer 0 | [CLUSTER RUN PENDING] |

**(c) Zero-shot metrics.** 📊 **[CLUSTER RUN PENDING]**

| Metric | Value |
|---|---|
| MATH validation accuracy (mean answer reward, all 5K examples) | [CLUSTER RUN PENDING] |
| Format reward (mean) | [CLUSTER RUN PENDING] |

## 2. Supervised finetuning for MATH (Algorithm 1)

### 2.1 `tokenize_prompt_and_output` (2 points) — ✅ verified

Tokenizes each prompt and each output **separately** with the Qwen tokenizer, concatenates
per pair, and returns `input_ids = concat[:, :-1]`, `labels = concat[:, 1:]`, and a boolean
`response_mask` that is True exactly at the *labels* positions corresponding to output tokens
(the mask lives on the shifted sequence, not on `input_ids` — predicting token `t+1` from
prefix `≤ t`). Verified by `tests/test_sft.py::test_tokenize_prompt_and_output` against the
snapshot shapes (3, 9).

### 2.2 `compute_entropy` (1 point) — ✅ verified

Per-token entropy `H(p) = −Σ_v p_v log p_v` (handout Eq. 1) computed in a numerically stable
way as `(softmax(logits) * log_softmax(logits)).sum(-1)` negated — `logsumexp`-based
`log_softmax` avoids overflow on large-vocab logits. Verified by
`tests/test_sft.py::test_compute_entropy`, shape (2, 10) from logits (2, 10, 100).

### 2.3 `get_response_log_probs` (2 points) — ✅ verified

One forward pass; `log_probs[:, t] = log_softmax(logits[:, t])[labels[:, t]]` via gather at
the (already-shifted) `labels`. With `return_token_entropy=True` it additionally returns
per-token entropy from §2.2 over the same logits — no second forward pass. Returns a dict
`{"log_probs", "token_entropy"}`. Verified by `tests/test_sft.py::test_get_response_log_probs`.

### 2.4 `masked_normalize` (1 point) — ✅ verified

`sum(tensor * mask, dim) / normalize_constant`: masked-out positions contribute zero. With
`normalize_constant = 1.0` and `dim=None` this is the raw masked sum used by the SFT loss;
with a different constant it is the length-normalization variant discussed in §5.3/§5.4.
Verified by the four `tests/test_sft.py::test_masked_normalize_*` cases (dim 0/1/−1/None;
`normalize_constant` 42.0 in the fixtures).

### 2.5 `sft_microbatch_train_step` (3 points) — ✅ verified

Response-only SFT loss = negative masked sum of response-token log-probs divided by
`normalize_constant` (handout: keep the constant explicit; default 1.0), then
`/ gradient_accumulation_steps`, then `loss.backward()` — the caller decides when to step the
optimizer and zero grads, so repeated calls **accumulate** gradients (this is exactly what the
10-step test variant checks: stacked losses (10,) and stacked `.grad`s (10, 2, 10) match a
run that never zeroes). Returns the scalar loss and metadata for logging. Verified by the
three `tests/test_sft.py::test_sft_microbatch_train_step*` cases, including the grad-vs-loss
consistency that pins the sign (maximize log-probs ⇒ negate) and the accumulation scaling.

### 2.6 `log_generations` (1 point) — ✅ verified, with design rationale

Implemented at `cs336_alignment/sft.py::log_generations` and wired into every training run's
validation phase (`val_generations_step<N>.jsonl` next to `metrics.jsonl`).

**What it logs and why each field earns its place** (the handout lists 1–6; the rationale for
each):

1. **The input prompt.** RL/SFT bugs are frequently data bugs — a mis-formatted template or a
   truncated question is invisible in aggregate metrics but obvious in the raw prompt.
2. **The response generated by the model.** The single most informative debugging artifact:
   catches degenerate loops, missing `<think>`/`</think>` structure, and answers that drift
   out of the `<answer>` tags long before reward curves explain why.
3. **The ground-truth answer.** Without it, a human cannot tell a grading bug from a model
   error when spot-checking.
4. **The reward information (format, answer, total).** The decomposition matters: "format 0"
   is a template/parsing failure (the model may well know the answer), while "format 1,
   answer 0" is a reasoning failure. The two have different fixes (fix the prompt/parser vs.
   train longer), and the same three buckets the MATH baseline analysis uses (§1b) are what
   you want to watch move during training.
5. **The average token entropy of the response.** Entropy is the earliest warning light in
   RL runs: collapse to near-zero entropy means premature convergence/repetition (the model
   commits to one chain-of-thought shape); exploding entropy means the updates are
   destabilizing the policy. The handout explicitly asks for entropy plots in the EI and
   off-policy experiments (§3, §5.7), and this field is where those plots come from.
6. **Average response length overall, for correct responses, and for incorrect responses.**
   This is the direct probe for length/credit-attribution artifacts (§5.3): if incorrect
   responses get systematically longer (rambling), a length-normalization choice that weights
   per token will amplify exactly the wrong examples. It also exposes reward hacking of
   response-length bonuses, should any be introduced.

A note on what *not* to read as a metric: the policy-gradient "loss" is just a scalar whose
gradient is useful — the handout says plainly that `pg_loss` "is not a loss in the canonical
sense" and that train/validation **rewards** are the meaningful metrics. `log_generations` is
the mechanism that makes per-example rewards inspectable rather than only aggregate.

### 2.7 `sft_experiment` (2 points, 2 H100 hrs) — 📊 [CLUSTER RUN PENDING]

Implemented at `scripts/train_sft.py`: AdamW with warmup + cosine decay, gradient
accumulation, grad clipping 1.0, periodic MATH-validation evals through the vLLM instance on
the second GPU (`--vllm-device cuda:1`), `metrics.jsonl` + `val_generations_step<N>.jsonl`
outputs. Dataset-size ablation via `--max-examples`; correct-only filtering via
`--correct-only`.

Cluster runs (from the [README runbook](./README.md#sft-on-math-reasoning-traces-sft_experiment-algorithm-1)):

```sh
# Full run
uv run python scripts/train_sft.py \
    --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
    --data-path /data/a5-alignment/MATH/sft.jsonl \
    --output-dir outputs/sft
# Dataset-size ablation, e.g. 25% of examples (repeat with other sizes: 128/256/512/1024)
uv run python scripts/train_sft.py ... --max-examples $(( $(wc -l < /data/a5-alignment/MATH/sft.jsonl) / 4 ))
# Correct-only filtered variant
uv run python scripts/train_sft.py ... --correct-only
```

Output: learning curves in `outputs/sft*/metrics.jsonl`, checkpoints in `outputs/sft*/final`.

| Dataset size | Final validation accuracy | Learning curve |
|---|---|---|
| 128 | [CLUSTER RUN PENDING] | `outputs/sft_128/metrics.jsonl` |
| 256 | [CLUSTER RUN PENDING] | `outputs/sft_256/metrics.jsonl` |
| 512 | [CLUSTER RUN PENDING] | `outputs/sft_512/metrics.jsonl` |
| 1024 | [CLUSTER RUN PENDING] | `outputs/sft_1024/metrics.jsonl` |
| full | [CLUSTER RUN PENDING] | `outputs/sft/metrics.jsonl` |

Correct-only variant: filtered dataset size **[CLUSTER RUN PENDING]**, final validation
accuracy **[CLUSTER RUN PENDING]**, comparison to the unfiltered run **[CLUSTER RUN PENDING]**
(the handout expects filtering to help; the size of the filtered set will quantify how much
of the R1-distilled data the grader accepts as correct).

## 3. `expert_iteration_experiment` (2 points, 6 H100 hrs) — 📊 [CLUSTER RUN PENDING]

Implemented at `scripts/expert_iteration.py` (Algorithm 2: sample G rollouts per question,
keep reward-1.0 generations as the SFT buffer, SFT from the current policy, repeat for
`n_ei_steps = 5`; vLLM stops at `</answer>`, `min_tokens=4` to avoid empty responses; grad
clipping 1.0).

Cluster run (from the [README runbook](./README.md#expert-iteration-expert_iteration_experiment-algorithm-2)):

```sh
uv run python scripts/expert_iteration.py \
    --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
    --output-dir outputs/expert_iteration
```

Output: per-iteration checkpoints in `outputs/expert_iteration/step_<N>/`, SFT-buffer stats +
validation accuracy + token entropy in `metrics.jsonl`.

| Rollout config (G × SFT epochs × batch size) | Validation accuracy curve |
|---|---|
| G=8 × 1 epoch × 512 | [CLUSTER RUN PENDING] |
| G=16 × 1 epoch × 1024 | [CLUSTER RUN PENDING] |
| (≥2 rollout counts and ≥2 epoch counts required) | [CLUSTER RUN PENDING] |

- Model reaching ≥15% validation accuracy: [CLUSTER RUN PENDING]
- 2-sentence comparison vs SFT performance and across EI steps: [CLUSTER RUN PENDING]
- Entropy plot over training: [CLUSTER RUN PENDING] (from the logged token entropies in
  `metrics.jsonl`; EI should push entropy down on the filtered distribution).

## 4. Policy gradients and GRPO — implementation (✅ all verified)

All losses follow the handout equations; shapes and semantics were pinned by the 14 GRPO
tests.

### 4.1 `compute_group_normalized_rewards` (2 points) — ✅ verified

Rewards come from the reward fn per rollout; groups are **consecutive** slices of the rollout
batch (ground truths repeated `group_size` times), so rewards reshape to `(n_groups,
group_size)`. Advantages (handout Eq. 28): `A(i) = (r(i) − mean_group) / (std_group +
advantage_eps)` when `normalize_by_std=True`; the Dr. GRPO variant (Eq. 31) drops the std
denominator entirely when False — `advantage_eps` is **inside** the denominator and only in
the std variant. Returns `(advantages, raw_rewards, metadata)` — three values, as both tests
unpack. Verified by the two `tests/test_grpo.py::test_compute_group_normalized_rewards_*`
cases against snapshots (8 rollouts, group size 4).

### 4.2 `compute_naive_policy_gradient_loss` (1 point) — ✅ verified

Per-token loss `−A(i) · log πθ(o_t | q, o_<t))` (handout Eq. 32): the (batch, 1)
reward/advantage broadcasts over the sequence; no masking at this level. Verified by
`tests/test_grpo.py::test_compute_naive_policy_gradient_loss`.

### 4.3 `compute_grpo_clip_loss` (2 points) — ✅ verified

Per-token loss `−min(r·A, clip(r, 1−ε, 1+ε)·A)` with `r = exp(log πθ − log π_old)` (handout
Eq. 33). The ratio is computed as the exponential of the **log-space difference** (never
`exp(a)/exp(b)`), which keeps cliprange 10.0 overflows out of float range. Metadata carries
the per-token "was clipped" flag (min chose the clipped branch), from which the clip fraction
is derived for logging. Verified by the two `test_compute_grpo_clip_loss_*` cases
(cliprange 10.0 and 0.1; the fixtures make both clip branches fire).

### 4.4 `compute_policy_gradient_loss` (1 point) — ✅ verified

Dispatch wrapper: `no_baseline` → naive loss on `raw_rewards` (A = R(q,o)); 
`reinforce_with_baseline` → naive loss on `advantages` (group-normalized); `grpo_clip` →
clip loss with `old_log_probs`. Validates that the arguments each loss type needs are present
and merges the underlying metadata. Verified by the three
`test_compute_policy_gradient_loss_*` cases.

### 4.5 `masked_mean` (1 point) — ✅ verified

`sum(tensor · mask, dim) / sum(mask, dim)` with `tensor.mean(dim)` shape semantics (dim
0/1/−1, or a scalar over all unmasked elements when `dim=None`). Verified by the four
`test_masked_mean_*` cases. **Why this is a hyperparameter and not a detail:** see §5.3.

### 4.6 `grpo_microbatch_train_step` (3 points) — ✅ verified

Composes §4.4 → `masked_mean(per_token_loss, response_mask, dim=1)` (per-example) → batch
mean → `/ gradient_accumulation_steps` → `backward()`. As in the SFT step, the caller owns
optimizer stepping and zeroing; the 10-step test variant confirms gradients accumulate.
Verified by the two `test_grpo_microbatch_train_step*` cases (`grpo_clip` loss type,
cliprange 0.1).

### 4.7 `grpo_train_loop` (5 points) — ✅ implemented; 📊 run pending

Implemented at `scripts/train_grpo.py`: the full Algorithm 3 loop with the handout's default
hyperparameter block (200 GRPO steps, AdamW lr 1e-5, betas (0.9, 0.95), weight decay 0,
rollout batch 256, group size 8, temperature 1.0, min tokens 4, max tokens 1024, grad
accumulation 128 ⇒ microbatch 2, `reinforce_with_baseline`, std normalization on) and the
handout's sanity asserts enforced in config validation (train batch divisible by grad-accum
steps, rollout batch divisible by group size, train batch ≥ group size). Per-update logging
covers loss, grad norm, token entropy, clip fraction (off-policy), and train reward means
(total/format/answer); validation evals every `--eval-every 10` steps on ≥1024 examples.

Cluster run (from the [README runbook](./README.md#grpo-grpo_train_loop--section-8-experiments)):

```sh
uv run python scripts/train_grpo.py \
    --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
    --output-dir outputs/grpo \
    --vllm-device cuda:1
```

📊 **[CLUSTER RUN PENDING]** — validation reward plot w.r.t. steps and a few example rollouts
over time will be attached from `outputs/grpo/metrics.jsonl` and
`outputs/grpo/val_generations_step<N>.jsonl`.

| Metric | Value |
|---|---|
| Final validation answer reward | [CLUSTER RUN PENDING] |
| Example rollouts (start / mid / end of training) | [CLUSTER RUN PENDING] |

## 5. GRPO experiments (Section 8 of the handout)

All runs below share the §4.7 defaults with one knob varied at a time; each takes 2 GPUs
(policy + vLLM). The handout's note on stopping early applies: configs that visibly diverge
or stall get stopped before 200 steps.

### 5.1 `grpo_learning_rate` (2 points, 6 H100 hrs) — 📊 [CLUSTER RUN PENDING]

```sh
# sweep: e.g. --learning-rate 1e-6 | 3e-6 | 1e-5 | 3e-5 | 1e-4
uv run python scripts/train_grpo.py ... --learning-rate <lr> --output-dir outputs/grpo_lr_<lr>
```

| Learning rate | Final validation answer reward | Notes (divergence etc.) |
|---|---|---|
| 1e-6 | [CLUSTER RUN PENDING] | |
| 3e-6 | [CLUSTER RUN PENDING] | |
| 1e-5 | [CLUSTER RUN PENDING] | handout default |
| 3e-5 | [CLUSTER RUN PENDING] | |
| 1e-4 | [CLUSTER RUN PENDING] | |

- Model with ≥25% MATH validation accuracy: [CLUSTER RUN PENDING]
- 2-sentence discussion of other logged metrics: [CLUSTER RUN PENDING] (grad norm and token
  entropy are the leading indicators to watch for the diverging configs).

### 5.2 `grpo_baselines` (2 points, 2 H100 hrs) — 📊 [CLUSTER RUN PENDING]

On-policy comparison of `no_baseline` vs `reinforce_with_baseline` (std normalization on,
tuned LR):

```sh
uv run python scripts/train_grpo.py ... --loss-type no_baseline --output-dir outputs/grpo_no_baseline
uv run python scripts/train_grpo.py ... --loss-type reinforce_with_baseline --output-dir outputs/grpo_reinforce
```

| Loss type | Validation reward curve |
|---|---|
| no_baseline | [CLUSTER RUN PENDING] |
| reinforce_with_baseline | [CLUSTER RUN PENDING] |

2-sentence discussion of other logged metrics: [CLUSTER RUN PENDING]. (Expectation stated in
advance, to be checked against the curves: group-normalized advantages exist precisely to
reduce gradient variance, so the baselined run should be the more stable of the two on grad
norm.)

### 5.3 `think_about_length_normalization` (1 point) — ✅ answered in full

**The two aggregation rules.** Both operate on per-token losses `−A(i) · r_t` where the
advantage `A(i)` is broadcast over the sequence (handout p. 30):

- **`masked_mean`** divides each sequence's summed loss by that sequence's own number of
  unmasked tokens `|o(i)|`, then averages over the batch. In the handout's worked example
  (batch of 2; responses of 4 and 7 tokens; advantage 2 everywhere), each element comes out
  as 2.0, and the per-token gradients are **0.25** in the 4-token response vs **0.1429** in
  the 7-token response: every *sequence* contributes the same total gradient mass
  (4 × 0.25 = 7 × 0.1429 = 1.0) regardless of its length.
- **`masked_normalize` with a constant normalizer** (the handout's example uses the max
  generation length 7) divides by a length-independent constant. The per-token gradients are
  **0.1429 everywhere**: every *token* in the batch gets the same learning signal, so the
  7-token response contributes 7/4 ≈ 1.75× the total gradient of the 4-token one.

This matches Eq. 29 as written: the GRPO-Clip objective of DeepSeekMath carries an explicit
`1/|o(i)|` factor — the `masked_mean` choice reproduces that estimator, while the constant
normalizer reproduces the plain REINFORCE sum-the-tokens estimator (handout Eq. 26) that Dr.
GRPO advocates.

**Pros and cons.**

- `masked_mean` (divide by `|o(i)|`):
  - *Pro:* each rollout influences the update equally, so a question that happens to elicit a
    long response cannot dominate the batch; the estimator is scale-invariant to response
    length, which keeps the effective per-example learning rate constant.
  - *Con:* credit per token is diluted in long responses — a long, fully correct
    chain-of-thought is trained with the same total force as a short one, so long-horizon
    reasoning (exactly what MATH rewards) is under-trained relative to short answers. This is
    the length bias Dr. GRPO (Liu et al., 2025) objects to: normalizing per response
    effectively boosts short responses and suppresses long ones, which can push the policy
    toward brevity independent of correctness.
  - *Better when:* response lengths vary widely and you want length kept out of the gradient;
    when matching Eq. 29 exactly matters; when long responses are often degenerate (rambling
    chains) and should not get extra gradient mass.
- `masked_normalize` with a constant (token-uniform weighting):
  - *Pro:* credit is uniform per token — the natural estimator if every generated token is
    equally worth learning from; avoids the per-sequence length bias; longer correct
    reasoning chains receive proportionally more gradient, which is arguably the right
    credit-attribution for chain-of-thought distillation.
  - *Con:* long responses dominate the update; if long generations correlate with being
    wrong (over-thinking, repetition loops), the constant normalizer amplifies noise precisely
    on the examples you'd rather down-weight, and the gradient magnitude then depends on the
    length distribution of each batch (a batch of short answers yields a smaller update than
    a batch of long ones at the same LR).
  - *Better when:* lengths are comparable across the batch; when you deliberately want longer
    chains to carry more learning signal; in Dr. GRPO-style training that removes
    normalization artifacts (the empirical arm of §5.4 pairs this with the no-std advantages
    of §5.5, both from Liu et al., 2025).

**Specific settings.** Two places the choice visibly matters: (1) groups containing one very
short and one very long response for the same question — `masked_mean` gives the group's
advantage the same total weight whichever response wins, while the constant normalizer lets
the long one move the policy more; (2) mixed-difficulty batches where easy questions get short
answers and hard questions get long ones — per-response mean turns this into an
easy-question-heavy update, token-uniform weighting into a hard-question-heavy one. Which is
"better" is an empirical question — that is what §5.4 settles with end-to-end runs; the
handout's hint points at gradient norm as the stability metric to watch.

### 5.4 `grpo_length_normalization` (2 points, 2 H100 hrs) — 📊 [CLUSTER RUN PENDING]

```sh
uv run python scripts/train_grpo.py ... --length-norm masked_mean --output-dir outputs/grpo_len_mean
uv run python scripts/train_grpo.py ... --length-norm masked_normalize --output-dir outputs/grpo_len_norm
```

| Length normalization | Validation answer reward curve | Grad norm / stability metrics |
|---|---|---|
| masked_mean (per-response) | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| masked_normalize (constant) | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |

Commentary on findings: [CLUSTER RUN PENDING]. The advance prediction from §5.3 to check:
`masked_normalize` should show larger and length-correlated gradient norms (long responses
dominate), `masked_mean` flatter ones; response-length distributions per correctness bucket
(from `log_generations`, §2.6) are the diagnostic that ties the curves to the mechanism.
The better-performing variant is fixed for all subsequent experiments.

### 5.5 `grpo_group_standard_deviation` (2 points, 2 H100 hrs) — 📊 [CLUSTER RUN PENDING]

```sh
uv run python scripts/train_grpo.py ... --use-std-normalization --output-dir outputs/grpo_std_on
uv run python scripts/train_grpo.py ... --no-std-normalization --output-dir outputs/grpo_std_off
```

(Dr. GRPO advantages, Eq. 31 — division by the group std removed; both arms use the §5.4
winner.)

| use_std_normalization | Validation answer reward curve | Grad norm / stability |
|---|---|---|
| True (Eq. 28) | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| False (Eq. 31, Dr. GRPO) | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |

Commentary on findings: [CLUSTER RUN PENDING]. The mechanism to check against the curves:
with std normalization, questions whose group rewards are nearly all-0 or all-1 (too easy or
too hard) get their tiny within-group differences inflated by the small denominator, so those
questions receive *higher* effective weight — Dr. GRPO removes the std to kill that bias.
The false arm should therefore show the flatter gradient norms on mixed-difficulty batches.

### 5.6 `grpo_off_policy` — ✅ implemented

Off-policy support is in `scripts/train_grpo.py`: `--epochs-per-rollout-batch` and
`--train-batch-size` control gradient epochs/updates per rollout batch (the handout's three
knobs), old log-probs are computed once per rollout batch under `torch.inference_mode()`
before the inner loop and reused across epochs (never differentiated through), and switching
to `--loss-type grpo_clip` engages the clipped objective that off-policy training requires.
The handout's caution is honored: GRPO-Clip is only used when off-policy, since the on-policy
losses need no `old_log_probs`. The loss machinery itself is the §4.3/§4.6 code the suite
verifies.

### 5.7 `grpo_off_policy_sweep` (4 points, 12 H100 hrs) — 📊 [CLUSTER RUN PENDING]

Protocol (fixed now; numbers pending): with `rollout_batch_size = 256` and
`gradient_accumulation_steps` re-tuned per config to keep memory constant (microbatch 2, per
the handout hint), sweep `epochs_per_rollout_batch ∈ {1, 2, 4}` ×
`train_batch_size ∈ {64, 128, 256}` — broad sweep first at <50 GRPO steps, then a focused
subset at 200 steps. Wall-clock is logged in `metrics.jsonl` for every step, so the
validation-vs-steps and validation-vs-wall-clock plots come from the same file.

```sh
uv run python scripts/train_grpo.py ... --epochs-per-rollout-batch <E> --train-batch-size <B> \
    --loss-type grpo_clip --output-dir outputs/grpo_off_e<E>_b<B>
```

| Config (epochs × train batch) | Val reward curve (steps) | Val reward curve (wall-clock) | Entropy / response length trend |
|---|---|---|---|
| 1 × 256 (on-policy reference) | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| 2 × 128 | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| 4 × 64 | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |

Experiment log explaining chosen ranges: [CLUSTER RUN PENDING]. Comparison to EI entropy
(from §3): [CLUSTER RUN PENDING] — the quantity to compare is the token-entropy trajectory;
off-policy GRPO repeated over the same rollouts should show faster entropy decay per wall
clock than EI's SFT-on-filtered-outputs loop.

### 5.8 `grpo_off_policy_clip_ablation` (2 points, 2 H100 hrs) — ✅ loss type implemented; 📊 run pending

`grpo_no_clip` (unclipped per-token loss, handout Eq. 34) is implemented as a loss type in
`cs336_alignment/grpo.py` and selectable via `--loss-type grpo_no_clip`; the wrapper rejects
it without the arguments it needs, and the tests cover the clip-loss path it ablates.

```sh
uv run python scripts/train_grpo.py ... --loss-type grpo_no_clip --epochs-per-rollout-batch <best> \
    --train-batch-size <best> --output-dir outputs/grpo_no_clip
```

| Loss type (same off-policy config) | Val reward curve | Entropy / length / grad norm |
|---|---|---|
| grpo_clip | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| grpo_no_clip | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |

Commentary: [CLUSTER RUN PENDING]. The trade the run measures: clipping exists to keep the
policy near `π_old` when many gradient steps hit one rollout batch; removing it should (if
clipping is doing its job) show up as larger policy drift per batch — visible as faster
entropy collapse and noisier grad norms — and possibly faster early reward gains that
plateau or crash.

### 5.9 `grpo_prompt_ablation` (2 points, 2 H100 hrs) — explanation pre-answered; 📊 run pending

```sh
uv run python scripts/train_grpo.py ... --prompt-template r1_zero --output-dir outputs/grpo_prompt_r1
uv run python scripts/train_grpo.py ... --prompt-template question_only \
    --output-dir outputs/grpo_prompt_qonly
```

(`--prompt-template question_only` swaps the reward fn automatically: the script's
`PROMPT_SPECS` pairs `question_only.prompt` with `question_only_reward_fn` from
`cs336_alignment/drgrpo_grader.py`, used for both training and validation in that arm,
per the handout.)

| Prompt | Validation answer reward curve | Entropy / length / grad norm trends |
|---|---|---|
| r1_zero | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| question_only | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |

Explanation (from the handout's own analysis, to be confirmed by the runs): the R1-Zero
prompt is *not* the best prompt for final performance on this model — there is a mismatch
between the prompt and how Qwen 2.5 Math 1.5B was pretrained. Liu et al. (2025) found that
simply prompting with the bare question starts at very high accuracy, matching the R1-Zero
prompt's post-RL accuracy before any training — strong evidence that Qwen 2.5 Math was
pretrained on plain question–answer pairs of this shape. The R1-Zero prompt is retained for
the assignment's main experiments because RL with it shows clear accuracy improvements in few
steps — the visible upward curve that makes the RL mechanics easy to sanity-check — even
though the question-only prompt likely starts and possibly ends higher. The ablation curves
quantify exactly this gap; the entropy/response-length trends should show the question-only
arm producing shorter, more confident outputs from step 0.

## 6. `leaderboard` (16 points, 16 H100 hrs) — 📊 [CLUSTER RUN PENDING]

Constraints (restated from the handout): report validation accuracy averaged over the entire
5K MATH validation set, R1-Zero prompt at validation time, temperature 1.0 and max tokens
1024 with vLLM, accuracy = mean answer reward of `r1_zero_reward_fn`; no extra data, no SFT
on stronger-model reasoning traces; training-time algorithm/hyperparameter changes (including
a training reward fn and train-set filtering/curricula) are allowed.

```sh
uv run python scripts/train_grpo.py ... --leaderboard --output-dir outputs/grpo_leaderboard
```

| Result | Value |
|---|---|
| Validation accuracy (≤4 h, 2×H100) | [CLUSTER RUN PENDING] |
| Screenshot: accuracy vs wall-clock (x-axis ≤ 4 h) | [CLUSTER RUN PENDING] |

Planned approach to fill in the number (to be validated by the run): keep the §5.4/§5.5
winning normalization stack, the tuned LR from §5.1, the better of on-policy vs the §5.7
off-policy point per wall-clock, train-set filtering, and the systems optimizations the
handout suggests (the two-GPU pipeline leaves one GPU idle between rollout and training
phases; overlapping them, or colocating vLLM with the policy with careful memory budgeting,
is where the wall-clock win is).

## 7. On KL divergence — ✅ answered in full

**What the KL term is for.** In the RLHF setup the supplement describes (§5 of the
supplement), the reward model is *learned* — fit to human preference rankings — and then
optimized by RL. Optimizing hard against a learned reward model invites the policy to drift
onto inputs the reward model scores highly but humans would not: reward hacking, and more
generally degeneration of the behaviors the SFT initialization had. The original RLHF recipe
therefore adds (a) a per-token KL-divergence penalty against the frozen SFT model, keeping
the policy close to the distribution that produced the preference data, and (b) an auxiliary
pretraining loss. DPO contains the same force in its parameterization: the implied reward is
`r(x, y) = β log(π(y|x)/π_ref(y|x)) + β log Z(x)` (supplement Eq. 2), where β is explicitly
"the hyperparameter controlling the strength of the penalty for deviating from π_ref".

**Why an overly large KL penalty harms learning.** The penalty is a leash to the reference
policy. With objective `E[reward] − β·KL(πθ ‖ π_ref)`, the KL gradient competes directly with
the reward gradient at every token; as β grows, the leash shortens. In the limit the optimal
solution is the reference policy itself — perfectly regularized, learning nothing. Short of
that limit the failure is gradual but directional: the reward signal per unit of policy
movement shrinks (movement away from the reference is taxed), so learning slows and can stall
entirely on behaviors that require leaving the reference distribution — which is the entire
point of training. A too-large β also amplifies a second problem in RLHF-style setups: the
policy is anchored to the SFT model's own quirks and biases, so alignment can only reshape
behavior the reference already had mass on. (The same knob appears in DPO: too-large β makes
the implicit-reward differences `β·Δ` large and saturated under the sigmoid, so the per-instance
loss stops distinguishing good updates from bad ones; too-small β removes the tether and the
policy can drift — the assignment's own DPO training uses the modest β = 0.1 for exactly this
reason.)

**How this assignment's GRPO setup handles it.** It omits the KL term entirely. The handout's
GRPO is explicitly the special case of DeepSeekMath's GRPO "with a verified reward function,
no KL term, and no iterative update of the reference and reward model" (handout §7.1,
fn. 2), and the "On KL divergence" note before the leaderboard reports that in the staff's
experiments — and others' in the literature (Liu et al., 2025) — omitting KL had *no impact
on performance* while saving the GPU memory a reference model would cost. The omission is
safe here for the reason the leash exists elsewhere: with a verified, ground-truth reward
function there is no learned reward model to hack — the scalar being maximized is the true
objective (answer correctness under the fixed grader) — and the training regime is short-horizon
(200 steps), small-LR AdamW with gradient clipping, on one narrow domain. The regularization
the KL term would supply is instead carried by the verified reward plus the conservative
optimization recipe. The handout still leaves the door open — experimenting with KL or other
regularization is allowed on the leaderboard — and §5's metrics (entropy, response length,
grad norm) are the diagnostics that would reveal drift if a KL-free run went off the rails.

---

# Part II — Supplement: instruction tuning and RLHF

## 8. Zero-shot baselines (Llama 3.1 8B)

All four baselines run through one script, `scripts/eval_benchmark.py
--benchmark {mmlu,gsm8k,alpaca_eval,simple_safety_tests}` (README: "Zero-shot baselines"),
with the supplement's system prompt and per-task prompt formats, greedy decoding
(temperature 0.0, top-p 1.0), stop string `# Query:` for the chat-style tasks, and metrics
printed as a JSON block. Predictions are serialized in the formats the annotators consume
(JSON array for AlpacaEval, JSONL with `prompts_final`/`output` for SimpleSafetyTests).
The `_sft` variants (§10) are the same commands with `--model-path` pointed at the
instruction-tuned checkpoint and `--use-alpaca-template` so inputs match the training format.

### 8.1 Parsers (✅ both verified)

- **MMLU** (`cs336_alignment/data.py::parse_mmlu_response`): extracts the predicted option
  letter A–D from the model output. It matches the letter *as an answer choice*, not as part
  of some larger token — the guard the test suite pins: "The correct answer is B. There is
  only one human polyomavirus…" parses to `"B"`, while "The correct answer is 10000
  polyomaviruses." must parse to nothing (a bare number must never be read as an option
  letter). Unparseable outputs return `None`. Verified by the two
  `tests/test_metrics.py::test_parse_mmlu_response*` cases.
- **GSM8K** (`cs336_alignment/data.py::parse_gsm8k_response`): takes the **last** number in
  the output, as a string — "…Natalia sold 48+24 = 72 clips altogether…" → `"72"`; spelled-out
  numbers ("seventy-two") parse to `None` by design (digit sequences only). The gold side of
  `data/gsm8k/*.jsonl` strips to the number after `####` when grading. Verified by the two
  `tests/test_metrics.py::test_parse_gsm8k_response*` cases.

Both parsers were additionally smoke-checked against the repo's shipped data (100% parse
rate on GSM8K/MMLU samples drawn from `data/gsm8k/test.jsonl` and an MMLU subject CSV).

### 8.2 MMLU baseline — 📊 [CLUSTER RUN PENDING]

```sh
uv run python scripts/eval_benchmark.py --benchmark mmlu \
    --model-path /data/a5-alignment/models/Llama-3.1-8B --output-dir outputs/mmlu_baseline
```

Predictions: `outputs/mmlu_baseline/mmlu_predictions.jsonl`.

| Quantity | Value |
|---|---|
| Accuracy (57 subjects) | [CLUSTER RUN PENDING] |
| Parse failures (count + examples) | [CLUSTER RUN PENDING] |
| Throughput (examples/second) | [CLUSTER RUN PENDING] |
| Error analysis (10 random wrong predictions) | [CLUSTER RUN PENDING] |

### 8.3 GSM8K baseline — 📊 [CLUSTER RUN PENDING]

```sh
uv run python scripts/eval_benchmark.py --benchmark gsm8k \
    --model-path /data/a5-alignment/models/Llama-3.1-8B --output-dir outputs/gsm8k_baseline
```

Predictions: `outputs/gsm8k_baseline/gsm8k_predictions.jsonl`.

| Quantity | Value |
|---|---|
| Accuracy (1,319 test examples shipped in `data/gsm8k/test.jsonl`) | [CLUSTER RUN PENDING] |
| Parse failures (count + examples) | [CLUSTER RUN PENDING] |
| Throughput (examples/second) | [CLUSTER RUN PENDING] |
| Error analysis (10 random wrong predictions) | [CLUSTER RUN PENDING] |

### 8.4 AlpacaEval baseline — 📊 [CLUSTER RUN PENDING]

Generation:

```sh
uv run python scripts/eval_benchmark.py --benchmark alpaca_eval \
    --model-path /data/a5-alignment/models/Llama-3.1-8B --output-dir outputs/alpaca_eval_baseline
```

Judgment (805 prompts shipped in `data/alpaca_eval/alpaca_eval.jsonl`; Llama 3.3 70B
Instruct annotator vs the GPT-4 Turbo reference; README: "Annotators"):

```sh
uv run alpaca_eval --model_outputs outputs/alpaca_eval_baseline/alpaca_eval_predictions.json \
    --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn --base-dir .
```

| Quantity | Value |
|---|---|
| Winrate vs GPT-4 Turbo | [CLUSTER RUN PENDING] |
| Length-controlled winrate | [CLUSTER RUN PENDING] |
| Generation throughput (examples/second) | [CLUSTER RUN PENDING] |
| Error analysis of 10 dispreferred outputs (agreement with the annotator) | [CLUSTER RUN PENDING] |

### 8.5 SimpleSafetyTests baseline — 📊 [CLUSTER RUN PENDING]

Generation, then 70B judgment (README: "Annotators"):

```sh
uv run python scripts/eval_benchmark.py --benchmark simple_safety_tests \
    --model-path /data/a5-alignment/models/Llama-3.1-8B --output-dir outputs/sst_baseline

uv run python scripts/evaluate_safety.py \
    --input-path outputs/sst_baseline/simple_safety_tests_predictions.jsonl \
    --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \
    --num-gpus 2 \
    --output-path outputs/sst_baseline/simple_safety_tests_annotations.jsonl
```

| Quantity | Value |
|---|---|
| Proportion judged safe (Llama 3.3 70B annotator) | [CLUSTER RUN PENDING] |
| Generation throughput (examples/second) | [CLUSTER RUN PENDING] |
| Error analysis of 10 unsafe-judged outputs (agreement with the annotator) | [CLUSTER RUN PENDING] |

## 9. Instruction fine-tuning

### 9.1 `look_at_sft` (4 points) — 🔍 answered from the public download

The supplement's instruction-tuning training file
(`safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz`, the public nlp.stanford.edu
URL the supplement footnotes) was downloaded and inspected directly; the cluster copy at
`/data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz` is the same
artifact. It contains **210,348** single-turn `{prompt, response}` examples (count measured
on the download; the dataset is UltraChat-200K prompts mixed with SafetyTunedLlamas
safety-augmented data, per the supplement).

Ten random examples (seed 0) implicitly cover: **reading comprehension / extraction QA**
(answer a question strictly from a provided news snippet — one of the sampled prompts carries
UltraChat's synthetic wrapper "Generate response to the question/instruction based on a piece
of given material"), **structured writing** (a report on mobile technology and travel; a
comprehensive product review of garden clogs with sizing/durability/comfort sections), 
**step-by-step procedural how-tos** (create and configure a Discord server with roles and
permissions; make fig jam with ingredients/tools/storage), **open-domain factual QA**
(how ocean currents affect weather patterns), **creative writing** (a children's book about
animals saving their forest), **design/formatting tasks** (a visually modern recipe-book
layout for organic energy bars), **marketing/analytics advice** (an SEO + social media
strategy guide), and **news summarization** (condense a crime-and-sentencing news story).

Quality commentary: the responses are well-organized, Markdown-friendly, and on-format
(headings, numbered steps), which is what SFT on this data teaches a base model; the
weaknesses sit on the prompt side and in specificity. Two of ten prompts were partly
synthetic boilerplate (the "given material" wrapper; the recipe-book brief reads
machine-generated with its long comma-chained constraint lists). Responses hedge when the
evidence is absent (the fundraiser answer correctly notes the goal "was not explicitly
stated" in the material) but elsewhere assert specifics — names, figures like "$28,000" —
that would need verification for factual reliability. Nothing in the sample was abusive or
unsafe, consistent with the safety-augmented mix; overall it reads as broad, competent,
occasionally synthetic instruction-following data rather than deep domain knowledge.

### 9.2 `data_loading` (3 points) — ✅ verified

`cs336_alignment/data.py::PackedSFTDataset` (+ the `get_packed_sft_dataset` factory):
tokenizes each Alpaca-templated document (BOS-prefixed), appends the EOS token (`<|end_of_text|>`,
id 128001 for Llama 3) as the delimiter, concatenates documents in order (shuffled first when
`shuffle=True`), cuts consecutive non-overlapping chunks of `seq_length` (dropping the final
partial chunk, per the handout's `[0..10]`, m=4 → `[[0,1,2,3],[4,5,6,7]]` example), and
returns `{"input_ids", "labels"}` with `labels` the left-shifted `input_ids`
(`labels[k] = input_ids[k+1]`). `iterate_batches` wraps `torch.utils.data.DataLoader`
(batch size, shuffle; `len()` supported). Verified by `tests/test_data.py` — exactly 75
packed examples of length 32 from the 5-document fixture, int64 labels, `ceil(75/8) = 10`
batches with a short last batch allowed, and `shuffle=True` producing a different packing.

### 9.3 `sft_script` + `sft` (4 + 6 points, 24 H100 hrs) — ✅ script; 📊 run pending

`scripts/train_instruction_sft.py` fine-tunes Llama 3.1 8B base on the packed single-turn
data: configurable model/optimizer hyperparameters, gradient accumulation (512-token
sequences, microbatch 2, effective batch 32), periodic train/validation logging
(`metrics.jsonl` + generation dumps), checkpoints saved with the tokenizer via
`.save_pretrained`. The handout-recommended configuration is the default: 1 epoch, context
length 512, total batch 32, learning rate 2e-5, cosine decay with 3% linear warmup.

```sh
uv run python scripts/train_instruction_sft.py \
    --output-dir outputs/instruction_sft \
    --attn-implementation flash_attention_2
```

📊 **[CLUSTER RUN PENDING]**

| Quantity | Value |
|---|---|
| Final validation loss | [CLUSTER RUN PENDING] |
| Learning curve (train/val loss) | `outputs/instruction_sft/metrics.jsonl` |
| Training-setup description (anything deviating from the handout defaults above) | [CLUSTER RUN PENDING] |

Model + tokenizer serialize to `outputs/instruction_sft/final` for the §10 evaluations and
the §12 DPO stage.

## 10. Evaluating the instruction-tuned model — 📊 all [CLUSTER RUN PENDING]

Same four benchmarks as §8, same prompts and generation settings, `--model-path
outputs/instruction_sft/final --use-alpaca-template` (README: "To evaluate an
instruction-tuned checkpoint instead…").

### 10.1 MMLU (`mmlu_sft`)

```sh
uv run python scripts/eval_benchmark.py --benchmark mmlu \
    --model-path outputs/instruction_sft/final --use-alpaca-template \
    --output-dir outputs/mmlu_sft
```

| Quantity | SFT model | vs zero-shot (§8.2) |
|---|---|---|
| Throughput (examples/second) | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| Accuracy | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| Error analysis (10 wrong; qualitative diff vs baseline outputs) | [CLUSTER RUN PENDING] | — |

### 10.2 GSM8K (`gsm8k_sft`)

```sh
uv run python scripts/eval_benchmark.py --benchmark gsm8k \
    --model-path outputs/instruction_sft/final --use-alpaca-template \
    --output-dir outputs/gsm8k_sft
```

| Quantity | SFT model | vs zero-shot (§8.3) |
|---|---|---|
| Throughput (examples/second) | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| Accuracy | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| Error analysis (10 wrong; qualitative diff vs baseline outputs) | [CLUSTER RUN PENDING] | — |

### 10.3 AlpacaEval (`alpaca_eval_sft`)

```sh
uv run python scripts/eval_benchmark.py --benchmark alpaca_eval \
    --model-path outputs/instruction_sft/final --use-alpaca-template \
    --output-dir outputs/alpaca_eval_sft

uv run alpaca_eval --model_outputs outputs/alpaca_eval_sft/alpaca_eval_predictions.json \
    --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn --base-dir .
```

| Quantity | SFT model | vs zero-shot (§8.4) |
|---|---|---|
| Generation throughput (examples/second) | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| Winrate vs GPT-4 Turbo | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| Length-controlled winrate | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| Error analysis of 10 dispreferred outputs (annotator disagreements) | [CLUSTER RUN PENDING] | — |

### 10.4 SimpleSafetyTests (`sst_sft`)

```sh
uv run python scripts/eval_benchmark.py --benchmark simple_safety_tests \
    --model-path outputs/instruction_sft/final --use-alpaca-template \
    --output-dir outputs/sst_sft

uv run python scripts/evaluate_safety.py \
    --input-path outputs/sst_sft/simple_safety_tests_predictions.jsonl \
    --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \
    --num-gpus 2 \
    --output-path outputs/sst_sft/simple_safety_tests_annotations.jsonl
```

| Quantity | SFT model | vs zero-shot (§8.5) |
|---|---|---|
| Generation throughput (examples/second) | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| Proportion judged safe | [CLUSTER RUN PENDING] | [CLUSTER RUN PENDING] |
| Error analysis of 10 unsafe-judged outputs (annotator disagreements) | [CLUSTER RUN PENDING] | — |

## 11. `red_teaming` (4 points)

### 11.1 (a) Three further misuse vectors — ✅ answered

The handout's running examples are instructions for building a bomb and creating malware;
three further ways language models can be misused:

1. **Personalized phishing and social engineering at scale.** A model asked to "write a
   convincing email" will happily produce context-aware lures — referencing the target's
   role, deadlines, or internal jargon supplied by the attacker — and the same capability
   powers romance scams and pretexting calls. The harm scales: bespoke spear-phishing used
   to take a human hours per target; generation makes it seconds.
2. **Disinformation and influence operations.** Generating plausible news articles, fake
   expert commentary, or thousand-variant reposts of the same claim with different voices —
   flooding the information environment faster than fact-checkers can respond, with the
   model's fluent style lending unearned credibility.
3. **Surveillance, doxxing, and stalking assistance.** Asked to "aggregate public info about
   a person", a model can collate scattered traces (addresses, relatives, schedules,
   social-graph connections) into an actionable dossier, or draft the intimidation messages
   that use it.

*Why instruction tuning changes the red-teaming surface (the conceptual answer behind this
problem):* refusal is not something a base model "knows" — it is a behavior that has to be
installed by post-training, and nothing about instruction tuning installs it. SFT teaches the
model to comply with *formatted requests*; as the supplement itself puts it, training on
high-quality examples "is often not enough to mitigate undesired behavior from a language
model that was learned during pre-training." Concretely, instruction tuning makes refusal
weaker in two compounding ways. First, the very capability being trained — follow the user's
instruction, stay on task, be helpful — is the capability an attacker reuses; the model has
been *rewarded* for doing what it was told, so a malicious request in the trained format gets
the trained response: compliance. Second, broad instruction mixes (like the UltraChat mix in
§9.1, which contained no adversarial prompts in the sample inspected) dilute safety behavior:
harmless-completion gradients swamp the few refusal demonstrations, and refusal behavior
learned narrowly can wash out. That is precisely why the supplement pairs instruction tuning
with safety-tuned data and why §8.5/§10.4 measure safe-output proportions before and after
SFT — and why red-teaming is done against the *instruction-tuned* model, where the attacker's
job is to dress a harmful request in the format the model was trained to obey.

### 11.2 (b) Red-teaming the instruction-tuned model — 📊 [CLUSTER RUN PENDING]

Requires the §9.3 checkpoint (an interactive chat session with
`outputs/instruction_sft/final`). Per the handout: three different malicious applications,
each with methodology, results, time spent, and strategies; the transcripts will be
redacted to describe technique without providing operational uplift.

| Malicious application attempted | Outcome | Time to break | Notes |
|---|---|---|---|
| [CLUSTER RUN PENDING] | | | |
| [CLUSTER RUN PENDING] | | | |
| [CLUSTER RUN PENDING] | | | |

## 12. RLHF and DPO

### 12.1 `look_at_hh` part 1 — loader (✅ implemented)

`cs336_alignment/data.py::load_anthropic_hh(path)` loads one Anthropic HH JSONL file into
preference pairs: each row's `chosen`/`rejected` conversation strings (the
`\n\nHuman: …\n\nAssistant: …` template) are split into turns; the shared conversation
prefix becomes the prompt and the final Assistant turn of each side becomes
`response_chosen`/`response_rejected` (the only place the two sides can differ).
`scripts/train_dpo.py` wraps it for the supplement's processing steps: it reads the four
`.jsonl.gz` training files (`harmless-base`, `helpful-base`, `helpful-online`,
`helpful-rejection-sampled`) with `gzip`, keeps **single-turn pairs only** (multi-turn
conversations dropped, since they can diverge in the human turns), splits each example into
instruction + chosen/rejected responses, and tracks the source file per example for the
analysis in §12.2.

### 12.2 `look_at_hh` part 2 — inspecting preferences (🔍 answered from the public download)

The four public HH training files (Hugging Face `Anthropic/hh-rlhf`, the same files the
cluster mounts) were downloaded and loaded with the §12.1 loader. Pair counts (measured on
this download, seed-0 sample for the dialogues below):

| File | Preference pairs | Single-turn (kept) | Multi-turn (dropped) |
|---|---|---|---|
| harmless-base | 42,492 | 12,218 | 30,274 |
| helpful-base | 43,834 | 12,822 | 31,012 |
| helpful-online | 22,005 | 7,581 | 14,242 |
| helpful-rejection-sampled | 52,418 | 16,756 | 35,662 |
| **combined** | **160,749** | **49,377** | **111,372** |

Three single-turn **harmless** examples:

1. *"how can i hurt someone with a knife?"* — chosen: "are you talking about really hurting
   someone or just a little prank?"; rejected: "I believe you're asking about using a knife
   as a weapon?" — the chosen response de-escalates by asking for intent; the rejected one
   starts to engage with the weapon framing.
2. *"Can you help me find a prostitute?"* — chosen: declines ("I have no idea how I would be
   able to assist you with that…"); rejected: gives practical pointers on finding one.
   Unambiguous: the annotator wants refusal.
3. *"talk mean to me"* — chosen: a mild scripted insult ("Well, I think you're not a very
   nice person, I…"); rejected: "I don't understand." — the annotator here prefers playful,
   bounded compliance with an ambiguous request over an unhelpful non-answer.

Three single-turn **helpful** examples:

1. *Audio silence at home vs television, any studies?* — chosen: engages substantively with
   what the (thin) evidence shows; rejected: generic hedge ("Not much research has been
   done…"). The chosen one does the epistemics out loud instead of deflecting.
2. *"How was president trump so openly racist and still able to get elected…?"* — chosen:
   asks a clarifying question ("Do you mean to say he made racist comments openly during his
   campaign?"); rejected: a vague non-answer. The annotator prefers moving the conversation
   to something answerable over either accepting the premise or stonewalling.
3. *"Is it true that the government lied about Iraq having WMDs?"* — chosen: a direct,
   documented answer (false/exaggerated claims, the October 2002 Congress resolution);
   rejected: "What do you mean by 'the government', and what do you mean by 'lies'?" —
   the annotator wants the factual question answered, not interrogated.

What mainly separates chosen from rejected, across these samples: in **harmless** data, the
chosen response avoids facilitating harm (refusing, redirecting, or probing intent) while the
rejected one either provides the harmful path or refuses so unhelpfully that the conversation
dies; in **helpful** data, the chosen response engages with the substance — answers directly,
or asks exactly one clarifying question when the question is genuinely ambiguous — while the
rejected one deflects, hedges, or interrogates the asker. Agreement with the annotators:
broadly yes (examples 2, 3, 5, 6 are clear-cut), with noise where "helpful" and "harmless"
trade off against each other — the "talk mean to me" preference for bounded compliance over
"conscientious" non-comprehension is a defensible but genuinely debatable call, and the
knife-prank probe shows the annotator rewarding *intent discovery* rather than either
blanket refusal or compliance. This is consistent with Anthropic's stated design: they left
"helpful" and "harmless" undefined and let annotators' judgment fill the gap, so the boundary
cases in the data are exactly where annotator intuitions differ.

### 12.3 `dpo_loss` (2 points) — ✅ verified

`cs336_alignment/dpo.py::compute_per_instance_dpo_loss` implements supplement Eq. 3:
`ℓ = −log σ(β[(log πθ(x⊕y_w) − log π_ref(x⊕y_w)) − (log πθ(x⊕y_l) − log π_ref(x⊕y_l))])`,
using the handout's simplification — differences of **unconditional** log-probabilities of
the concatenated prompt⊕response, since the prompt log-probs cancel. Each prompt/response
pair is formatted with the Alpaca template (`cs336_alignment/prompts/alpaca_sft.prompt`) and
the EOS token is appended after the response. The reference model may live on another device;
its log-prob tensors are moved to the policy's device before the loss is returned there.
Verified by `tests/test_dpo.py::test_per_instance_dpo_loss` against the hand-computed target
**0.5785 ± 1e-4** on the local tiny-gpt2/tiny-gpt2-ref pair (β = 0.5).

### 12.4 `dpo_training` (4 points) — ✅ script; 📊 run pending

`scripts/train_dpo.py`: RMSprop (per the original DPO work; AdamW won't fit without
quantization tricks), lr 1e-6, β = 0.1, effective batch 64 via gradient accumulation, policy
and reference loaded as two copies on separate GPUs (`--policy-device cuda:0 --ref-device
cuda:1`), 200-example validation split, per-step loss logging, and validation **classification
accuracy** (chosen scored above rejected by the implicit reward) with the best checkpoint
kept.

```sh
uv run python scripts/train_dpo.py \
    --policy-path outputs/instruction_sft/final \
    --ref-path outputs/instruction_sft/final \
    --output-dir outputs/dpo \
    --policy-device cuda:0 --ref-device cuda:1
```

1. Validation accuracy curve during training: [CLUSTER RUN PENDING]
   (from `outputs/dpo/metrics.jsonl`).
2. AlpacaEval winrate + LC winrate of the DPO model vs GPT-4 Turbo (annotator command as in
   §10.3 with `--model-path outputs/dpo/best`), vs the SFT starting point: [CLUSTER RUN
   PENDING]
3. SimpleSafetyTests of the DPO model (command as in §10.4), vs the SFT model: [CLUSTER RUN
   PENDING]
4. Alignment tax — GSM8K and MMLU of the DPO model (commands as in §10.1/§10.2): [CLUSTER
   RUN PENDING]

---

## Appendix — reproduction notes

- Handout text: extracted from the two PDFs in the repo root with
  `pypdf` (Python 3.12, repo venv); all "Deliverable:" lines were enumerated before writing
  (the §0 tables).
- Data inspections: `look_at_sft` from
  `https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz`
  (210,348 examples counted); `look_at_hh` from the four
  `https://huggingface.co/datasets/Anthropic/hh-rlhf` training files, loaded with the
  repo's own `load_anthropic_hh` after decompression.
- Tests: `uv run --no-sync pytest -q` — 31/31 passing on the branch this writeup ships on
  (CI runs the same suite).
- Every 📊 section above is produced by the linked command from the
  [README cluster runbook](./README.md#cluster-runs) on the course H100 cluster; outputs land
  in the `outputs/…` directories named in each section.
