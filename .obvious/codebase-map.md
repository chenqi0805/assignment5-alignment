# Codebase map — chenqi0805/assignment5-alignment

| Path | Type | Purpose |
|---|---|---|
| `cs336_alignment/` | package | Learner solution code (SFT, DPO, GRPO, metrics). Currently empty `__init__.py` — fresh assignment state. |
| `cs336_alignment/prompts/` | prompts | `.prompt` templates: `r1_zero`, `alpaca_sft`, `question_only`, `zero_shot_system_prompt`. |
| `cs336_alignment/drgrpo_grader.py` | module | Math-answer grading function (from sail-sg/understand-r1-zero), provided. |
| `tests/` | pytest | 31 tests. `tests/adapters.py` is the wiring point learners edit; `conftest.py` (line 213) hardcodes the Qwen2.5-Math-1.5B path. |
| `tests/_snapshots/` | data | Golden `.npz` arrays the tests compare against. |
| `tests/fixtures/` | data | Test fixtures. |
| `data/` | data | `gsm8k/`, `mmlu/`, `alpaca_eval/`, `simple_safety_tests/` evaluation datasets (tracked despite `data/` in `.gitignore`). |
| `scripts/` | scripts | `evaluate_safety.py`; `alpaca_eval_vllm_llama3_3_70b_fn/` (alpaca-eval vLLM judge config). |
| `*.pdf` | docs | Assignment handout + optional safety/RLHF supplement. |
| `test_and_make_submission.sh` | script | Runs the suite, then zips the repo for course submission. |
| `pyproject.toml` / `uv.lock` | config | Package + locked deps (uv); includes flash-attn build metadata override. |
