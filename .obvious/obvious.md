# obvious.md — chenqi0805/assignment5-alignment

CS336 Spring 2025 Assignment 5 (Alignment): a Python course-assignment repo.
Learners implement LLM alignment algorithms (SFT, DPO, GRPO/RLVR) in
`cs336_alignment/` and wire them into the pytest suite via `tests/adapters.py`.

## Stack

- **Language/runtime:** Python 3.12 (`requires-python >=3.11,<3.13`; pinned via `.python-version`)
- **Package manager:** `uv` (locked: `uv.lock`)
- **Heavy deps:** torch 2.5.1+cu124, transformers 4.51.3, vllm 0.7.2, flash-attn 2.7.4.post1
- **Test framework:** pytest (31 tests)
- **Services:** none — no web app, no database, no broker, no ports. CPU-only workflow for the test suite; GPU is only needed for training/inference runs described in the assignment PDF.

## Commands

```sh
# Environment setup (README-documented two-step; flash-attn is CUDA-only, see below)
uv sync --no-install-package flash-attn   # install everything else
uv sync                                   # add flash-attn (requires CUDA toolkit + GPU)

# Run the test suite (canonical local check; mirrors test_and_make_submission.sh)
uv run --no-sync pytest -v

# Assignment submission (runs tests, then zips the repo)
bash test_and_make_submission.sh
```

`uv run` re-checks the lockfile against the environment; when flash-attn is
absent (CPU-only machines), always pass `--no-sync` or `uv run` fails trying to
build it. No lint/typecheck tools are configured (`TODO(confirm)` if the course
adds them).

## Codebase map

See [codebase-map.md](./codebase-map.md) for the folder table.

## Local verification

1. `uv run --no-sync python -c "import torch, transformers, vllm"` — imports succeed.
2. `uv run --no-sync pytest -v` — suite executes end-to-end.
   - **Expected on a fresh checkout:** all adapter-backed tests fail with
     `NotImplementedError` (per README: "Initially, all tests should fail with
     NotImplementedErrors"). Observed at onboarding: 29 failed + 2 errors.
   - The 2 errors need a local `Qwen2.5-Math-1.5B` checkpoint (see Known limitations).

## Sandbox snapshot

- **snapshotId:** `v9z21znmn3ijdttiekmk:default`
- **Captured:** 2026-10-09T17:10:08.993Z (ISO-8601)
- **State at capture:** deps synced (minus flash-attn), suite runs; sandboxId `isd7f4cvdtjrx9gp4u441`

## Known limitations

- **flash-attn cannot build on CPU-only sandboxes** (needs CUDA toolkit/GPU).
  It is not imported by `cs336_alignment/` or `tests/`, so the test suite runs
  without it. Only training/inference scripts from the handout need it.
- **2 test errors are environmental:** `tests/conftest.py:213` hardcodes
  `/data/a5-alignment/models/Qwen2.5-Math-1.5B` (instructor cluster path).
  `TODO(confirm)` preferred location/env var for this checkpoint; until it
  exists locally, `test_tokenize_prompt_and_output` and
  `test_get_response_log_probs` error with `HFValidationError`.
- **No GPU** on the onboarding sandbox: `vllm` logs
  "No platform detected, running on UnspecifiedPlatform" — harmless for tests.

## Local Verification Summary

- **Result:** PASS — dev_stack_healthy: true (2026-10-09T17:09Z)
- **Setup:** `uv sync --no-install-package flash-attn` → exit 0 (uv 0.12.24, Python 3.12.15)
- **Primary flow evidence:** `uv run --no-sync pytest -v` — 31 tests collected, full run 3.75s: 29 failed (`NotImplementedError`, documented fresh-assignment state per README) + 2 errors (missing local Qwen2.5-Math-1.5B checkpoint, `tests/conftest.py:213`). Log: `/tmp/a5-evidence/pytest-full.log`, junit XML: `/tmp/a5-evidence/pytest-report.xml`. Single-test rerun (`tests/test_metrics.py::test_parse_gsm8k_response`) reaches the adapter stub deterministically.
- **Import smoke test:** torch 2.5.1+cu124, transformers 4.51.3, vllm 0.7.2 import OK; `import cs336_alignment` OK.
- **Blockers:** none (flash-attn/CUDA and model-checkpoint caveats are documented environmental limitations, not setup blockers).
