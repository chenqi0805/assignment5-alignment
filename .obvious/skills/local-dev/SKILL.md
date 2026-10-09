---
name: local-dev
---

# local-dev — assignment5-alignment

Durable record of the 2026-10-09 onboarding run that brought this repo to a
healthy local dev state on a CPU-only sandbox.

## Prerequisites

- `uv` (installed at `~/.local/bin/uv`; `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- No Docker/DB/services required. No secrets/env vars required for the test suite.
- CPU-only is fine for tests; GPU + CUDA toolkit only for flash-attn/training runs.

## Steps that worked

1. `uv sync --no-install-package flash-attn` — installs torch 2.5.1+cu124,
   transformers 4.51.3, vllm 0.7.2 and all other deps (~few GB, minutes).
   Verified exit 0.
2. Skip/expect failure of full `uv sync` on CPU-only machines: flash-attn
   2.7.4.post1 fails with `OSError: CUDA_HOME environment variable is not set`
   (no nvcc). It is not imported by `cs336_alignment/` or `tests/`.
3. Run tests with `uv run --no-sync pytest -v` — **`--no-sync` is required**,
   otherwise `uv run` re-syncs, tries to build flash-attn, and fails.

## Expected first-run result

31 tests collected in 0.04s; run completes in ~4s with 29
`NotImplementedError` failures (README documents this as the fresh-assignment
state — learner fills in `cs336_alignment/` via `tests/adapters.py`) and 2
errors from the hardcoded missing checkpoint
`/data/a5-alignment/models/Qwen2.5-Math-1.5B` (`tests/conftest.py:213`,
`TODO(confirm)` how to provide it locally).

## Gotchas

- System python is 3.13 on the sandbox — uv provisions the pinned 3.12 itself.
- `data/` is in `.gitignore` but its dataset files are tracked anyway.
- vllm prints "No platform detected, running on UnspecifiedPlatform" without a
  GPU; harmless for the suite.
