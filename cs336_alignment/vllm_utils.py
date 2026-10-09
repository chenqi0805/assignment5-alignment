"""vLLM utilities: engine init, policy weight refresh, and MATH evaluation.

Imported by the experiment scripts; vLLM itself is imported lazily by the
callers (the module stays importable on CPU-only machines, where the
``vllm`` package is installed but unusable).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch

if TYPE_CHECKING:  # pragma: no cover - type-only, never imported at runtime
    from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def init_vllm(
    model_id: str,
    device: str = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    enforce_eager: bool = True,
    gpu_memory_utilization: float = 0.85,
    seed: int | None = None,
    max_model_len: int | None = None,
) -> "LLM":
    """Create a vLLM ``LLM`` for offline inference of ``model_id``.

    Args:
        model_id: str, HuggingFace model id or path to a local checkpoint.
        device: str, device for the engine, e.g. ``"cuda:0"`` or ``"auto"``.
        dtype: str | torch.dtype, weight dtype (e.g. ``"bfloat16"``).
        enforce_eager: bool, skip CUDA-graph capture (cheaper startup, and
            required semantics under frequent external weight refreshes).
        gpu_memory_utilization: float, fraction of GPU memory for vLLM.
        seed: int | None, sampling seed for reproducible rollouts.
        max_model_len: int | None, override the model's context length.

    Returns:
        vllm.LLM configured for offline generation.
    """
    from vllm import LLM  # lazy: keep CPU-only imports cheap

    kwargs: dict[str, Any] = {
        "model": model_id,
        "device": device,
        "dtype": dtype,
        "enforce_eager": enforce_eager,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if seed is not None:
        kwargs["seed"] = seed
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)


def _vllm_model(llm: "LLM") -> torch.nn.Module:
    """Reach the raw ``torch.nn.Module`` inside a vLLM ``LLM`` (v0 engine)."""
    return llm.llm_engine.model_executor.driver_worker.model_runner.model


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: "LLM") -> None:
    """Copy the policy's weights into a running vLLM instance, in place.

    Used between rollout phases so the inference engine always mirrors the
    training policy. Every vLLM parameter must be sourced from the policy's
    state dict: a name that cannot be matched raises (stale rollouts from
    silently skipped weights are far worse than a loud failure).

    Tied embeddings (e.g. Qwen2.5-Math-1.5B ties ``lm_head`` to
    ``model.embed_tokens``) are handled: vLLM may expose ``lm_head.weight``
    even when the policy state dict only carries the embedding.
    """
    vllm_model = _vllm_model(llm)
    policy_state = policy.state_dict()
    tied = bool(getattr(getattr(policy, "config", None), "tie_word_embeddings", False))

    with torch.no_grad():
        for name, vllm_param in vllm_model.state_dict().items():
            if name in policy_state:
                source = policy_state[name]
            elif tied and name == "lm_head.weight" and "model.embed_tokens.weight" in policy_state:
                source = policy_state["model.embed_tokens.weight"]
            else:
                raise KeyError(
                    f"vLLM parameter {name!r} not found in policy state_dict; "
                    "cannot safely refresh the inference engine"
                )
            vllm_param.data.copy_(
                source.to(device=vllm_param.device, dtype=vllm_param.dtype)
            )
    logger.info("Refreshed vLLM weights from policy (%d tensors)", len(vllm_model.state_dict()))


def evaluate_vllm(
    vllm_model: "LLM",
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    eval_sampling_params: "SamplingParams",
    ground_truths: Sequence[str],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate for ``prompts``, score with ``reward_fn``, aggregate metrics.

    The handout's ``evaluate_vllm`` starter, extended with the ground truths
    the reward function needs and an optional JSONL dump of every example
    (prompt, generation, and per-component reward) for later analysis.

    Args:
        vllm_model: vllm.LLM, the inference engine.
        reward_fn: Callable[[str, str], dict[str, float]], scores
            (response, ground_truth) pairs, e.g. ``r1_zero_reward_fn``.
        prompts: list[str], fully formatted prompts.
        eval_sampling_params: vllm.SamplingParams, generation config.
        ground_truths: sequence of str, one gold answer per prompt.
        output_path: str | None, if given, write one JSON object per line.

    Returns:
        dict with keys ``metrics`` (accuracy/format/answer aggregates and
        per-category counts), ``generations``, ``scores`` (per-example
        reward dicts), and ``rewards`` (per-example total reward).
    """
    if len(prompts) != len(ground_truths):
        raise ValueError(
            f"prompts ({len(prompts)}) and ground_truths ({len(ground_truths)}) lengths differ"
        )
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    generations = [output.outputs[0].text for output in outputs]

    scores = [
        reward_fn(generation, ground_truth)
        for generation, ground_truth in zip(generations, ground_truths)
    ]
    rewards = [float(score["reward"]) for score in scores]

    def _count(predicate) -> int:
        return sum(1 for score in scores if predicate(score))

    metrics = {
        "num_examples": len(generations),
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "accuracy": _count(lambda s: s["reward"] == 1.0) / len(scores) if scores else 0.0,
        "format_accuracy": _count(lambda s: s["format_reward"] == 1.0) / len(scores)
        if scores
        else 0.0,
        "answer_accuracy": _count(lambda s: s["answer_reward"] == 1.0) / len(scores)
        if scores
        else 0.0,
        # math_baseline (b) categories: correct (both 1), formatted but
        # wrong (format 1 / answer 0), and format failures (format 0).
        "num_correct": _count(lambda s: s["reward"] == 1.0),
        "num_format_correct_answer_wrong": _count(
            lambda s: s["format_reward"] == 1.0 and s["answer_reward"] == 0.0
        ),
        "num_format_wrong": _count(lambda s: s["format_reward"] == 0.0),
    }

    if output_path is not None:
        with open(output_path, "w") as f:
            for prompt, ground_truth, generation, score in zip(
                prompts, ground_truths, generations, scores
            ):
                f.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "ground_truth": ground_truth,
                            "generation": generation,
                            **score,
                        }
                    )
                    + "\n"
                )
        logger.info("Wrote %d evaluated generations to %s", len(generations), output_path)

    return {"metrics": metrics, "generations": generations, "scores": scores, "rewards": rewards}
