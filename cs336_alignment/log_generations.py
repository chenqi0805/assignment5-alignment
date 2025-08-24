import json
import numpy as np
import torch
from typing import List, Dict, Any, Callable, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output

def log_generations(
    prompt_template: str,
    model: PreTrainedModel,
    llm: LLM,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    ground_truth_answers: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]] = r1_zero_reward_fn,
    sampling_params: Optional[SamplingParams] = None,
) -> Dict[str, Any]:
    llm_prompts = [prompt_template.format(question=question) for question in prompts]
    llm_outputs = llm.generate(
        llm_prompts,
        sampling_params,
    )

    results = []
    for i in range(len(prompts)):
        output = llm_outputs[i]
        prompt = output.prompt
        if prompt is None:
            raise ValueError("Output prompt is None")
        reward_dict = reward_fn(prompt, output.outputs[0].text)
        results.append({
            "prompt": output.prompt,
            "generated_text": output.outputs[0].text,
            "ground_truth_answer": ground_truth_answers[i],
            "format_reward": reward_dict["format_reward"],
            "answer_reward": reward_dict["answer_reward"],
            "reward": reward_dict["reward"],
        })

    average_incorrect_response_len = np.average(np.array([r["generated_text"] for r in results if r["answer_reward"] < 1]))
    average_correct_response_len = np.average(np.array([r["generated_text"] for r in results if r["answer_reward"] > 0]))
    average_response_len = np.average(np.array([r["generated_text"] for r in results]))
    accuracy = np.average(np.array([r["answer_reward"] for r in results]))

    return {
        "average_incorrect_response_len": average_incorrect_response_len,
        "average_correct_response_len": average_correct_response_len,
        "average_response_len": average_response_len,
        "accuracy": accuracy,
        "results": results
    }