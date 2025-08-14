import json
from pathlib import Path
from typing import Callable, List
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        eval_sampling_params: SamplingParams,
        output_file: str
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    results = []
    for output in outputs:
        prompt = output.prompt
        if prompt is None:
            raise ValueError("Output prompt is None")
        generated_text = output.outputs[0].text
        results.append(reward_fn(prompt, generated_text))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    # Model repo name from Hugging Face
    model_name = "Qwen/Qwen2.5-Math-1.5B"

    # Create the LLM instance
    llm = LLM(model=model_name, device="cuda")

    eval_sampling_params = SamplingParams(
        include_stop_str_in_output=True,
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024, 
        stop=["</answer>"])
    
    script_dir = Path(__file__).parent
    # Build the full path to the prompt file
    prompt_path = script_dir / "prompts" / "r1_zero.prompt"

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    questions = [
        "Please solve the following math problem: 1+1=",
        "Please solve the following math problem: 2+2=",
        "Please solve the following math problem: 3+3="
    ]

    prompts = [prompt_template.format(question=question) for question in questions]
    
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=eval_sampling_params,
        output_file=str(script_dir / "evaluation_metrics.json")
    )