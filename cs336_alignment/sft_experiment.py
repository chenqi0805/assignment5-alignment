import argparse
import pathlib
from typing import Iterable
from unittest.mock import patch
import numpy as np
from sympy import test
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import wandb

from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.log_generations import log_generations
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_data_loading import math_data_loading
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on 
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None)
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def batch_data_loading(prompts: list[str], output_strs: list[str], batch_size: int, tokenizer: PreTrainedTokenizer) -> Iterable[tuple]:
    n = len(prompts)
    indices = np.arange(n)

    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]

        batch_prompts = [prompts[i] for i in batch_indices]
        batch_outputs = [output_strs[i] for i in batch_indices]

        tokenized = tokenize_prompt_and_output(batch_prompts, batch_outputs, tokenizer)

        yield (
            tokenized["input_ids"],
            tokenized["labels"],
            tokenized["response_mask"],
        )

def run_sft_experiment(
    batch_size: int,
    n_sft: int, 
    math_data_set_dir: str,
    model_id: str,
    gradient_accumulation_steps: int,
    output_dir: str,
):
    """
    Run the supervised fine-tuning experiment with the given parameters.
    
    Args:
        batch_size: Size of each training batch
        n_sft: Number of SFT training steps
        math_data_set_dir: Directory containing the MATH dataset
        model_id: HuggingFace model identifier
        gradient_accumulation_steps: Number of steps to accumulate gradients
        output_dir: Directory to save the fine-tuned model
    """
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")

    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    eval_columns = [
        "prompt",
        "generated_text",
        "ground_truth_answer",
        "log_probs",
        "token_entropy",
        "format_reward",
        "answer_reward",
        "reward",
    ]
    
    llm = init_vllm(model_id, device="cuda:0", seed=42)

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    script_dir = pathlib.Path(__file__).parent
    # Build the full path to the prompt file
    prompt_path = script_dir / "prompts" / "r1_zero.prompt"

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    training_prompts, training_output_strs = math_data_loading(math_data_set_dir + "/train")
    testing_prompts, testing_output_strs = math_data_loading(math_data_set_dir + "/test")

    optimizer = torch.optim.AdamW(model.parameters())

    step = 0
    micro_step = 0
    while True:
        for inputs, labels, response_mask in batch_data_loading(training_prompts, training_output_strs, batch_size, tokenizer):
            micro_step += 1
            response_dict = get_response_log_probs(model, inputs, labels)
            training_loss, training_metadata = sft_microbatch_train_step(response_dict["log_probs"], response_mask, gradient_accumulation_steps)

            if micro_step % gradient_accumulation_steps == 0:
                step += 1
                # Update weights
                optimizer.step()
                # Zero gradients every `gradient_accumulation_steps` batches.
                optimizer.zero_grad()

            if step % args.log_interval == 0:
                wandb.log({
                    "train_step": step,
                    "train/loss": training_loss.item(),
                    "train/average_token_entropy": training_metadata["avg_ce_per_token"].item(),
                })

            if step % args.eval_interval == 0:
                load_policy_into_vllm_instance(model, llm)

                eval_sampling_params = SamplingParams(
                    include_stop_str_in_output=True,
                    temperature=1.0, 
                    top_p=1.0, 
                    max_tokens=1024, 
                    stop=["</answer>"])
                
                benchmark = log_generations(
                    prompt_template,
                    model,
                    llm,
                    tokenizer,
                    testing_prompts,
                    testing_output_strs,
                    r1_zero_reward_fn,
                    eval_sampling_params,
                )
                
                # TODO: evaluate and log samples from test set
                eval_table = wandb.Table(columns=eval_columns)
                for result in benchmark["results"]:
                    row = [result[col] for col in eval_columns]  # preserve order
                    eval_table.add_data(*row)
                wandb.log({f"eval_results_{step}": eval_table})

                wandb.log({
                    "eval_step": step,
                    "eval/average_incorrect_response_len": benchmark["average_incorrect_response_len"],
                    "eval/average_correct_response_len": benchmark["average_correct_response_len"],
                    "eval/average_response_len": benchmark["average_response_len"],
                    "eval/accuracy": benchmark["accuracy"]
                })

            if step >= n_sft:
                break

        if step >= n_sft:
            break

    wandb.finish()

    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning experiment on MATH dataset")
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Size of each training batch (default: 32)"
    )
    
    parser.add_argument(
        "--n_sft", 
        type=int,
        default=10000,
        help="Number of SFT training steps (default: 10000)"
    )
    
    parser.add_argument(
        "--math_data_set_dir",
        type=str,
        default="data/MATH",
        help="Directory containing the MATH dataset (default: data/MATH)"
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HuggingFace model identifier (default: Qwen/Qwen2.5-Math-1.5B)"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients (default: 4)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the fine-tuned model (default: auto-generated based on model_id and n_sft)"
    )

    parser.add_argument(
        "--log_interval", 
        type=int, 
        default=500,
        help="Log every N steps")
    
    parser.add_argument(
        "--eval_interval", 
        type=int, 
        default=1000,
        help="Evaluate every N steps")
    
    parser.add_argument("--wandb_project", type=str, default="sft",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set default output_dir if not provided
    if args.output_dir is None:
        # Remove the leading slash and create a more reasonable default path
        model_name = args.model_id.replace("/", "-")
        args.output_dir = f"{model_name}-sft-{args.n_sft}"
    
    print(f"Running SFT experiment with the following parameters:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of SFT steps: {args.n_sft}")
    print(f"  MATH dataset directory: {args.math_data_set_dir}")
    print(f"  Model ID: {args.model_id}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Output directory: {args.output_dir}")
    print()

    config = vars(args)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config
    )
    
    run_sft_experiment(
        batch_size=args.batch_size,
        n_sft=args.n_sft,
        math_data_set_dir=args.math_data_set_dir,
        model_id=args.model_id,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
    )
