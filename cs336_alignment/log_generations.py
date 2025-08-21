import json
import torch
from typing import List, Dict, Any, Callable, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.compute_entropy import compute_entropy
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn


def log_generations(
    model: Union[torch.nn.Module, LLM],
    tokenizer: Optional[AutoTokenizer],
    prompts: List[str],
    ground_truth_answers: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]] = r1_zero_reward_fn,
    sampling_params: Optional[SamplingParams] = None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_file: Optional[str] = None,
    return_results: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Log generations in-the-loop for SFT/RL models.
    
    Args:
        model: Either a PyTorch model or vLLM LLM instance
        tokenizer: Tokenizer for PyTorch models (not needed for vLLM)
        prompts: List of input prompts to generate responses for
        ground_truth_answers: List of ground truth answers corresponding to prompts
        reward_fn: Function to compute reward metrics (format, answer, total reward)
        sampling_params: vLLM sampling parameters (if using vLLM)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        output_file: Optional file path to save results as JSON
        return_results: Whether to return the results
        
    Returns:
        List of dictionaries containing logged information for each example, or None
    """
    
    if len(prompts) != len(ground_truth_answers):
        raise ValueError("Number of prompts must match number of ground truth answers")
    
    results = []
    
    # Handle vLLM models
    if isinstance(model, LLM):
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                include_stop_str_in_output=True
            )
        
        outputs = model.generate(prompts, sampling_params)
        
        for i, output in enumerate(outputs):
            prompt = output.prompt
            if prompt is None:
                prompt = prompts[i]
            
            generated_text = output.outputs[0].text
            ground_truth = ground_truth_answers[i]
            
            # Compute reward information
            reward_info = reward_fn(generated_text, ground_truth)
            
            # For vLLM, we can't easily get token-level entropy without model internals
            # So we'll compute response length statistics
            response_length = len(generated_text.split())
            
            result = {
                "input_prompt": prompt,
                "generated_response": generated_text,
                "ground_truth_answer": ground_truth,
                "reward_info": reward_info,
                "response_length": response_length,
                "average_token_entropy": None,  # Not available for vLLM without more work
            }
            
            results.append(result)
    
    # Handle PyTorch models
    else:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for PyTorch models")
        
        model.eval()
        
        for i, (prompt, ground_truth) in enumerate(zip(prompts, ground_truth_answers)):
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            
            # Generate response
            with torch.no_grad():
                generate_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": True,
                    "pad_token_id": tokenizer.eos_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
                
                generation_output = model.generate(**generate_kwargs)
                generated_ids = generation_output.sequences
                scores = generation_output.scores
            
            # Extract generated text (remove input prompt)
            generated_tokens = generated_ids[0][input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Compute reward information
            reward_info = reward_fn(generated_text, ground_truth)
            
            # Compute average token entropy
            if scores:
                # Stack scores to get logits for all generated tokens
                all_logits = torch.stack(scores, dim=0).transpose(0, 1)  # [batch_size, seq_len, vocab_size]
                token_entropies = compute_entropy(all_logits)  # [batch_size, seq_len]
                avg_token_entropy = token_entropies.mean().item()
            else:
                avg_token_entropy = None
            
            # Compute response length
            response_length = len(generated_tokens)
            
            result = {
                "input_prompt": prompt,
                "generated_response": generated_text,
                "ground_truth_answer": ground_truth,
                "reward_info": reward_info,
                "response_length": response_length,
                "average_token_entropy": avg_token_entropy,
            }
            
            results.append(result)
    
    # Compute aggregate statistics
    total_responses = len(results)
    if total_responses > 0:
        # Overall response length statistics
        response_lengths = [r["response_length"] for r in results]
        avg_response_length = sum(response_lengths) / len(response_lengths)
        
        # Response length for correct vs incorrect responses
        correct_responses = [r for r in results if r["reward_info"]["answer_reward"] > 0]
        incorrect_responses = [r for r in results if r["reward_info"]["answer_reward"] == 0]
        
        avg_correct_length = (
            sum(r["response_length"] for r in correct_responses) / len(correct_responses)
            if correct_responses else 0
        )
        avg_incorrect_length = (
            sum(r["response_length"] for r in incorrect_responses) / len(incorrect_responses)
            if incorrect_responses else 0
        )
        
        # Average token entropy (if available)
        entropies = [r["average_token_entropy"] for r in results if r["average_token_entropy"] is not None]
        avg_token_entropy = sum(entropies) / len(entropies) if entropies else None
        
        # Reward statistics
        format_rewards = [r["reward_info"]["format_reward"] for r in results]
        answer_rewards = [r["reward_info"]["answer_reward"] for r in results]
        total_rewards = [r["reward_info"]["reward"] for r in results]
        
        summary_stats = {
            "total_examples": total_responses,
            "average_response_length": avg_response_length,
            "average_response_length_correct": avg_correct_length,
            "average_response_length_incorrect": avg_incorrect_length,
            "average_token_entropy": avg_token_entropy,
            "format_reward_rate": sum(format_rewards) / len(format_rewards),
            "answer_reward_rate": sum(answer_rewards) / len(answer_rewards),
            "total_reward_rate": sum(total_rewards) / len(total_rewards),
            "num_correct": len(correct_responses),
            "num_incorrect": len(incorrect_responses),
        }
        
        # Add summary to results
        full_results = {
            "summary_statistics": summary_stats,
            "individual_results": results
        }
    else:
        full_results = {
            "summary_statistics": {},
            "individual_results": results
        }
    
    # Save to file if requested
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    # Print summary statistics
    if total_responses > 0:
        print("\n=== Generation Logging Summary ===")
        print(f"Total examples: {summary_stats['total_examples']}")
        print(f"Average response length: {summary_stats['average_response_length']:.2f}")
        print(f"Average response length (correct): {summary_stats['average_response_length_correct']:.2f}")
        print(f"Average response length (incorrect): {summary_stats['average_response_length_incorrect']:.2f}")
        if summary_stats['average_token_entropy'] is not None:
            print(f"Average token entropy: {summary_stats['average_token_entropy']:.4f}")
        print(f"Format reward rate: {summary_stats['format_reward_rate']:.3f}")
        print(f"Answer reward rate: {summary_stats['answer_reward_rate']:.3f}")
        print(f"Total reward rate: {summary_stats['total_reward_rate']:.3f}")
        print(f"Correct responses: {summary_stats['num_correct']}/{total_responses}")
    
    return full_results if return_results else None


def log_generations_vllm(
    model_name: str,
    prompts: List[str],
    ground_truth_answers: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]] = r1_zero_reward_fn,
    sampling_params: Optional[SamplingParams] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for logging generations with vLLM models.
    
    Args:
        model_name: HuggingFace model name or path
        prompts: List of input prompts
        ground_truth_answers: List of ground truth answers
        reward_fn: Reward function to use
        sampling_params: vLLM sampling parameters
        output_file: Optional output file path
        
    Returns:
        Results dictionary with summary statistics and individual results
    """
    # Create vLLM model
    llm = LLM(model=model_name)
    
    result = log_generations(
        model=llm,
        tokenizer=None,
        prompts=prompts,
        ground_truth_answers=ground_truth_answers,
        reward_fn=reward_fn,
        sampling_params=sampling_params,
        output_file=output_file
    )
    
    if result is None:
        return {"summary_statistics": {}, "individual_results": []}
    return result


def log_generations_pytorch(
    model_name_or_path: str,
    prompts: List[str],
    ground_truth_answers: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]] = r1_zero_reward_fn,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for logging generations with PyTorch models.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        prompts: List of input prompts  
        ground_truth_answers: List of ground truth answers
        reward_fn: Reward function to use
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        output_file: Optional output file path
        
    Returns:
        Results dictionary with summary statistics and individual results
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    result = log_generations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        ground_truth_answers=ground_truth_answers,
        reward_fn=reward_fn,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        output_file=output_file
    )
    
    if result is None:
        return {"summary_statistics": {}, "individual_results": []}
    return result
