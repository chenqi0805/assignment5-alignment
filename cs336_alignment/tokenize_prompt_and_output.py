import torch
from transformers import PreTrainedTokenizerBase

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase) -> dict[str, torch.Tensor]:
    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompt_strs]
    output_tokens = [tokenizer.encode(output) for output in output_strs]
    full_input_ids = []
    response_masks = []
    for prompt_ids, output_ids in zip(prompt_tokens, output_tokens):
        full_input_ids.append(torch.tensor(prompt_ids + output_ids, dtype=torch.long))
        response_mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        response_masks.append(torch.tensor(response_mask, dtype=torch.long))

    batch_size = len(full_input_ids)
    max_len = max(len(ids) for ids in full_input_ids)
    input_ids_batch = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
    response_mask_batch = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(full_input_ids, response_masks)):
        seq_len = len(ids)
        input_ids_batch[i, :seq_len] = ids
        response_mask_batch[i, :seq_len] = mask

    return {
        "input_ids": input_ids_batch[:, :-1],
        "labels": input_ids_batch[:, 1:],
        "response_mask": response_mask_batch[:, 1:]
    }