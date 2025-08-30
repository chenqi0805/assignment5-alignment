import json
import os
import random
from typing import Iterable
import numpy as np
import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from tests.conftest import labels


def load_alpaca_template():
    """
    Load the Alpaca SFT template from the prompts directory.
    
    Returns:
        str: The Alpaca template string
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, 'prompts', 'alpaca_sft.prompt')
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


# Load the Alpaca template as a constant
ALPACA_TEMPLATE = load_alpaca_template()

def read_sft_data(jsonl_path: str | os.PathLike):
    """
    Read SFT (Supervised Fine-Tuning) data from a JSONL file.
    
    Args:
        jsonl_path (str): Path to the JSONL file containing prompt-response pairs
        
    Returns:
        tuple: (prompts, responses) where both are lists of strings
    """
    prompts = []
    responses = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data = json.loads(line)
                prompts.append(data['prompt'])
                responses.append(data['response'])
    
    return prompts, responses


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_path: str | os.PathLike, seq_length: int, shuffle: bool):
        prompts, responses = read_sft_data(dataset_path)
        examples = []
        for prompt, response in zip(prompts, responses):
            examples.append(ALPACA_TEMPLATE.format(instruction=prompt, response=response) + tokenizer.eos_token)

        if shuffle:
            random.shuffle(examples)

        example_ids = []
        for example in examples:
            example_ids.extend(tokenizer.encode(example))

        input_ids, labels = example_ids[:-1], example_ids[1:]
        
        n = len(input_ids)
        n = n - n % seq_length
        starts = np.arange(0, n, step=seq_length)
        
        input_ids_seqs = []
        labels_seqs = []
        
        for start in starts:
            input_ids_seqs.append(input_ids[start : start + seq_length])
            labels_seqs.append(labels[start : start + seq_length])
        
        self.input_ids, self.labels = torch.tensor(input_ids_seqs, dtype=torch.long), torch.tensor(labels_seqs, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx], 
            "labels": self.labels[idx]
        }

def iterate_batches(dataloader: torch.utils.data.Dataset, batch_size: int):
    input_ids_batch = []
    labels_batch = []
    for item in dataloader:
        input_ids_batch.append(item["input_ids"])
        labels_batch.append(item["labels"])
        if len(input_ids_batch) == batch_size:
            yield {
                "input_ids": torch.stack(input_ids_batch),
                "labels": torch.stack(labels_batch)
            }
            input_ids_batch = []
            labels_batch = []
    if input_ids_batch != []:
        yield {
            "input_ids": torch.stack(input_ids_batch),
            "labels": torch.stack(labels_batch)
        }