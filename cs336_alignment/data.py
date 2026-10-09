"""Data loading and metric parsing: packed SFT dataset, batching, parsers."""

import json
import logging
import random
import re
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

ALPACA_SFT_PROMPT_PATH = Path(__file__).parent / "prompts" / "alpaca_sft.prompt"


class PackedSFTDataset(Dataset):
    """Constant-length packed language-model dataset over instruction pairs.

    Each document is tokenized (with the tokenizer's special tokens, e.g.
    BOS), terminated with the EOS token id as a delimiter, and the documents
    are concatenated into one token stream. The stream is cut into
    non-overlapping ``seq_length`` chunks; the trailing partial chunk is
    dropped. ``labels`` are the next-token targets (the input stream shifted
    left by one), so examples cross document boundaries exactly as in
    standard LM packing.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str,
        seq_length: int,
        shuffle: bool = False,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        with open(dataset_path) as f:
            documents = [json.loads(line) for line in f if line.strip()]

        if shuffle:
            random.shuffle(documents)

        eos_id = tokenizer.eos_token_id
        template = ALPACA_SFT_PROMPT_PATH.read_text()
        stream: list[int] = []
        for doc in documents:
            # The supplement packs Alpaca-template-formatted documents; the
            # template's trailing newline is stripped so the EOS delimiter
            # directly follows the response (per the tokenized fixture).
            text = template.format(instruction=doc["prompt"], response=doc["response"]).rstrip()
            token_ids = tokenizer(text)["input_ids"]
            stream.extend(token_ids)
            stream.append(eos_id)

        num_examples = len(stream) // seq_length
        if num_examples == 0:
            raise ValueError(
                f"Token stream ({len(stream)} tokens) is shorter than seq_length={seq_length}"
            )
        # Predict-the-next-token targets: labels use stream[1:] so the final
        # label of each chunk comes from the token that follows it.
        targets = stream[1:] + [eos_id]
        self.examples: list[dict[str, Tensor]] = [
            {
                "input_ids": torch.tensor(
                    stream[i * seq_length : (i + 1) * seq_length], dtype=torch.long
                ),
                "labels": torch.tensor(
                    targets[i * seq_length : (i + 1) * seq_length], dtype=torch.long
                ),
            }
            for i in range(num_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.examples[index]


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """Factory for PackedSFTDataset (adapter-facing name from the handout)."""
    return PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        seq_length=seq_length,
        shuffle=shuffle,
    )


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Iterable over batches of ``batch_size`` examples; one epoch per iteration.

    Args:
        dataset: Dataset to emit batches from.
        batch_size: int, number of examples per batch (final batch may be
            smaller).
        shuffle: bool, shuffle example order before batching.

    Returns:
        torch.utils.data.DataLoader with default collation (dicts of stacked
        tensors), supporting ``len()``.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


_MMLU_LETTER_RE = re.compile(r"\b([A-D])\b")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_mmlu_response(mmlu_example: dict, model_output: str) -> str | None:
    """Parse an MMLU model output into a predicted option letter.

    Takes the last standalone option letter (A-D) occurring in the output.
    Option *numbers* are deliberately not matched: "The correct answer is
    10000 polyomaviruses" contains no standalone letter and yields None
    rather than a spurious parse.

    Args:
        mmlu_example: dict with the MMLU example (unused beyond the option
            list length contract; the parser needs only the output).
        model_output: str with the model's output.

    Returns:
        str ("A"-"D") if the output can be parsed, else None.
    """
    matches = _MMLU_LETTER_RE.findall(model_output)
    if not matches:
        return None
    return matches[-1]


def parse_gsm8k_response(model_output: str) -> str | None:
    """Parse a GSM8K model output into its last numeric answer, as a string.

    Commas inside numbers are stripped ("1,234" -> "1234"). Spelled-out
    numbers ("seventy-two") are not numbers: no digits means None.

    Args:
        model_output: str with the model's output.

    Returns:
        str with the predicted numeric answer if one occurs, else None.
    """
    matches = _NUMBER_RE.findall(model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")


_HH_TURN_RE = re.compile(r"\n\n(Human|Assistant):\s*", re.DOTALL)


def _split_hh_dialogue(text: str) -> list[dict[str, str]]:
    """Split an Anthropic HH-RLHF conversation string into chat turns."""
    parts = [p.strip() for p in _HH_TURN_RE.split(text) if p.strip()]
    turns: list[dict[str, str]] = []
    for i in range(0, len(parts) - 1, 2):
        turns.append({"role": parts[i].lower(), "content": parts[i + 1]})
    return turns


def load_anthropic_hh(path: str) -> list[dict]:
    """Load an Anthropic HH-RLHF jsonl file into preference-pair examples.

    The source format stores one conversation string per row under
    ``chosen`` and ``rejected`` keys, in the "\n\nHuman: ...\n\nAssistant:
    ..." template. Each row becomes {"prompt": [...], "response_chosen":
    str, "response_rejected": str} where the prompt is the shared
    conversation prefix and the responses are the final Assistant turn of
    each side (the only part where chosen and rejected differ).
    """
    examples: list[dict] = []
    with open(path) as f:
        for line_number, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            chosen_turns = _split_hh_dialogue(row["chosen"])
            rejected_turns = _split_hh_dialogue(row["rejected"])
            if not chosen_turns or not rejected_turns:
                logger.warning(
                    "Skipping HH row %d: no complete Human/Assistant turns", line_number
                )
                continue
            # The preference pair shares the context; the final Assistant
            # turn is the response being chosen or rejected.
            if chosen_turns[-1]["role"] != "assistant":
                logger.warning("Skipping HH row %d: chosen does not end with Assistant turn", line_number)
                continue
            prompt = chosen_turns[:-1]
            examples.append(
                {
                    "prompt": prompt,
                    "response_chosen": chosen_turns[-1]["content"],
                    "response_rejected": rejected_turns[-1]["content"],
                }
            )
    return examples
