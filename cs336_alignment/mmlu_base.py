import re


def parse_mmlu_response(response: str) -> str:
    match = re.search(r'\bThe correct answer is (\w)\b', response)
    if match:
        return match.group(1)