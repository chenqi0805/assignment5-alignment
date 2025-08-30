import re


def parse_gsm8k_response(response: str) -> str:
    pattern = r'(\d+(\.\d+)?)(?!.*\d)'

    match = re.search(pattern, response)
    if match:
        return match.group(1)