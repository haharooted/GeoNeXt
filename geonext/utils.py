from __future__ import annotations
import json, collections
import re
def deep_to_str(obj, indent: int = 2) -> str:
    """Pretty‑prints any nested JSON/dict/list as a compact string for LLM."""
    return json.dumps(obj, ensure_ascii=False, indent=indent)

def extract_json(text: str) -> str:
    # Match code block: ```json ... ```
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    raise ValueError("No valid JSON found in output.")