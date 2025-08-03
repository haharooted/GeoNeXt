from __future__ import annotations
import json, collections

def deep_to_str(obj, indent: int = 2) -> str:
    """Pretty‑prints any nested JSON/dict/list as a compact string for LLM."""
    return json.dumps(obj, ensure_ascii=False, indent=indent)
