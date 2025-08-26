#!/usr/bin/env python
"""
4omini.py – Quick benchmark of OpenAI o4-mini on GeoCorpora
===========================================================

Compares an LLM extraction approach to your BoW baseline
on a *subset* of GeoCorpora to keep token-cost down.

* Zero-shot (`--shots 0`) or 1-shot (`--shots 1`)
* Evaluates up to `--limit` tweets (default 1000)
* Caches every response under cache_llm/ so reruns are free
* Prints token-level precision / recall / F1 identical to the BoW script

Usage
-----
export OPENAI_API_KEY="sk-…"           # set your key

# zero-shot on 1 000 tweets
python 4omini.py --geocorpora geocorpora.json

# few-shot on 500 random tweets
python 4omini.py --geocorpora geocorpora.json --shots 1 --limit 500
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from hashlib import sha1
from pathlib import Path
from typing import List, Tuple

import numpy as np
from openai import OpenAI
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

# ------------------------------------------------------------------#
# crude tokeniser (same regex used in BoW baseline)                 #
# ------------------------------------------------------------------#
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


# ------------------------------------------------------------------#
# minimal GeoCorpora JSON loader (array or JSON-Lines)              #
# ------------------------------------------------------------------#
def load_gc(path: str) -> Tuple[List[str], List[List[int]]]:
    """Return tweet texts and gold BIO tags (0=O,1=LOC)."""
    raw = re.sub(r":\s*NaN", ": null", Path(path).read_text("utf-8"))
    records = (
        json.loads(raw)
        if raw.lstrip().startswith("[")
        else [json.loads(l) for l in raw.splitlines() if l.strip()]
    )

    texts, tags = [], []
    for obj in records:
        tweet = obj.get("text") or obj.get("tweet_text")
        if not tweet:
            continue
        toks = tokenize(tweet)

        # char-offset → token index
        offset_map, pos = {}, 0
        for i, tok in enumerate(toks):
            start = tweet.find(tok, pos)
            if start == -1:
                start = pos
            for c in range(start, start + len(tok)):
                offset_map[c] = i
            pos = start + len(tok)

        lab = [0] * len(toks)
        for ent in obj.get("entities", []):
            if (
                ent.get("entity_type", "LOC").upper()
                not in {"LOCATION", "LOC", "GPE"}
            ):
                continue
            if "indices" in ent:
                s, e = ent["indices"]
            elif ent.get("char_position") and ent.get("text"):
                s = int(float(ent["char_position"]))
                e = s + len(ent["text"])
            else:
                continue
            for c in range(s, e):
                if c in offset_map:
                    lab[offset_map[c]] = 1

        texts.append(tweet)
        tags.append(lab)
    return texts, tags


# ------------------------------------------------------------------#
# OpenAI chat wrapper + SHA-1 cache                                 #
# ------------------------------------------------------------------#
client = OpenAI()
CACHE_DIR = Path("cache_llm")
CACHE_DIR.mkdir(exist_ok=True)

SYSTEM_MSG = (
    "You are a helpful assistant that extracts location names "
    "(toponyms) from text. Return them as a JSON array of strings, "
    "exactly as they appear in the text."
)

FEW_SHOT = {
    "text": "The UN says fighting near Kharkiv and in Lviv displaced thousands.",
    "locs": ["Kharkiv", "Lviv"],
}


def llm_extract(text: str, shots: int) -> List[str]:
    key = sha1(f"{shots}_{text}".encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    messages = [{"role": "system", "content": SYSTEM_MSG}]
    if shots:
        messages.append({"role": "user", "content": FEW_SHOT["text"]})
        messages.append({"role": "assistant", "content": json.dumps(FEW_SHOT["locs"])})
    messages.append({"role": "user", "content": text})

    rsp = client.chat.completions.create(
        model="o4-mini", messages=messages
    )
    try:
        locs = json.loads(rsp.choices[0].message.content)
    except Exception:
        locs = []

    cache_file.write_text(json.dumps(locs))
    return locs


# ------------------------------------------------------------------#
# evaluation (token-level)                                          #
# ------------------------------------------------------------------#
def evaluate(texts: List[str], gold: List[List[int]], shots: int):
    y_true, y_pred = [], []
    for txt, g_tags in tqdm(list(zip(texts, gold)), desc="LLM extract"):
        toks = tokenize(txt)
        preds = [1 if t in set(llm_extract(txt, shots)) else 0 for t in toks]
        y_true.extend(g_tags)
        y_pred.extend(preds)

    print(f"\n[LLM] o4-mini  | shots = {shots}")
    print(
        classification_report(
            y_true, y_pred, target_names=["O", "LOC"], digits=3
        )
    )


# ------------------------------------------------------------------#
# CLI                                                               #
# ------------------------------------------------------------------#
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geocorpora", required=True)
    p.add_argument("--shots", type=int, default=0, choices=[0, 1])
    p.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="max tweets to evaluate (default 1000)",
    )
    args = p.parse_args()

    texts, tags = load_gc(args.geocorpora)
    if args.limit and len(texts) > args.limit:
        rnd = random.Random(42)
        idx = rnd.sample(range(len(texts)), args.limit)
        texts = [texts[i] for i in idx]
        tags = [tags[i] for i in idx]

    evaluate(texts, tags, args.shots)


if __name__ == "__main__":
    main()
