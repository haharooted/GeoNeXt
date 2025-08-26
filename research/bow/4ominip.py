#!/usr/bin/env python
"""
llm_geocorpora_eval.py – Async evaluation of any OpenAI chat model on GeoCorpora
================================================================================

Improvements vs. 4omini_async.py
--------------------------------
* Works with *any* chat model (`--model`, default **o4-mini**)
* Accepts multiple shot counts in one run (`--shots 0 1 3 …`)
* Robust exponential back-off on **RateLimitError** *and* **APIError**
* Per-tweet SHA-1 cache remains under  `cache_llm/`  (same layout)
* Deterministic subsampling with `--limit` + `--seed`
* Much cleaner asyncio + tqdm glue
* Prints **precision / recall / F1** in one table for every shot setting

Usage
-----
export OPENAI_API_KEY="sk-…"                # set your key

# Zero-shot vs. 1-shot on 1 000 random tweets, 200 parallel calls
python llm_geocorpora_eval.py \
    --geocorpora geocorpora.json --shots 0 1 \
    --limit 1000 --concurrency 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from hashlib import sha1
from pathlib import Path
from typing import List

import numpy as np
from openai import AsyncOpenAI, APIError, RateLimitError
from sklearn.metrics import classification_report
from tabulate import tabulate
from tqdm.asyncio import tqdm_asyncio

# ────────────────────────────────────────────────────────────────── #
# crude tokenizer (regex identical to BoW baseline)                 #
# ────────────────────────────────────────────────────────────────── #
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


# ────────────────────────────────────────────────────────────────── #
# GeoCorpora loader (accepts JSON array *or* JSONL)                 #
# ────────────────────────────────────────────────────────────────── #
def load_geocorpora(path: str | Path) -> tuple[list[str], list[list[int]]]:
    """Return tweet texts and gold BIO tags (0 = O, 1 = LOC)."""
    raw = re.sub(r":\s*NaN", ": null", Path(path).read_text("utf-8"))
    records = (
        json.loads(raw)
        if raw.lstrip().startswith("[")
        else [json.loads(l) for l in raw.splitlines() if l.strip()]
    )

    texts, tags = [], []
    for obj in records:
        tweet = obj.get("text") or obj.get("tweet_text") or ""
        if not tweet:
            continue
        toks = tokenize(tweet)

        # char-offset → token index
        offset, pos = {}, 0
        for i, tok in enumerate(toks):
            start = tweet.find(tok, pos)
            if start == -1:
                start = pos
            for c in range(start, start + len(tok)):
                offset[c] = i
            pos = start + len(tok)

        lab = [0] * len(toks)
        for ent in obj.get("entities", []):
            if (ent.get("entity_type", "LOC").upper() not in {"LOCATION", "LOC", "GPE"}):
                continue
            if "indices" in ent:
                s, e = ent["indices"]
            elif ent.get("char_position") and ent.get("text"):
                s = int(float(ent["char_position"]))
                e = s + len(ent["text"])
            else:
                continue
            for c in range(s, e):
                if c in offset:
                    lab[offset[c]] = 1

        texts.append(tweet)
        tags.append(lab)
    return texts, tags


# ────────────────────────────────────────────────────────────────── #
# OpenAI wrapper + SHA-1 cache                                      #
# ────────────────────────────────────────────────────────────────── #
CACHE_DIR = Path("cache_llm")
CACHE_DIR.mkdir(exist_ok=True)

SYSTEM_MSG = (
    "You are a helpful assistant that extracts every location name "
    "(toponym) from the text. Return a JSON array of strings, "
    "exactly as they appear, including abbreviations or repeated mentions."
)


FEW_SHOT_EXAMPLE = {
    "text": "The UN says fighting near Kharkiv and in Lviv displaced thousands.",
    "locs": ["Kharkiv", "Lviv"],
}

FEW_SHOT_2 = {
    "text": "I left NYC, changed planes in LAX, and finally landed in New York City again.",
    "locs": ["NYC", "LAX", "New York City"],
}

async def extract_locs(
    client: AsyncOpenAI,
    text: str,
    shots: int,
    model: str,
    sema: asyncio.Semaphore,
) -> list[str]:
    """Single-tweet extraction with on-disk cache and retry logic."""
    cache_key = sha1(f"{model}|{shots}|{text}".encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    messages = [{"role": "system", "content": SYSTEM_MSG}]
    if shots:
        messages += [
            {"role": "user", "content": FEW_SHOT_EXAMPLE["text"]},
            {"role": "assistant", "content": json.dumps(FEW_SHOT_EXAMPLE["locs"])},
        ]
    messages.append({"role": "user", "content": text})

    backoff = 1.0
    while True:  # retry until success
        async with sema:
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=messages
                )
                break
            except (RateLimitError, APIError):
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    try:
        locs = json.loads(resp.choices[0].message.content or "[]")
    except Exception:
        locs = []

    cache_file.write_text(json.dumps(locs))
    return locs


# ────────────────────────────────────────────────────────────────── #
# async evaluation                                                  #
# ────────────────────────────────────────────────────────────────── #
async def evaluate(
    texts: list[str],
    gold: list[list[int]],
    model: str,
    shots: int,
    concurrency: int,
) -> tuple[str, dict]:
    client = AsyncOpenAI()
    sema = asyncio.Semaphore(concurrency)

    coro = [
        extract_locs(client, txt, shots, model, sema) for txt in texts
    ]
    loc_lists = await tqdm_asyncio.gather(*coro, desc=f"{model} | {shots}-shot")

    y_true, y_pred = [], []
    for txt, gold_tags, locs in zip(texts, gold, loc_lists):
        toks = tokenize(txt)
        loc_set = set(locs)
        y_true.extend(gold_tags)
        y_pred.extend([1 if t in loc_set else 0 for t in toks])

    report = classification_report(
        y_true, y_pred, target_names=["O", "LOC"], digits=3, output_dict=True
    )
    return f"{shots}-shot", report["LOC"]  # precision/recall/f1 for LOC only


# ────────────────────────────────────────────────────────────────── #
# CLI                                                               #
# ────────────────────────────────────────────────────────────────── #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--geocorpora", required=True, help="Path to GeoCorpora file")
    p.add_argument("--model", default="o4-mini", help="Chat model name")
    p.add_argument("--shots", type=int, nargs="+", default=[0], help="e.g. 0 1 2")
    p.add_argument("--limit", type=int, default=1000, help="Max tweets to eval")
    p.add_argument("--seed", type=int, default=42, help="Random seed for subsample")
    p.add_argument("--concurrency", type=int, default=100, help="Parallel requests")
    args = p.parse_args()

    texts, tags = load_geocorpora(args.geocorpora)
    if args.limit and len(texts) > args.limit:
        rnd = random.Random(args.seed)
        idx = rnd.sample(range(len(texts)), args.limit)
        texts = [texts[i] for i in idx]
        tags = [tags[i] for i in idx]

    # run every shot setting sequentially (same tweet subset & cache)
    results = {}
    for s in sorted(set(args.shots)):
        label, metrics = asyncio.run(
            evaluate(texts, tags, args.model, s, args.concurrency)
        )
        results[label] = metrics

    print("\n" + tabulate(
        [(k, v["precision"], v["recall"], v["f1-score"]) for k, v in results.items()],
        headers=["Shots", "Precision", "Recall", "F1"],
        floatfmt=".3f",
    ))


if __name__ == "__main__":
    main()
