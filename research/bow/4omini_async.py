#!/usr/bin/env python
"""
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from hashlib import sha1
from pathlib import Path
from typing import List, Tuple

import numpy as np
from openai import AsyncOpenAI, RateLimitError
from sklearn.metrics import classification_report
from tqdm.asyncio import tqdm_asyncio  # progress bar for coroutines

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
    """Return tweet texts and gold BIO tags (0=O, 1=LOC)."""
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
            if ent.get("entity_type", "LOC").upper() not in {"LOCATION", "LOC", "GPE"}:
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


async def llm_extract_async(
    text: str, shots: int, client: AsyncOpenAI, sema: asyncio.Semaphore
) -> List[str]:
    """One-tweet extraction with caching + rate-limit retries."""
    key = sha1(f"{shots}_{text}".encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    messages = [{"role": "system", "content": SYSTEM_MSG}]
    if shots:
        messages += [
            {"role": "user", "content": FEW_SHOT["text"]},
            {"role": "assistant", "content": json.dumps(FEW_SHOT["locs"])},
        ]
    messages.append({"role": "user", "content": text})

    backoff = 1.0
    while True:  # retry loop
        async with sema:  # cap concurrency
            try:
                rsp = await client.chat.completions.create(
                    model="o4-mini", messages=messages
                )
                break
            except RateLimitError:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)  # ceil at 30 s

    try:
        locs = json.loads(rsp.choices[0].message.content or "[]")
    except Exception:
        locs = []

    cache_file.write_text(json.dumps(locs))
    return locs


# ------------------------------------------------------------------#
# asynchronous evaluation (token-level)                             #
# ------------------------------------------------------------------#
async def evaluate_async(
    texts: List[str],
    gold: List[List[int]],
    shots: int,
    concurrency: int,
):
    client = AsyncOpenAI()
    sema = asyncio.Semaphore(concurrency)

    # launch all requests
    tasks = [
        llm_extract_async(txt, shots, client, sema) for txt in texts
    ]
    all_locs = await tqdm_asyncio.gather(*tasks, desc="LLM extract")

    # flatten predictions + gold for classification_report
    y_true, y_pred = [], []
    for txt, g_tags, locs in zip(texts, gold, all_locs):
        toks = tokenize(txt)
        loc_set = set(locs)
        y_true.extend(g_tags)
        y_pred.extend([1 if t in loc_set else 0 for t in toks])

    print(f"\n[LLM] o4-mini | shots = {shots} | concurrency = {concurrency}")
    print(classification_report(y_true, y_pred, target_names=["O", "LOC"], digits=3))


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
    p.add_argument(
        "--concurrency",
        type=int,
        default=100,
        help="simultaneous OpenAI requests (default 100)",
    )
    args = p.parse_args()

    texts, tags = load_gc(args.geocorpora)
    if args.limit and len(texts) > args.limit:
        rnd = random.Random(42)
        idx = rnd.sample(range(len(texts)), args.limit)
        texts = [texts[i] for i in idx]
        tags = [tags[i] for i in idx]

    asyncio.run(evaluate_async(texts, tags, args.shots, args.concurrency))


if __name__ == "__main__":
    main()
