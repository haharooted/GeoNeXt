#!/usr/bin/env python
"""

"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from hashlib import sha1
from pathlib import Path
from typing import List, Tuple

import numpy as np
from openai import AsyncOpenAI, RateLimitError
from sklearn.metrics import classification_report
from tqdm.asyncio import tqdm_asyncio

# ──────────────────────────────────────────────────────────────────────
# tokeniser (same regex as BoW baseline)
# ──────────────────────────────────────────────────────────────────────
TOKEN_RE = re.compile(r"\w+|[^\w\s]")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)

# ──────────────────────────────────────────────────────────────────────
# minimal GeoCorpora JSON loader (array or JSONL)
# ──────────────────────────────────────────────────────────────────────
def load_gc(path: str) -> Tuple[List[str], List[List[int]]]:
    txt = re.sub(r":\s*NaN", ": null", Path(path).read_text("utf-8"))
    recs = json.loads(txt) if txt.lstrip().startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]

    texts, tags = [], []
    for obj in recs:
        tweet = obj.get("text") or obj.get("tweet_text")
        if not tweet:
            continue
        toks = tokenize(tweet)

        # char-offset → token index
        o2t, pos = {}, 0
        for i, tok in enumerate(toks):
            start = tweet.find(tok, pos)
            if start == -1:
                start = pos
            for c in range(start, start + len(tok)):
                o2t[c] = i
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
                if c in o2t:
                    lab[o2t[c]] = 1

        texts.append(tweet)
        tags.append(lab)
    return texts, tags

# ──────────────────────────────────────────────────────────────────────
# OpenAI async wrapper + SHA-1 cache
# ──────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache_llm"); CACHE_DIR.mkdir(exist_ok=True)
client = AsyncOpenAI()

SYSTEM = (
    "You are a meticulous information extraction assistant.\n"
    "Task: identify every word or phrase that is a GEOgraphic location "
    "(country, city, village, region, river, mountain, etc.). "
    "Return **only** the location names as a JSON array of strings, "
    "exactly as they appear.\n"
    "Do NOT return organisations, person names, weekdays, months, or plain nouns.\n"
    "Note: in this dataset only ~2 % of words are locations, so be conservative."
)

EX1 = {
    "text": "Evacuation centres opened in Lviv and Chernihiv after the floods.",
    "locs": ["Lviv", "Chernihiv"],
}
EX2 = {
    "text": "Clashes erupted near the southern outskirts of New York City late Monday.",
    "locs": ["New York City"],
}

async def llm_extract(text: str, shots: int, sema: asyncio.Semaphore) -> List[str]:
    key = sha1(f"{shots}_{text}".encode()).hexdigest()[:16]
    cpath = CACHE_DIR / f"{key}.json"
    if cpath.exists():
        return json.loads(cpath.read_text())

    msgs = [{"role": "system", "content": SYSTEM}]
    if shots >= 1:
        msgs += [
            {"role": "user", "content": EX1["text"]},
            {"role": "assistant", "content": json.dumps(EX1["locs"])},
        ]
    if shots == 2:
        msgs += [
            {"role": "user", "content": EX2["text"]},
            {"role": "assistant", "content": json.dumps(EX2["locs"])},
        ]
    msgs.append({"role": "user", "content": text})

    backoff = 1.0
    while True:
        async with sema:
            try:
                rsp = await client.chat.completions.create(
                    model="o4-mini",
                    messages=msgs,
                    top_p=0.0,
                )
                break
            except RateLimitError:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    try:
        locs = json.loads(rsp.choices[0].message.content or "[]")
    except Exception:
        locs = []

    cpath.write_text(json.dumps(locs))
    return locs

# ──────────────────────────────────────────────────────────────────────
# evaluation
# ──────────────────────────────────────────────────────────────────────
async def evaluate(texts: List[str], gold: List[List[int]], shots: int, concur: int):
    sema = asyncio.Semaphore(concur)
    tasks = [llm_extract(t, shots, sema) for t in texts]
    preds = await tqdm_asyncio.gather(*tasks, desc="LLM extract")

    y_true, y_pred = [], []
    for txt, g, locs in zip(texts, gold, preds):
        tset = set(locs)
        y_true.extend(g)
        y_pred.extend([1 if tok in tset else 0 for tok in tokenize(txt)])

    print(f"\n[LLM] o4-mini | examples = {shots} | concurrency = {concur}")
    print(classification_report(y_true, y_pred, target_names=["O", "LOC"], digits=3))

# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geocorpora", required=True)
    ap.add_argument("--examples", type=int, default=0, choices=[0, 1, 2],
                    help="number of in-context examples (0-2)")
    ap.add_argument("--limit", type=int, default=1000,
                    help="max tweets to evaluate (default 1000)")
    ap.add_argument("--concurrency", type=int, default=100,
                    help="parallel OpenAI calls (default 100)")
    args = ap.parse_args()

    texts, tags = load_gc(args.geocorpora)
    if args.limit and len(texts) > args.limit:
        rnd = random.Random(42)
        pick = rnd.sample(range(len(texts)), args.limit)
        texts = [texts[i] for i in pick]
        tags = [tags[i] for i in pick]

    asyncio.run(evaluate(texts, tags, args.examples, args.concurrency))

if __name__ == "__main__":
    main()
