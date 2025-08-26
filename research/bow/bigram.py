#!/usr/bin/env python
"""
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tabulate import tabulate
from tqdm.auto import tqdm

# ────────────────────────────────────────────────────────────────── #
# hyper-parameters                                                  #
# ────────────────────────────────────────────────────────────────── #
NGRAM_RANGE = (1, 2)
WINDOW = 2          # tokens on each side (total 2*WINDOW+1 context)
THRESH = 0.5        # probability cutoff for LOC vs O
ART_DIR = Path("artifacts")
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


# ────────────────────────────────────────────────────────────────── #
# GeoCorpora loader (same as in LLM script, but yields token lists) #
# ────────────────────────────────────────────────────────────────── #
def load_geocorpora(path: str | Path) -> Tuple[list[list[str]], list[list[int]]]:
    raw = re.sub(r":\s*NaN", ": null", Path(path).read_text("utf-8"))
    records = (
        json.loads(raw)
        if raw.lstrip().startswith("[")
        else [json.loads(l) for l in raw.splitlines() if l.strip()]
    )

    tokens, tags = [], []
    for obj in records:
        tweet = obj.get("text") or obj.get("tweet_text") or ""
        if not tweet:
            continue
        toks = tokenize(tweet)

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

        tokens.append(toks)
        tags.append(lab)
    return tokens, tags


# ────────────────────────────────────────────────────────────────── #
# helpers                                                           #
# ────────────────────────────────────────────────────────────────── #
def context(tokens: List[str], i: int) -> str:
    left = max(0, i - WINDOW)
    right = i + WINDOW + 1
    return " ".join(tokens[left:right])


def flatten(
    token_seqs: list[list[str]],
    tag_seqs: list[list[int]],
    add_caps: bool,
) -> Tuple[list[str], list[int], list[List[bool]]]:
    docs, y, caps = [], [], []
    for toks, tags in zip(token_seqs, tag_seqs):
        for i, tag in enumerate(tags):
            docs.append(context(toks, i))
            y.append(tag)
            caps.append([toks[i].istitle()] if add_caps else [])
    return docs, y, caps


def add_cap_feature(X: sparse.spmatrix, caps: list[List[bool]]) -> sparse.spmatrix:
    if not caps or not caps[0]:
        return X
    return sparse.hstack([X, np.array(caps, dtype=float)], format="csr")


# ────────────────────────────────────────────────────────────────── #
# training / evaluation                                             #
# ────────────────────────────────────────────────────────────────── #
def run(
    tokens: list[list[str]],
    tags: list[list[int]],
    seed: int,
    add_caps: bool,
) -> None:
    rnd = random.Random(seed)
    idx = list(range(len(tokens)))
    rnd.shuffle(idx)
    cut = int(len(idx) * 0.8)
    tr_idx, te_idx = idx[:cut], idx[cut:]

    Xtr_raw, ytr, caps_tr = flatten(
        [tokens[i] for i in tr_idx],
        [tags[i] for i in tr_idx],
        add_caps,
    )
    vec = CountVectorizer(
        lowercase=False,
        ngram_range=NGRAM_RANGE,
        min_df=1,
    )
    Xtr = vec.fit_transform(Xtr_raw)
    Xtr = add_cap_feature(Xtr, caps_tr)

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", n_jobs=-1, verbose=0
    )
    clf.fit(Xtr, ytr)

    Xte_raw, yte, caps_te = flatten(
        [tokens[i] for i in te_idx],
        [tags[i] for i in te_idx],
        add_caps,
    )
    Xte = vec.transform(Xte_raw)
    Xte = add_cap_feature(Xte, caps_te)

    preds = (clf.predict_proba(Xte)[:, 1] >= THRESH).astype(int)
    report = classification_report(
        yte, preds, target_names=["O", "LOC"], digits=3, output_dict=True
    )
    print(
        "\n" + tabulate(
            [("LOC", report["LOC"]["precision"], report["LOC"]["recall"], report["LOC"]["f1-score"])],
            headers=["Class", "Precision", "Recall", "F1"],
            floatfmt=".3f",
        )
    )

    ART_DIR.mkdir(exist_ok=True)
    joblib.dump(vec, ART_DIR / "vectorizer_bow.joblib")
    joblib.dump(clf, ART_DIR / "logreg_bow.joblib")


# ────────────────────────────────────────────────────────────────── #
# CLI                                                               #
# ────────────────────────────────────────────────────────────────── #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geocorpora", required=True, help="GeoCorpora JSON/JSONL")
    ap.add_argument("--seed", type=int, default=42, help="Train/test split seed")
    ap.add_argument(
        "--cap-feature",
        dest="cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add boolean feature for capitalised token",
    )
    args = ap.parse_args()

    toks, tags = load_geocorpora(args.geocorpora)
    if not toks:
        sys.exit("GeoCorpora file empty or unreadable.")
    run(toks, tags, seed=args.seed, add_caps=args.cap)


if __name__ == "__main__":
    main()
