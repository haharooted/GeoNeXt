#!/usr/bin/env python
"""

"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd
from packaging.version import parse as V

# ------------------------------------------------------------------#
# external libs (fail fast if missing)                              #
# ------------------------------------------------------------------#
try:
    import datasets  # noqa: F401 –  imported lazily elsewhere
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Sequence,
        Value,
        ClassLabel,
        load_dataset,
    )
except ImportError as e:  # pragma: no cover
    sys.exit(f"Please `pip install datasets pandas numpy scikit-learn` → {e}")

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

# ------------------------------------------------------------------#
# global hyper-params                                               #
# ------------------------------------------------------------------#
WINDOW = 2               # tokens each side → 5-gram context
THRESH = 0.5            # probability cutoff for LOC vs O
RAND_SEED = 42
ART_DIR = Path("artifacts")

# ------------------------------------------------------------------#
# crude tokeniser (good enough for BoW)                             #
# ------------------------------------------------------------------#
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def offset2tok(text: str, tokens: List[str]) -> dict[int, int]:
    """Map every character offset in *text* to its token index."""
    mapping, pos = {}, 0
    for idx, tok in enumerate(tokens):
        start = text.find(tok, pos)
        if start == -1:
            start = pos
        for c in range(start, start + len(tok)):
            mapping[c] = idx
        pos = start + len(tok)
    return mapping


# ------------------------------------------------------------------#
# GeoCorpora readers                                                #
# ------------------------------------------------------------------#
def read_gc_tsv(path: Path) -> List[dict]:
    """Original GeoCorpora TSV (each LOC row, char_position + text)."""
    df = pd.read_csv(
        path,
        sep="\t",
        encoding="latin1",
        quoting=csv.QUOTE_NONE,
        usecols=["tweet_id_str", "tweet_text", "char_position", "text"],
        on_bad_lines="skip",
    ).dropna(subset=["tweet_text", "text"])

    examples: List[dict] = []
    for _, grp in df.groupby("tweet_id_str"):
        tweet = str(grp["tweet_text"].iloc[0])
        toks = tokenize(tweet)
        o2t = offset2tok(tweet, toks)
        tags = np.zeros(len(toks), dtype=int)

        for _, row in grp.iterrows():
            loc = str(row["text"])
            try:
                start = int(float(row["char_position"]))
            except Exception:
                m = re.search(re.escape(loc), tweet, flags=re.I)
                if not m:
                    continue
                start = m.start()
            for c in range(start, start + len(loc)):
                if c in o2t:
                    tags[o2t[c]] = 1

        examples.append({"tokens": toks, "ner_tags": tags.tolist()})
    return examples


def read_gc_json(path: Path) -> List[dict]:
    """Parse Kaggle-style JSON array *or* JSONL GeoCorpora dump."""
    txt = re.sub(r":\s*NaN", ": null", path.read_text("utf-8"))
    records: Iterable[dict]
    if txt.lstrip().startswith("["):
        records = json.loads(txt)
    else:
        records = (json.loads(l) for l in txt.splitlines() if l.strip())

    out: List[dict] = []
    for obj in records:
        tweet = obj.get("text") or obj.get("tweet_text") or ""
        if not tweet:
            continue
        toks = tokenize(tweet)
        o2t = offset2tok(tweet, toks)
        tags = np.zeros(len(toks), dtype=int)

        for ent in obj.get("entities", []):
            etype = (ent.get("entity_type") or "").upper()
            if etype and etype not in {"LOCATION", "LOC", "GPE"}:
                continue

            if "indices" in ent and len(ent["indices"]) == 2:
                s, e = ent["indices"]
            elif ent.get("char_position") not in {None, ""} and ent.get("text"):
                try:
                    s = int(float(ent["char_position"]))
                except Exception:
                    continue
                e = s + len(ent["text"])
            else:
                continue

            for c in range(s, e):
                if c in o2t:
                    tags[o2t[c]] = 1

        out.append({"tokens": toks, "ner_tags": tags.tolist()})
    return out


def load_geocorpora(path_like: str | Path, split: float = 0.8) -> DatasetDict:
    """Return HF DatasetDict(train,test) from GeoCorpora file/dir."""
    p = Path(path_like)
    if p.is_dir():
        for ext in (".jsonl", ".json", ".tsv"):
            files = list(p.glob(f"*{ext}"))
            if files:
                p = files[0]
                break
        else:
            sys.exit(f"No GeoCorpora file found in {p}")

    examples = read_gc_tsv(p) if p.suffix.lower() == ".tsv" else read_gc_json(p)
    if not examples:
        sys.exit("GeoCorpora parse produced 0 examples — file corrupt?")

    random.Random(RAND_SEED).shuffle(examples)
    cut = int(len(examples) * split)
    train, test = examples[:cut], examples[cut:]

    feats = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=["O", "LOC"])),
        }
    )
    return DatasetDict(
        {
            "train": Dataset.from_list(train, feats),
            "test": Dataset.from_list(test, feats),
        }
    )


# ------------------------------------------------------------------#
# Bag-of-Words helpers                                              #
# ------------------------------------------------------------------#
def context(tokens: List[str], i: int) -> str:
    left = max(0, i - WINDOW)
    right = i + WINDOW + 1
    return " ".join(tokens[left:right])


def flatten(split, loc_ids):
    docs, y = [], []
    for ex in split:
        for i, tag in enumerate(ex["ner_tags"]):
            docs.append(context(ex["tokens"], i))
            y.append(1 if tag in loc_ids else 0)
    return docs, y


def add_caps(mat, docs):
    caps = [[doc.split()[WINDOW].istitle()] for doc in docs]
    return sparse.hstack([mat, caps], format="csr")


# ------------------------------------------------------------------#
# training / evaluation                                             #
# ------------------------------------------------------------------#
def run(ds: DatasetDict, label: str):
    loc_ids = [1]  # LOC index in our two-class label list
    Xtr_raw, ytr = flatten(ds["train"], loc_ids)

    vec = CountVectorizer(min_df=1, ngram_range=(1, 2), lowercase=False)
    Xtr = add_caps(vec.fit_transform(Xtr_raw), Xtr_raw)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, ytr)

    Xte_raw, yte = flatten(ds["test"], loc_ids)
    Xte = add_caps(vec.transform(Xte_raw), Xte_raw)
    preds = (clf.predict_proba(Xte)[:, 1] >= THRESH).astype(int)

    print(f"\n[ EVAL ] {label}\n" + classification_report(yte, preds, target_names=["O", "LOC"], digits=3))

    ART_DIR.mkdir(exist_ok=True)
    joblib.dump(vec, ART_DIR / f"vectorizer_{label}.joblib")
    joblib.dump(clf, ART_DIR / f"logreg_{label}.joblib")


# ------------------------------------------------------------------#
# CLI                                                               #
# ------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="conll2003", help="HF dataset (ignored if --geocorpora)")
    ap.add_argument("--geocorpora", help="Path to GeoCorpora JSON/TSV/dir")
    args = ap.parse_args()

    if args.geocorpora:
        geo = load_geocorpora(args.geocorpora)
        run(geo, "geocorpora")
    else:
        if (
            args.dataset.lower() == "conll2003"
            and V(datasets.__version__) >= V("2.19.0")
        ):
            sys.exit(
                "datasets ≥2.19 cannot load CoNLL-2003 script. "
                "Run `pip install \"datasets<2.19\"` or pass --geocorpora."
            )
        conll = load_dataset(args.dataset)
        run(conll, "conll2003")


if __name__ == "__main__":
    main()
