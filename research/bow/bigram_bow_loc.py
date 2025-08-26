

from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset  # 🤗 Datasets – pip install datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report as sk_report
import joblib  # for model persistence

WINDOW = 2  # tokens on each side  → 5‑gram window

# ---------------------------------------------------------------------
# 1. Data helpers
# ---------------------------------------------------------------------

def window_string(tokens: List[str], i: int, span: int = WINDOW) -> str:
    """Return the (2·span+1)‑token context, joined by spaces."""
    left = max(0, i - span)
    right = i + span + 1
    return " ".join(tokens[left:right])


def flatten_split(split) -> Tuple[List[str], List[int]]:
    """Convert a CoNLL split to lists of window‑strings and binary labels."""
    docs, y = [], []
    for ex in split:
        toks, tags = ex["tokens"], ex["ner_tags"]
        for i, tag in enumerate(tags):
            docs.append(window_string(toks, i))
            y.append(1 if tag in (7, 8) else 0)  # 7=B‑LOC, 8=I‑LOC
    return docs, y

# ---------------------------------------------------------------------
# 2. Train & evaluate
# ---------------------------------------------------------------------

def train():
        # HuggingFace Datasets ≥ 2.19 blocks script-based datasets unless you
    # explicitly opt‑in.  We therefore add `trust_remote_code=True` to load
    # the off‑icial CoNLL‑2003 builder.
    ds = load_dataset("conll2003", trust_remote_code=True)
    X_tr, y_tr = flatten_split(ds["train"])
    X_dev, y_dev = flatten_split(ds["validation"])

    vec = CountVectorizer(min_df=2, ngram_range=(1, 2), lowercase=False)
    Xtr_mat = vec.fit_transform(X_tr)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr_mat, y_tr)

    # quick dev evaluation (token‑level)
    Xdev_mat = vec.transform(X_dev)
    preds = clf.predict(Xdev_mat)
    print(sk_report(y_dev, preds, target_names=["O", "LOC"], digits=3))

    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(vec, "artifacts/vectorizer.joblib")
    joblib.dump(clf, "artifacts/logreg_loc.joblib")


# ---------------------------------------------------------------------
# 3. Inference convenience
# ---------------------------------------------------------------------

def load_model(vec_path="artifacts/vectorizer.joblib", clf_path="artifacts/logreg_loc.joblib"):
    vec = joblib.load(vec_path)
    clf = joblib.load(clf_path)
    return vec, clf


def tag_tokens(tokens: List[str], vec, clf, span: int = WINDOW) -> List[str]:
    """Return a list of BIO predictions for one sentence."""
    docs = [window_string(tokens, i, span) for i in range(len(tokens))]
    X = vec.transform(docs)
    preds = clf.predict(X)

    # convert binary predictions to BIO scheme
    bio = []
    for p, t in itertools.zip_longest(preds, tokens):
        if p == 1:
            bio.append("B-LOC" if not bio or bio[-1] == "O" else "I-LOC")
        else:
            bio.append("O")
    return bio


if __name__ == "__main__":
    train()
