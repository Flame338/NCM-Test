from __future__ import annotations

from . import data


def evaluate_texts(texts, train_sentences, prefixes=None):
    """Exact, grammar-based quality metrics for generated sentences."""
    n = len(texts)
    parsed = [data.parse(t) for t in texts]
    train = set(train_sentences)
    valid = [p is not None for p in parsed]
    out = dict(
        valid=sum(valid) / n,
        novel_valid=sum(v and t not in train for v, t in zip(valid, texts)) / n,
        unique=len(set(texts)) / n,
    )
    if prefixes is not None:
        ok = [p is not None and f"{p[0]} {p[1]} {p[2]}" == pre for p, pre in zip(parsed, prefixes)]
        out["faithful"] = sum(ok) / n
    return out


def decode_batch(tokens):
    return [data.decode(r) for r in tokens]
