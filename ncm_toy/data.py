"""Synthetic toy-grammar corpus (character level) so everything runs offline.

Sentences look like  "the small cat chased a dog."  The grammar is finite and
known, so we can *measure* generation quality exactly: a sample is "valid" if
it parses under the grammar, "novel" if it was never in the training split,
and (for the conditional task) "faithful" if it matches the keyword prefix.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch

DETS = ["the", "a"]
ADJS = ["red", "big", "small", "old"]
NOUNS = ["cat", "dog", "bird", "fox", "fish"]
VERBS = ["saw", "chased", "liked", "fed"]

SEQ_LEN = 32      # target length (power of two -> clean scale pyramid)
PREFIX_LEN = 16   # keyword prefix length for the conditional task
CHARS = "_ abcdefghijklmnopqrstuvwxyz."  # index 0 ('_') doubles as PAD
PAD = 0
VOCAB = len(CHARS)
_STOI = {c: i for i, c in enumerate(CHARS)}

_ADJ = "|".join(ADJS)
_NOUN = "|".join(NOUNS)
_DET = "|".join(DETS)
_VERB = "|".join(VERBS)
SENTENCE_RE = re.compile(
    rf"^({_DET}) (?:({_ADJ}) )?({_NOUN}) ({_VERB}) ({_DET}) (?:({_ADJ}) )?({_NOUN})\.$"
)


def encode(s: str, length: int) -> list[int]:
    assert len(s) <= length, f"{s!r} longer than {length}"
    return [_STOI[c] for c in s] + [PAD] * (length - len(s))


def decode(ids, keep_pad: bool = False) -> str:
    s = "".join(CHARS[int(i)] for i in ids)
    return s if keep_pad else s.rstrip("_").replace("_", "·")


def parse(s: str):
    """Return (subj_noun, verb, obj_noun) if `s` is a grammatical sentence, else None."""
    m = SENTENCE_RE.match(s)
    if not m:
        return None
    return m.group(3), m.group(4), m.group(7)


def all_sentences() -> list[str]:
    nps = [f"{d} {a + ' ' if a else ''}{n}" for d in DETS for a in [""] + ADJS for n in NOUNS]
    out = [f"{s} {v} {o}." for s in nps for v in VERBS for o in nps]
    return [s for s in out if len(s) <= SEQ_LEN]


def keyword_prefix(s: str) -> str:
    subj, verb, obj = parse(s)
    return f"{subj} {verb} {obj}"


@dataclass
class Split:
    sentences: list[str]
    x: torch.Tensor  # (N, SEQ_LEN) target tokens
    p: torch.Tensor  # (N, PREFIX_LEN) keyword prefix tokens (empty width for task="lm")

    def __len__(self):
        return len(self.sentences)

    def batch(self, n: int, gen: torch.Generator | None = None):
        i = torch.randint(0, len(self), (n,), generator=gen)
        return self.x[i], self.p[i]


def load_splits(task: str = "lm", seed: int = 0, train_frac: float = 0.85) -> tuple[Split, Split]:
    assert task in ("lm", "cond")
    sents = all_sentences()
    random.Random(seed).shuffle(sents)
    cut = int(len(sents) * train_frac)

    def mk(ss):
        x = torch.tensor([encode(s, SEQ_LEN) for s in ss])
        if task == "cond":
            p = torch.tensor([encode(keyword_prefix(s), PREFIX_LEN) for s in ss])
        else:
            p = torch.zeros(len(ss), 0, dtype=torch.long)
        return Split(ss, x, p)

    return mk(sents[:cut]), mk(sents[cut:])
