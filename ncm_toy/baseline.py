"""Small autoregressive char-level baseline: one forward pass *per character*."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampling import sample_logits
from .vq import _tf


class ARBaseline(nn.Module):
    def __init__(self, vocab, seq_len=32, prefix_len=0, d_model=192, layers=4, heads=4):
        super().__init__()
        self.cfg = dict(vocab=vocab, seq_len=seq_len, prefix_len=prefix_len,
                        d_model=d_model, layers=layers, heads=heads)
        self.vocab, self.L, self.P = vocab, seq_len, prefix_len
        self.bos = vocab  # extra id
        self.tok = nn.Embedding(vocab + 1, d_model)
        self.pos = nn.Parameter(torch.randn(prefix_len + seq_len, d_model) * 0.02)
        self.tf = _tf(d_model, layers, heads)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab)
        n = prefix_len + seq_len
        self.register_buffer("mask", torch.triu(torch.ones(n, n, dtype=torch.bool), 1), persistent=False)

    def _logits(self, seq):
        n = seq.shape[1]
        h = self.tf(self.tok(seq) + self.pos[:n], mask=self.mask[:n, :n])
        return self.out(self.norm(h))

    def forward(self, x, prefix=None):
        B = x.shape[0]
        parts = ([prefix] if self.P else []) + [torch.full((B, 1), self.bos, device=x.device), x[:, :-1]]
        lg = self._logits(torch.cat(parts, 1))[:, self.P:]
        loss = F.cross_entropy(lg.reshape(-1, self.vocab), x.reshape(-1))
        return dict(loss=loss, acc=(lg.argmax(-1) == x).float().mean())

    @torch.no_grad()
    def generate(self, n=None, prefix=None, temperature=1.0, top_k=0, top_p=1.0, **_):
        self.eval()
        dev = self.bos_device()
        B = prefix.shape[0] if prefix is not None else n
        seq = torch.full((B, 1), self.bos, device=dev)
        if self.P:
            seq = torch.cat([prefix, seq], 1)
        for _i in range(self.L):  # no KV cache: simple, and O(L^2) -- see README
            nxt = sample_logits(self._logits(seq)[:, -1], temperature, top_k, top_p)
            seq = torch.cat([seq, nxt[:, None]], 1)
        return dict(tokens=seq[:, self.P + 1:], nfe=self.L)

    def bos_device(self):
        return self.out.weight.device
