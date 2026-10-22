"""Stage 2: next-concept generator.

One transformer predicts the code hierarchy coarse -> fine.  Every code inside a
level is sampled *in parallel*, so generating a sentence costs one forward pass
per level (6 here) instead of one per character (32 for an AR model).

Layout of the target sequence (L=32, scales 1,2,4,8,16,32 -> 63 positions):

    [prefix chars] | lvl0 (1) | lvl1 (2) | lvl2 (4) | ... | lvl5 (32)

* the *input* at level k is the running reconstruction z_hat_{k-1}, pooled down
  to n_k positions (level 0 gets a learned start token) -- VAR-style next-scale
  prediction, here applied to text concepts;
* attention is block-causal over levels (a level sees itself and coarser levels);
* the prefix attends only to itself, targets attend to the whole prefix;
* each level has its own output head over that level's codebook.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampling import sample_logits
from .vq import TextVQVAE, down, _tf


class NextConceptGenerator(nn.Module):
    def __init__(self, vqvae: TextVQVAE, d_model=192, layers=4, heads=4, prefix_len=0,
                 vocab=None, null_prob=0.1):
        super().__init__()
        self.cfg = dict(d_model=d_model, layers=layers, heads=heads, prefix_len=prefix_len,
                        vocab=vocab, null_prob=null_prob)
        self.vq_model = vqvae
        for p in vqvae.parameters():
            p.requires_grad_(False)
        vqvae.eval()
        self.scales, self.P, self.null_prob = vqvae.vq.scales, prefix_len, null_prob
        self.nlev, self.T = len(self.scales), sum(self.scales)
        D = vqvae.vq.dim
        self.start = nn.Parameter(torch.randn(d_model) * 0.02)
        self.in_proj = nn.Linear(D, d_model)
        self.pos = nn.Parameter(torch.randn(self.T, d_model) * 0.02)
        self.lvl = nn.Embedding(self.nlev, d_model)
        if prefix_len:
            self.p_tok = nn.Embedding(vocab, d_model)
            self.p_pos = nn.Parameter(torch.randn(prefix_len, d_model) * 0.02)
        self.tf = _tf(d_model, layers, heads)
        self.norm = nn.LayerNorm(d_model)
        self.heads_out = nn.ModuleList([nn.Linear(d_model, k) for k in self.vq.sizes])
        lvl_of = torch.cat([torch.full((n,), i) for i, n in enumerate(self.scales)])
        self.register_buffer("lvl_of", lvl_of, persistent=False)
        self.register_buffer("mask", self._build_mask(lvl_of), persistent=False)

    @property
    def vq(self):
        return self.vq_model.vq

    def train(self, mode=True):
        super().train(mode)
        self.vq_model.eval()  # tokenizer stays frozen
        return self

    def _build_mask(self, lvl_of):
        P, T = self.P, self.T
        blocked = torch.zeros(P + T, P + T, dtype=torch.bool)  # True = may NOT attend
        blocked[:P, P:] = True                                # prefix never sees targets
        blocked[P:, P:] = lvl_of[None, :] > lvl_of[:, None]   # block-causal over levels
        return blocked

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # -- core ---------------------------------------------------------------------------
    def _target_inputs(self, hats, B, upto):
        """Inputs for levels 0..upto. hats[i] = z_hat after level i."""
        pieces = [self.start.expand(B, 1, -1)]
        for k in range(1, upto + 1):
            pieces.append(self.in_proj(down(hats[k - 1], self.scales[k])))
        x = torch.cat(pieces, 1)
        n = x.shape[1]
        return x + self.pos[:n] + self.lvl(self.lvl_of[:n])

    def _run(self, tgt, prefix):
        n = tgt.shape[1]
        if self.P:
            pre = self.p_tok(prefix) + self.p_pos
            seq = torch.cat([pre, tgt], 1)
        else:
            seq = tgt
        L = self.P + n
        h = self.tf(seq, mask=self.mask[:L, :L])
        return self.norm(h[:, self.P:])

    def _level_logits(self, h, k):
        a = sum(self.scales[:k])
        return self.heads_out[k](h[:, a:a + self.scales[k]])

    # -- training -------------------------------------------------------------------------
    def forward(self, x, prefix=None):
        """Teacher-forced loss. `x`: (B, L) target tokens, `prefix`: (B, P) or None."""
        B = x.shape[0]
        idxs, hats = self.vq_model.encode(x)
        if self.P and self.training and self.null_prob > 0:  # CFG: randomly drop the condition
            drop = torch.rand(B, device=x.device) < self.null_prob
            prefix = torch.where(drop[:, None], torch.zeros_like(prefix), prefix)
        h = self._run(self._target_inputs(hats, B, self.nlev - 1), prefix)
        per_level, accs = [], []
        for k in range(self.nlev):
            lg = self._level_logits(h, k)
            per_level.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), idxs[k].reshape(-1)))
            accs.append((lg.argmax(-1) == idxs[k]).float().mean())
        pl = torch.stack(per_level)
        return dict(loss=pl.mean(), per_level=pl.detach(), acc=torch.stack(accs).detach())

    # -- sampling -------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, n=None, prefix=None, temperature=1.0, top_k=0, top_p=1.0, cfg=0.0,
                 level_temps=None):
        """Coarse-to-fine sampling.  Exactly `n_levels` transformer forward passes.

        cfg: classifier-free guidance scale; as in the reference code the strength ramps
             linearly with depth, t_k = cfg * k / (n_levels-1), logits = (1+t)*cond - t*uncond.
        Returns dict(tokens, codes, trace) where trace[k] is the text decoded after level k.
        """
        self.eval()
        dev = self.start.device
        B = prefix.shape[0] if prefix is not None else n
        use_cfg = cfg > 0 and self.P > 0
        if use_cfg:
            prefix = torch.cat([prefix, torch.zeros_like(prefix)], 0)
        z_hat = torch.zeros(B, self.vq.seq_len, self.vq.dim, device=dev)
        hats, codes, trace = [], [], []
        for k in range(self.nlev):
            tgt = self._target_inputs(hats, B, k)
            if use_cfg:
                tgt = torch.cat([tgt, tgt], 0)
            h = self._run(tgt, prefix)
            lg = self._level_logits(h, k)
            if use_cfg:
                t = cfg * k / max(self.nlev - 1, 1)
                cond, unc = lg[:B], lg[B:]
                lg = (1 + t) * cond - t * unc
            temp = level_temps[k] if level_temps is not None else temperature
            idx = sample_logits(lg, temp, top_k, top_p)
            z_hat = self.vq.add_level(k, idx, z_hat)
            hats.append(z_hat)
            codes.append(idx)
            trace.append(self.vq_model.decode_z(z_hat).argmax(-1))
        return dict(tokens=trace[-1], codes=codes, trace=trace, nfe=self.nlev)
