"""Stage 1: multiscale residual VQ tokenizer ("concepts").

A sentence of L characters is encoded to latents z (L, D).  Level k looks at
the *residual* left by levels < k, average-pools it down to n_k positions
(1, 2, 4, ..., L), snaps each position to a codebook entry, upsamples the
result back to L and adds it (through a small conv "phi") to the running
reconstruction z_hat.  Early levels are therefore few, coarse "concepts";
the last level is character-resolution detail.  Codebooks use EMA updates
with dead-code restarts, as in the reference implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def down(x: torch.Tensor, n: int) -> torch.Tensor:
    """(B, L, D) -> (B, n, D) by area pooling."""
    return F.adaptive_avg_pool1d(x.transpose(1, 2), n).transpose(1, 2)


def up(x: torch.Tensor, length: int) -> torch.Tensor:
    """(B, n, D) -> (B, L, D) by linear interpolation."""
    if x.shape[1] == length:
        return x
    return F.interpolate(x.transpose(1, 2), size=length, mode="linear", align_corners=False).transpose(1, 2)


class MultiScaleResidualVQ(nn.Module):
    def __init__(self, dim, seq_len, scales, codebook_sizes, decay=0.99, beta=0.25, dead_frac=0.03):
        super().__init__()
        self.dim, self.seq_len, self.scales = dim, seq_len, list(scales)
        self.sizes = list(codebook_sizes)
        self.decay, self.beta, self.dead_frac = decay, beta, dead_frac
        for i, k in enumerate(self.sizes):
            self.register_buffer(f"cb{i}", torch.randn(k, dim) * 0.1)
            self.register_buffer(f"cs{i}", torch.ones(k))      # EMA cluster sizes
            self.register_buffer(f"es{i}", torch.randn(k, dim) * 0.1)  # EMA embed sums
            self.register_buffer(f"init{i}", torch.zeros(()))
        # phi_k: residual conv refining the upsampled code (zero-init => plain RVQ at start)
        self.phi = nn.ModuleList([nn.Conv1d(dim, dim, 3, padding=1) for _ in self.scales])
        for p in self.phi:
            nn.init.zeros_(p.weight)
            nn.init.zeros_(p.bias)

    @property
    def n_levels(self):
        return len(self.scales)

    def codebook(self, i):
        return getattr(self, f"cb{i}")

    # -- single level ------------------------------------------------------------
    def _lookup(self, i, flat):
        cb = self.codebook(i)
        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ cb.t() + cb.pow(2).sum(1)[None]
        return d.argmin(1)

    @torch.no_grad()
    def _ema_update(self, i, flat, idx):
        k = self.sizes[i]
        cb, cs, es = self.codebook(i), getattr(self, f"cs{i}"), getattr(self, f"es{i}")
        if getattr(self, f"init{i}") == 0:  # data-dependent init on first batch
            pick = flat[torch.randint(0, flat.shape[0], (k,), device=flat.device)]
            cb.copy_(pick); es.copy_(pick); cs.fill_(1.0)
            getattr(self, f"init{i}").fill_(1)
            return
        counts = torch.bincount(idx, minlength=k).float()
        sums = torch.zeros_like(es).index_add_(0, idx, flat)
        cs.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        es.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        n = cs.sum()
        cb.copy_(es / ((cs + 1e-5) / (n + k * 1e-5) * n)[:, None])
        dead = cs < self.dead_frac * flat.shape[0] / k
        if dead.any():  # restart unused codes on random residuals
            pick = flat[torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)]
            cb[dead] = pick; es[dead] = pick; cs[dead] = 1.0

    def add_level(self, i, idx, z_hat):
        """z_hat + phi_i(up(codebook_i[idx]))  -- also used by the generator at sampling time."""
        h = up(self.codebook(i)[idx], self.seq_len)
        h = h + self.phi[i](h.transpose(1, 2)).transpose(1, 2)
        return z_hat + h

    # -- full pyramid ---------------------------------------------------------------
    def forward(self, z):
        """Returns dict(hats, idxs, loss, perplexity). `hats[i]` is z_hat after level i
        (straight-through, so gradients reach the encoder and phi)."""
        z_hat = torch.zeros_like(z)
        hats, idxs, loss, ppl = [], [], z.new_zeros(()), []
        for i, n in enumerate(self.scales):
            rd = down(z - z_hat.detach(), n)                       # (B, n, D)
            flat = rd.reshape(-1, self.dim)
            idx = self._lookup(i, flat.detach())
            if self.training:
                self._ema_update(i, flat.detach(), idx)
                idx = self._lookup(i, flat.detach())
            zq = self.codebook(i)[idx.view(rd.shape[0], n)]
            loss = loss + self.beta * F.mse_loss(rd, zq.detach())
            zq_st = rd + (zq - rd).detach()
            h = up(zq_st, self.seq_len)
            h = h + self.phi[i](h.transpose(1, 2)).transpose(1, 2)
            z_hat = z_hat + h
            hats.append(z_hat)
            idxs.append(idx.view(rd.shape[0], n))
            p = torch.bincount(idx, minlength=self.sizes[i]).float()
            p = p / p.sum()
            ppl.append(torch.exp(-(p * torch.log(p + 1e-10)).sum()))
        return dict(hats=hats, idxs=idxs, loss=loss, perplexity=torch.stack(ppl))


def _tf(d, layers, heads, dropout=0.0):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class TextVQVAE(nn.Module):
    def __init__(self, vocab, seq_len=32, latent_dim=32, d_model=128, layers=3, heads=4,
                 scales=(1, 2, 4, 8, 16, 32), codebook_size=64):
        super().__init__()
        self.cfg = dict(vocab=vocab, seq_len=seq_len, latent_dim=latent_dim, d_model=d_model,
                        layers=layers, heads=heads, scales=list(scales), codebook_size=codebook_size)
        self.seq_len, self.vocab = seq_len, vocab
        self.tok = nn.Embedding(vocab, d_model)
        self.pos_e = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.enc = _tf(d_model, layers, heads)
        self.to_lat = nn.Linear(d_model, latent_dim)
        # no learnable scale => quantization error can't be gamed by shrinking z
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        sizes = [codebook_size] * len(scales) if isinstance(codebook_size, int) else list(codebook_size)
        self.vq = MultiScaleResidualVQ(latent_dim, seq_len, scales, sizes)
        self.from_lat = nn.Linear(latent_dim, d_model)
        self.pos_d = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.dec = _tf(d_model, layers, heads)
        self.out = nn.Linear(d_model, vocab)

    def encode_z(self, x):
        h = self.tok(x) + self.pos_e
        return self.norm(self.to_lat(self.enc(h)))

    def decode_z(self, z_hat):
        return self.out(self.dec(self.from_lat(z_hat) + self.pos_d))

    def forward(self, x, n_partial=2, partial_weight=1.0):
        """Recon loss on the full pyramid plus on `n_partial` random truncations, so each
        coarse prefix of the hierarchy is individually decodable.  Without this the finest
        level alone can reconstruct characters and the coarse levels collapse to one code."""
        out = self.vq(self.encode_z(x))
        logits = self.decode_z(out["hats"][-1])
        rec = F.cross_entropy(logits.reshape(-1, self.vocab), x.reshape(-1))
        loss = rec + out["loss"]
        if self.training and n_partial:
            cuts = torch.randperm(self.vq.n_levels - 1)[:n_partial] + 1  # keep levels [0, c)
            for c in cuts.tolist():
                pl = self.decode_z(out["hats"][c - 1])
                loss = loss + partial_weight / len(cuts) * F.cross_entropy(
                    pl.reshape(-1, self.vocab), x.reshape(-1))
        acc = (logits.argmax(-1) == x).float().mean()
        return dict(loss=loss, rec=rec, vq=out["loss"], acc=acc, perplexity=out["perplexity"])

    @torch.no_grad()
    def encode(self, x):
        """-> (idxs, hats): per-level code indices (B, n_i) and cumulative z_hat after each level."""
        was = self.training
        self.eval()
        out = self.vq(self.encode_z(x))
        self.train(was)
        return out["idxs"], out["hats"]

    @torch.no_grad()
    def reconstruct(self, x, n_levels=None):
        _, hats = self.encode(x)
        return self.decode_z(hats[(n_levels or len(hats)) - 1]).argmax(-1)
