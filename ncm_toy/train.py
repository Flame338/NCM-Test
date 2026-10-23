from __future__ import annotations

import math
import os
import time

import torch

from . import data
from .baseline import ARBaseline
from .generator import NextConceptGenerator
from .vq import TextVQVAE


def set_seed(s):
    torch.manual_seed(s)


def n_params(m, trainable_only=True):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)


def _opt(params, lr, steps, warmup=100):
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0)))))
    return opt, sched


def _loop(model, params, step_fn, steps, lr, log_every, tag):
    opt, sched = _opt(params, lr, steps)
    t0 = time.time()
    for s in range(1, steps + 1):
        model.train()
        out = step_fn()
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if s % log_every == 0 or s == steps:
            extra = " ".join(f"{k}={_fmt(v)}" for k, v in out.items() if k != "loss")
            print(f"[{tag}] step {s}/{steps} loss={out['loss'].item():.4f} {extra} ({time.time()-t0:.0f}s)", flush=True)


def _fmt(v):
    if v.numel() == 1:
        return f"{v.item():.3f}"
    return "[" + ",".join(f"{x:.2f}" for x in v.tolist()) + "]"


def save(path, kind, model, **meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(dict(kind=kind, cfg=model.cfg, state=model.state_dict(), meta=meta), path)
    print(f"saved {path}")


def train_vq(out, steps=1500, bs=128, lr=2e-3, seed=0, log_every=250, **model_kw):
    set_seed(seed)
    train, _ = data.load_splits("lm")
    model = TextVQVAE(data.VOCAB, data.SEQ_LEN, **model_kw)
    print(f"VQ-VAE params: {n_params(model):,}  scales={model.vq.scales}")
    gen = torch.Generator().manual_seed(seed)

    def step():
        x, _ = train.batch(bs, gen)
        r = model(x)
        return dict(loss=r["loss"], rec=r["rec"], acc=r["acc"], ppl=r["perplexity"])

    _loop(model, list(model.parameters()), step, steps, lr, log_every, "vq")
    save(out, "vq", model)
    return model


def load_vq(path):
    ck = torch.load(path, map_location="cpu")
    m = TextVQVAE(**ck["cfg"]); m.load_state_dict(ck["state"]); return m.eval()


def train_gen(vq_path, out, task="lm", steps=3000, bs=128, lr=1e-3, seed=0, log_every=250, **model_kw):
    set_seed(seed)
    train, _ = data.load_splits(task)
    vq = load_vq(vq_path)
    P = train.p.shape[1]
    model = NextConceptGenerator(vq, prefix_len=P, vocab=data.VOCAB, **model_kw)
    print(f"NCM generator ({task}) trainable params: {n_params(model):,}  cfg={model.cfg}")
    codes = model.encode_codes(train.x)  # tokenizer is frozen: encode the training set once
    gen = torch.Generator().manual_seed(seed)

    def step():
        i = torch.randint(0, len(train), (bs,), generator=gen)
        r = model(prefix=train.p[i] if P else None, idxs=[c[i] for c in codes])
        return dict(loss=r["loss"], ce=r["per_level"], acc=r["acc"])

    _loop(model, model.trainable_parameters(), step, steps, lr, log_every, f"ncm-{task}")
    save(out, "ncm", model, task=task, vq_cfg=vq.cfg)
    return model


def load_gen(path):
    ck = torch.load(path, map_location="cpu")
    vq = TextVQVAE(**ck["meta"]["vq_cfg"])
    cfg = {"attn": "block_causal", **ck["cfg"]}  # checkpoints from before the attn option were block-causal
    m = NextConceptGenerator(vq, **cfg); m.load_state_dict(ck["state"])
    return m.eval(), ck["meta"]["task"]


def train_ar(out, task="lm", steps=3000, bs=128, lr=1e-3, seed=0, log_every=250, **model_kw):
    set_seed(seed)
    train, _ = data.load_splits(task)
    P = train.p.shape[1]
    model = ARBaseline(data.VOCAB, data.SEQ_LEN, P, **model_kw)
    print(f"AR baseline ({task}) params: {n_params(model):,}")
    gen = torch.Generator().manual_seed(seed)

    def step():
        x, p = train.batch(bs, gen)
        r = model(x, p if P else None)
        return dict(loss=r["loss"], acc=r["acc"])

    _loop(model, list(model.parameters()), step, steps, lr, log_every, f"ar-{task}")
    save(out, "ar", model, task=task)
    return model


def load_ar(path):
    ck = torch.load(path, map_location="cpu")
    m = ARBaseline(**ck["cfg"]); m.load_state_dict(ck["state"])
    return m.eval(), ck["meta"]["task"]
