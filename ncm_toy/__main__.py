"""python -m ncm_toy <command> --help"""
from __future__ import annotations

import argparse
import time

import torch

from . import data, train as T
from .metrics import decode_batch, evaluate_texts

CK = "checkpoints"


def cmd_train_vq(a):
    T.train_vq(a.out, steps=a.steps, bs=a.bs, lr=a.lr, seed=a.seed)


def cmd_train_gen(a):
    T.train_gen(a.vq, a.out, task=a.task, steps=a.steps, bs=a.bs, lr=a.lr, seed=a.seed,
                attn=a.attn, corrupt_mean=a.corrupt, alpha=a.alpha, dropout=a.dropout, wd=a.wd)


def cmd_train_ar(a):
    T.train_ar(a.out, task=a.task, steps=a.steps, bs=a.bs, lr=a.lr, seed=a.seed)


def cmd_train_all(a):
    T.train_vq(f"{CK}/vq.pt", steps=a.vq_steps, seed=a.seed)
    for task in ("lm", "cond"):
        T.train_gen(f"{CK}/vq.pt", f"{CK}/ncm_{task}.pt", task=task, steps=a.steps, seed=a.seed)
        T.train_ar(f"{CK}/ar_{task}.pt", task=task, steps=a.steps, seed=a.seed)


def cmd_recon(a):
    """How much of a sentence do the first k levels recover?"""
    vq = T.load_vq(a.vq)
    _, test = data.load_splits("lm")
    x = test.x[:a.n]
    print(f"{'levels':>6} {'codes':>5} {'char acc':>9} {'exact':>7}")
    for k in range(1, vq.vq.n_levels + 1):
        r = vq.reconstruct(x, k)
        print(f"{k:>6} {sum(vq.vq.scales[:k]):>5} {(r == x).float().mean():9.3f} {(r == x).all(1).float().mean():7.3f}")
    print("\ncoarse-to-fine reconstruction of held-out sentences:")
    for i in range(a.show):
        print(f"  target: {data.decode(x[i])}")
        for k in range(1, vq.vq.n_levels + 1):
            print(f"   L{k}: {data.decode(vq.reconstruct(x[i:i+1], k)[0])}")
    idxs, _ = vq.encode(x[:a.show])
    print("\ncodes per level for the first sentence:", [i[0].tolist() for i in idxs])


def cmd_sample(a):
    m, task = T.load_gen(a.gen)
    prefix = None
    if task == "cond":
        keys = a.prefix or ["cat chased dog"]
        prefix = torch.tensor([data.encode(k, data.PREFIX_LEN) for k in keys for _ in range(a.n)])
    torch.manual_seed(a.seed)
    out = m.generate(n=a.n, prefix=prefix, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, cfg=a.cfg,
                     level_temps=_level_temps(a.level_temps))
    toks = out["tokens"]
    print(f"task={task}  levels(NFE)={out['nfe']}  T={a.temperature} top_k={a.top_k} top_p={a.top_p} cfg={a.cfg}")
    for i in range(toks.shape[0]):
        head = f"[{data.decode(prefix[i])}] " if prefix is not None else ""
        print(f"{head}{data.decode(toks[i])}")
        if a.trace:
            for k, t in enumerate(out["trace"][:-1]):
                print(f"      after level {k}: {data.decode(t[i])}")


def _level_temps(spec):
    return [float(t) for t in spec.split(",")] if spec else None


def _gen_texts(model, task, n, bs, prefixes=None, **kw):
    texts, t0 = [], time.time()
    for s in range(0, n, bs):
        pre = None if prefixes is None else torch.tensor(
            [data.encode(p, data.PREFIX_LEN) for p in prefixes[s:s + bs]])
        out = model.generate(n=min(bs, n - s), prefix=pre, **kw)
        texts += decode_batch(out["tokens"])
    return texts, time.time() - t0, out["nfe"]


def cmd_eval(a):
    train, test = data.load_splits("lm")
    runs = []
    if a.gen:
        runs.append(("NCM", *T.load_gen(a.gen)))
    if a.ar:
        runs.append(("AR", *T.load_ar(a.ar)))
    torch.manual_seed(a.seed)
    print(f"{'model':<6}{'task':<6}{'NFE':>4}{'valid':>8}{'novel':>8}{'unique':>8}{'faithful':>10}{'sec':>8}")
    for name, m, task in runs:
        prefixes = None
        if task == "cond":  # keywords drawn from held-out sentences
            prefixes = [data.keyword_prefix(test.sentences[i % len(test)]) for i in range(a.n)]
        kw = {"cfg": a.cfg, "level_temps": _level_temps(a.level_temps)} if name == "NCM" else {}
        texts, secs, nfe = _gen_texts(m, task, a.n, a.bs, prefixes, temperature=a.temperature, **kw)
        r = evaluate_texts(texts, train.sentences, prefixes)
        print(f"{name:<6}{task:<6}{nfe:>4}{r['valid']:8.3f}{r['novel_valid']:8.3f}{r['unique']:8.3f}"
              f"{r.get('faithful', float('nan')):10.3f}{secs:8.2f}")


def build():
    p = argparse.ArgumentParser(prog="ncm_toy", description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        s = sp.add_parser(name, **kw); s.set_defaults(fn=fn); s.add_argument("--seed", type=int, default=0); return s

    s = add("train-vq", cmd_train_vq, help="stage 1: multiscale residual VQ-VAE")
    s.add_argument("--out", default=f"{CK}/vq.pt"); s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--bs", type=int, default=128); s.add_argument("--lr", type=float, default=2e-3)

    s = add("train-gen", cmd_train_gen, help="stage 2: next-concept generator")
    s.add_argument("--task", choices=["lm", "cond"], default="lm"); s.add_argument("--vq", default=f"{CK}/vq.pt")
    s.add_argument("--out", default=None); s.add_argument("--steps", type=int, default=3000)
    s.add_argument("--bs", type=int, default=128); s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--attn", choices=["block_diagonal", "block_causal"], default="block_diagonal")
    s.add_argument("--corrupt", type=float, default=0.0, help="mean conditioning-corruption rate (paper uses ~0.5)")
    s.add_argument("--alpha", type=float, default=1.0, help="loss weight exponent: 0 per-code, 1 per-level")
    s.add_argument("--dropout", type=float, default=0.0)
    s.add_argument("--wd", type=float, default=0.01, help="AdamW weight decay (paper: 0.1)")

    s = add("train-ar", cmd_train_ar, help="autoregressive baseline")
    s.add_argument("--task", choices=["lm", "cond"], default="lm"); s.add_argument("--out", default=None)
    s.add_argument("--steps", type=int, default=3000); s.add_argument("--bs", type=int, default=128)
    s.add_argument("--lr", type=float, default=1e-3)

    s = add("train-all", cmd_train_all, help="reproduce every checkpoint")
    s.add_argument("--vq-steps", type=int, default=1500); s.add_argument("--steps", type=int, default=3000)

    s = add("recon", cmd_recon, help="reconstruct from the first k levels")
    s.add_argument("--vq", default=f"{CK}/vq.pt"); s.add_argument("-n", type=int, default=500)
    s.add_argument("--show", type=int, default=3)

    s = add("sample", cmd_sample, help="generate text")
    s.add_argument("--gen", default=f"{CK}/ncm_lm.pt"); s.add_argument("-n", type=int, default=8)
    s.add_argument("--prefix", nargs="*", help='cond task keywords, e.g. "fox liked bird"')
    s.add_argument("--temperature", type=float, default=1.0); s.add_argument("--top-k", type=int, default=0)
    s.add_argument("--top-p", type=float, default=1.0); s.add_argument("--cfg", type=float, default=0.0)
    s.add_argument("--level-temps", default=None, help='per-level temperature, e.g. "1,1,0,0,0,0" (0 = argmax)')
    s.add_argument("--trace", action="store_true", help="show the text decoded after each level")

    s = add("eval", cmd_eval, help="grammar-based quality + speed, NCM vs AR")
    s.add_argument("--gen"); s.add_argument("--ar"); s.add_argument("-n", type=int, default=512)
    s.add_argument("--bs", type=int, default=256); s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--cfg", type=float, default=0.0)
    s.add_argument("--level-temps", default=None, help='per-level temperature, e.g. "1,1,0,0,0,0" (0 = argmax)')
    return p


def main():
    a = build().parse_args()
    if a.cmd in ("train-gen", "train-ar") and a.out is None:
        a.out = f"{CK}/{'ncm' if a.cmd == 'train-gen' else 'ar'}_{a.task}.pt"
    a.fn(a)


if __name__ == "__main__":
    main()
