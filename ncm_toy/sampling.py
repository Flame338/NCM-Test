import torch


def sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """logits: (..., V) -> sampled indices (...). temperature<=0 means argmax."""
    if temperature <= 0:
        return logits.argmax(-1)
    lg = logits / temperature
    if top_k and top_k < lg.shape[-1]:
        kth = lg.topk(top_k, dim=-1).values[..., -1:]
        lg = lg.masked_fill(lg < kth, float("-inf"))
    if top_p < 1.0:
        sl, si = lg.sort(dim=-1, descending=True)
        cum = sl.softmax(-1).cumsum(-1)
        drop = cum - sl.softmax(-1) > top_p
        sl = sl.masked_fill(drop, float("-inf"))
        lg = torch.full_like(lg, float("-inf")).scatter(-1, si, sl)
    probs = lg.softmax(-1)
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])
