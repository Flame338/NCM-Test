import torch

from ncm_toy import data
from ncm_toy.baseline import ARBaseline
from ncm_toy.generator import NextConceptGenerator
from ncm_toy.sampling import sample_logits
from ncm_toy.vq import TextVQVAE, down, up

SMALL = dict(latent_dim=16, d_model=32, layers=1, heads=2, codebook_size=16)


def tiny_vq():
    return TextVQVAE(data.VOCAB, data.SEQ_LEN, **SMALL)


def test_grammar_roundtrip():
    s = data.all_sentences()
    assert len(s) > 5000 and all(len(x) <= data.SEQ_LEN for x in s)
    assert all(data.parse(x) for x in s)
    assert data.decode(data.encode(s[0], data.SEQ_LEN)) == s[0]
    assert data.parse("the cat saw a dog") is None


def test_pool_upsample_shapes():
    x = torch.randn(2, 32, 8)
    assert [down(x, n).shape[1] for n in (1, 4, 32)] == [1, 4, 32]
    assert up(down(x, 4), 32).shape == x.shape


def test_vq_pyramid_and_partial_recon():
    m = tiny_vq()
    x, _ = data.load_splits("lm")[0].batch(8)
    out = m(x)
    out["loss"].backward()
    assert out["perplexity"].shape == (6,)
    idxs, hats = m.encode(x)
    assert [i.shape[1] for i in idxs] == [1, 2, 4, 8, 16, 32] and len(hats) == 6
    assert all(m.reconstruct(x, k).shape == x.shape for k in range(1, 7))


def test_generator_mask_is_block_causal():
    g = NextConceptGenerator(tiny_vq(), d_model=32, layers=1, heads=2, prefix_len=4, vocab=data.VOCAB)
    P, mask, lvl = 4, g.mask, g.lvl_of
    assert mask[:P, P:].all() and not mask[:P, :P].any()          # prefix sees only prefix
    assert not mask[P:, :P].any()                                 # targets see the prefix
    # a level sees itself and coarser levels but never finer ones
    assert (mask[P:, P:] == (lvl[None, :] > lvl[:, None])).all()


def test_generator_train_and_generate_use_one_pass_per_level():
    vq = tiny_vq()
    g = NextConceptGenerator(vq, d_model=32, layers=1, heads=2, prefix_len=data.PREFIX_LEN, vocab=data.VOCAB)
    tr = data.load_splits("cond")[0]
    x, p = tr.batch(8)
    out = g(x, p)
    out["loss"].backward()
    assert all(q.grad is None for q in vq.parameters())           # tokenizer stays frozen
    calls = []
    g.tf.register_forward_hook(lambda *a: calls.append(1))
    res = g.generate(prefix=p, cfg=1.5)                           # CFG batches cond+uncond into one pass
    assert res["tokens"].shape == (8, data.SEQ_LEN) and res["nfe"] == 6 and len(calls) == 6


def test_teacher_forcing_matches_sampling_inputs():
    """Level-k logits must not depend on tokens of finer levels (causality sanity check)."""
    g = NextConceptGenerator(tiny_vq(), d_model=32, layers=1, heads=2).eval()
    x = data.load_splits("lm")[0].batch(4)[0]
    _, hats = g.vq_model.encode(x)
    with torch.no_grad():
        full = g._level_logits(g._run(g._target_inputs(hats, 4, 5), None), 2)
        part = g._level_logits(g._run(g._target_inputs(hats, 4, 2), None), 2)
    assert torch.allclose(full, part, atol=1e-5)


def test_ar_baseline_nfe_and_sampling():
    m = ARBaseline(data.VOCAB, data.SEQ_LEN, 0, d_model=32, layers=1, heads=2)
    assert m.generate(n=3)["nfe"] == data.SEQ_LEN
    assert sample_logits(torch.tensor([[0.0, 5.0, 1.0]]), temperature=0).item() == 1
    assert sample_logits(torch.tensor([[0.0, 5.0, 1.0]]), top_k=1).item() == 1
