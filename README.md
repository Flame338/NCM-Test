# ncm-toy — a tiny Next-Concept Modeling playground

A small, CPU-friendly re-implementation of the core idea of **Next-Concept Modeling (NCM)**
([arXiv 2610.14162](https://arxiv.org/abs/2610.14162), Kumbong et al., Hazy Research / Together AI;
[reference code](https://github.com/HazyResearch/ncm)) for experimenting with the mechanism —
not for reproducing the paper's numbers.

> **Provenance.** The first version was built from the public reference repo and secondary summaries because
> `arxiv.org` was unreachable. I later read the paper PDF and compared it with this code (Sec. 3, App. C and F).
> The section "Closing the accuracy gap" below records what that comparison changed and what did not work.
> This is still a small re-implementation, not a reproduction: details marked *(toy)* are my simplifications.

## The idea in one picture

Autoregressive LMs emit one token per forward pass. Diffusion LMs emit many tokens per pass but need many passes.
NCM instead **learns a coarse-to-fine hierarchy of discrete "concepts"** and generates it level by level:

```
text ──encoder──▶ z ──▶ level 0: 1 code   (coarse gist)         ┐ multiscale residual
                         level 1: 2 codes  (what level 0 missed)  │ VQ (stage 1, frozen
                         level 2: 4 codes                         │ after training)
                         ...                                      │
                         level 5: 32 codes (character detail)     ┘
                      └──▶ decoder ──▶ text

stage 2: a transformer predicts level k's codes *all at once* given levels < k  →  6 forward passes
         per sentence here, versus 32 for a character-level AR model.
```

| Component | Here | Paper / reference code |
|---|---|---|
| Tokenizer | transformer enc/dec, **multiscale residual VQ**, scales 1·2·4·8·16·32, 64 codes/level, EMA codebooks + dead-code restart, conv `phi` per level, pooled residual / interpolated upsample | same family; far larger, 16-level config appears in the repo's example |
| Generator input at level k | running reconstruction `z_hat` pooled to `n_k` positions (level 0: learned start token) | `get_next_autoregressive_input`, VAR-style |
| Attention | the shipped checkpoints are block-causal; `--attn block_diagonal` (new default for training, and what the paper uses, App. F.2.1) lets each level see only itself plus the prefix, and sampling then runs one level block per step | paper: block-diagonal; repo also has block-causal |
| Heads | one per level over that level's codebook | same |
| Guidance | classifier-free guidance, strength ramped with depth `t_k = cfg·k/(K-1)`, `(1+t)·cond − t·uncond` | same formula as the reference sampler |
| Training objective | per-level cross-entropy on codes, teacher-forced *(toy: loss code for the real model isn't in the files I read)* | — |
| Tokenizer extra | extra decode-from-first-k-levels loss so coarse prefixes are meaningful *(toy; resembles the repo's "fine drop")* | "fine drop" option |
| Data | synthetic grammar, 32 chars | LM1B, OpenWebText, XSum, WMT14, Sudoku, … |

## Data: a grammar you can measure against

`the small cat chased a dog.` — 9,502 sentences from a finite grammar (85/15 train/held-out split).
Because the grammar is known, quality is exact rather than a proxy:

* **valid** – parses under the grammar
* **novel** – valid *and* not in the training split (generalisation, not memorisation)
* **unique** – distinct samples
* **faithful** – (conditional task) the sentence's subject/verb/object equal the keyword prefix, e.g. `fox liked bird`

Two tasks: `lm` (unconditional) and `cond` (keywords → sentence, with CFG).

## Quick start

```bash
pip install torch numpy pytest
python -m pytest -q                                   # 11 fast tests

# pretrained toy checkpoints are in checkpoints/ — play right away:
python -m ncm_toy recon                               # how much do the first k levels recover?
python -m ncm_toy sample -n 8 --trace                 # watch a sentence sharpen level by level
python -m ncm_toy sample --gen checkpoints/ncm_cond.pt --prefix "fox liked bird" -n 6 --cfg 2
python -m ncm_toy eval --gen checkpoints/ncm_lm.pt --ar checkpoints/ar_lm.pt
python -m ncm_toy eval --gen checkpoints/ncm_lm.pt --verify 1 --candidates 16   # cycle-consistency rejection
python -m ncm_toy eval --gen checkpoints/ncm_cond.pt --ar checkpoints/ar_cond.pt --cfg 2

# retrain everything (~1.5 h on 4 CPU cores with the jobs run in parallel; see `python -m ncm_toy <cmd> --help`)
python -m ncm_toy train-all
```

## What the checkpoints in `checkpoints/` actually do

Single run, seed 0, 512 samples per row, temperature 1.0 unless noted. Tokenizer: 1,500 steps; generators and
baselines: 3,000 steps each, ~1.8M trainable parameters each.

**Tokenizer** (`python -m ncm_toy recon`, 500 held-out sentences). Reconstruction from only the first *k* levels:

| levels used | codes | char accuracy | exact match |
|---|---|---|---|
| 1 | 1 | 0.664 | 0.000 |
| 2 | 3 | 0.964 | 0.656 |
| 3 | 7 | 0.992 | 0.910 |
| 6 | 63 | 1.000 | 0.992 |

One code gives the rough shape of the sentence (`the small dox saw a o d f.`); the first two levels give
almost everything. Most information sits in the coarse levels, which is the hierarchy doing its job, though
it also means the last levels add little on this easy data.

**Generation** (`python -m ncm_toy eval ...`):

| model | task | forward passes | valid | novel* | unique | faithful | sec |
|---|---|---|---|---|---|---|---|
| NCM | lm | **6** | 0.850 | 0.082 | 0.969 | – | 1.9 |
| AR | lm | 32 | **0.996** | 0.162 | 0.977 | – | 4.3 |
| NCM | cond, cfg 0 | **6** | 0.861 | 0.090 | 0.979 | 0.857 | 2.9 |
| NCM | cond, cfg 2 | 6 | 0.846 | 0.084 | 0.977 | 0.844 | 5.3 |
| NCM | cond, cfg 4 | 6 | 0.783 | 0.080 | 0.973 | 0.783 | 5.0 |
| NCM | cond, T=0.6, cfg 0 | 6 | 0.906 | 0.084 | 0.953 | 0.904 | 2.6 |
| AR | cond | 32 | **1.000** | 0.146 | 0.973 | **0.998** | 9.0 |

\* "novel" counts valid, unseen sentences over *all* samples. The held-out split is 15% of the grammar, so
a model that samples the grammar uniformly would score about 0.15; the AR baseline is near that, NCM is lower.

How to read this:

* **The mechanism works but this toy NCM is less accurate than the baseline.** It makes 6 forward passes
  instead of 32 (and ran about 2x faster even though the AR baseline has no KV cache), but 85-90% of its
  samples parse versus ~100% for AR. The common failures are fine-level mismatches such as `a fox liked a bibird`.
  This is a 3,000-step, one-seed run on a tiny model; I did not tune it, so I can't say how much of the gap
  is the method and how much is under-training.
* **Level 0 and 1 are the hard part.** During training, level-0/1 code accuracy stays at ~0.05-0.4 while levels 2-5
  reach 0.85-1.0: choosing the gist is a high-entropy decision, so those codes are close to a sample from a
  distribution, not a deterministic answer.
* **CFG did not help here.** Guidance at 2 and 4 lowered validity and faithfulness. Cause is unknown;
  the guidance ramp or the 10% condition-dropout rate may not suit a toy model, and I have not investigated.
* Lower temperature helped the conditional model (0.86 -> 0.91 valid) and did little for the unconditional one.

## Closing the accuracy gap (NCM vs. the AR baseline)

The shipped NCM is ~83% valid vs ~100% for AR at 6 vs 32 forward passes. I read the paper and tried its
suggestions, plus a diagnosis-driven fix. All numbers: unconditional task, 1,024 samples, single seed, 3,000-step
generators, so differences of ~3 points are noise.

**What did not help (and one that hurt badly)**

| change | valid | note |
|---|---|---|
| shipped model (block-causal, no corruption) | 82.8% | reference |
| block-diagonal attention (paper's choice) | 81.6% | no accuracy change here; sampling is cheaper |
| conditioning corruption, mean rate 0.25 / 0.5 | 33% / 4% | paper's headline remedy; **destroyed generation** in this toy |
| dropout 0.1 + weight decay 0.1 (paper values) | 82.5% | held-out code accuracy barely moved |
| dropout 0.2 + weight decay 0.1 | 80.6% | |
| dropout + corruption 0.1 | 57-61% | even mild corruption hurts |
| per-level temperature (argmax on fine levels) | 80-85% | all within noise |

Why corruption fails here (partly confident, partly a guess): the toy's first two levels carry almost all the
information (~8.8 of ~9.0 nats per sentence), and randomizing even one coarse code wrecks the sentence (5-10% valid)
while the targets stay clean, so the model learns to hedge and ignore its conditioning. The paper's tokenizer has 12
levels with 4,096 codes each, so a swapped code costs far less. The exact "shifted linear" corruption schedule is
unspecified, so mine may also differ from theirs.

**Where the errors come from.** Fixing the first *f* levels to the true codes of a held-out sentence and sampling
the rest (`generate(given=...)`): f=0 81%, f=2 90%, f=3 92%, f=4 95%, f=5 98%, f=6 99.5%. Errors are spread across
levels: the two codes of level 1 are sampled independently, so combinations that no real sentence produces get
through, and those off-support prefixes then reach finer levels that never saw such inputs. Separately, the
generator predicts codes of *held-out* sentences far worse than training ones (0.67-0.73 vs 0.90-0.99 at levels
2-5), and regularization did not close that gap, so the codes behave partly like a hash of the sentence.

**What did help: cycle-consistency rejection (no retraining, no extra generator passes).** Found by an Opus
design review of this code against the paper, then re-measured here. At a chosen level, draw `--candidates`
code sets from the *same* logits, decode the partial reconstruction with the frozen tokenizer, re-encode it, and
keep the first candidate whose codes for levels 0..k survive the round trip. This rejects off-support prefixes.

| model | verify | candidates | valid | novel | unique | faithful |
|---|---|---|---|---|---|---|
| NCM lm | none | – | 82.8% | 8.1% | 95.1% | – |
| NCM lm | level 1 | 4 / 16 / 32 | 91.6% / 92.9% / 92.9% | 10.2% / 7.7% / 7.8% | ~94% | – |
| NCM lm | levels 1-5 | 8 | 93.0% | 8.8% | 92.3% | – |
| NCM cond | none | – | 85.4% | 8.6% | 94.6% | 84.8% |
| NCM cond | level 1 | 16 | 94.7% | 9.0% | 93.8% | 94.4% |

```bash
python -m ncm_toy eval --gen checkpoints/ncm_lm.pt --verify 1 --candidates 16
python -m ncm_toy sample --gen checkpoints/ncm_cond.pt --prefix "fox liked bird" --verify 1 --candidates 16
```

Caveats: the generator still makes 6 forward passes, but each verified level adds tokenizer encode + decode on
`B x candidates` sequences, which the paper's speed numbers do not include. I did not measure that cost cleanly
(the CPU was shared with training), so do not read the NFE column as the whole cost. The remaining gap to AR is
about 5-7 points. The most promising untried idea is a tokenizer redesign so that the high-entropy decisions are
made one code at a time (e.g. scales `1,1,1,2,4,8,16,32`, paper-style nested dropout, smaller fine codebooks); that
needs retraining both stages.

## Things worth poking at

* **Levels as a dial.** `recon` shows quality vs. number of levels. Truncating generation after level *k* gives a
  cheaper, blurrier answer (`sample --trace`).
* **Per-level temperature.** `NextConceptGenerator.generate(level_temps=[0.3, 0.7, 1, 1, 1, 1])` — sample the gist
  conservatively, let details vary (or the reverse).
* **CFG ramp.** `--cfg 0|1|2|4` on the conditional model; watch `faithful` vs. `unique`.
* **Hierarchy shape.** Edit `scales=` / `codebook_size=` in `TextVQVAE` (e.g. `(1,4,16,32)`, or 8 codes at level 0).
  The partial-decode loss (`n_partial` in `TextVQVAE.forward`) is what pushes information into the coarse levels.
  In an early 60-step smoke run with a weaker version of it, the level-0 codebook collapsed to one code
  (perplexity 1.00), because the finest level alone can already spell out characters. I did not run a
  converged ablation, so treat `n_partial=0` as an experiment to try rather than an established result.
* **Swap the data.** `ncm_toy/data.py` is ~80 lines; any fixed-length char corpus works if you keep `SEQ_LEN`
  a power of two (or change `scales`).

```python
from ncm_toy import load_gen, data
m, task = load_gen("checkpoints/ncm_lm.pt")
out = m.generate(n=4, temperature=0.8, top_p=0.95)
print([data.decode(t) for t in out["tokens"]], "forward passes:", out["nfe"])
```

## Layout

```
ncm_toy/data.py       grammar corpus + char tokenizer + exact metrics' parser
ncm_toy/vq.py         MultiScaleResidualVQ + TextVQVAE        (stage 1)
ncm_toy/generator.py  NextConceptGenerator                    (stage 2)
ncm_toy/baseline.py   small autoregressive baseline
ncm_toy/train.py      training loops / checkpoint IO
ncm_toy/__main__.py   CLI
tests/                shape, causality, NFE and frozen-tokenizer tests
```

## Honest limitations

* A 9.5k-sentence synthetic grammar is far easier than natural text; "valid" rates say the mechanism works,
  not that it scales. The paper's speed/quality claims are the authors'; nothing here tests them.
* The AR baseline recomputes the full prefix every step (no KV cache), so its **wall-clock** is pessimistic.
  Compare **forward-pass counts (NFE)**, which are exact, more than seconds.
* No diffusion-LM baseline is included.
* Tiny models, one seed, short training — expect noisy numbers.
* The shipped checkpoints predate the paper-derived options (block-causal, no corruption, dropout 0); the newer
  variants I trained were not better, so they are not shipped.
