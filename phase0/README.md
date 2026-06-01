# Phase 0 — Vanilla Encoder-Decoder Transformer

Reference implementation of **Vaswani et al. 2017, "Attention Is All You Need"**
([arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)).
Built from scratch in PyTorch, trained on Multi30k German→English translation.

This phase is the foundation. Every later phase reuses primitives defined
here (multi-head attention, FFN, masks, label-smoothing loss, Noam schedule).
Read this file with the paper open on the side — section numbers in the
comments point straight to it.

---

## Files

| File | Role |
|------|------|
| [transformer.py](transformer.py) | The model: embeddings, attention, encoder, decoder, full Transformer class. |
| [train.py](train.py) | DE→EN training loop: tokenizer, vocab, dataset, Noam scheduler, label smoothing, greedy decoder. |
| [transformer.ipynb](transformer.ipynb) / [train.ipynb](train.ipynb) | Notebook companions (same code, runnable cell-by-cell). |
| `checkpoints/best_model.pt` | Saved by `train.py` on best validation loss. |

---

## The architecture in one picture

```
                  ┌─────────────┐                ┌──────────────────┐
   src tokens ──► │  Encoder    │  memory ────►  │  Decoder          │ ──► logits
                  │  (×N=6)     │                │  (×N=6)           │
                  └─────────────┘                └──────────────────┘
                                                          ▲
                                                          │ tgt tokens (right-shifted)

  Encoder block:                                 Decoder block:
    1. Masked-self-attention                       1. Masked-self-attention (causal)
    2. FFN                                         2. CROSS-attention (Q=dec, K/V=enc)
                                                   3. FFN
```

Three things to internalise from this picture before touching Phase 1:

1. The decoder has **three** sublayers (self-attn, cross-attn, FFN).
   Phase 1 strips sublayer 2; that's the entire "decoder-only" transition.
2. **Cross-attention** is the only place encoder and decoder talk.
   Q comes from the decoder hidden states, K and V come from `memory`
   (the encoder output). This is how translation conditions on the source.
3. Layer norms here are **Post-LN** — applied *after* the residual add.
   This is the original 2017 design. It needs warm-up because gradients can
   blow up early in training. Phase 2 flips to Pre-LN to fix that.

---

## [transformer.py](transformer.py) — section-by-section

### §1 `TransformerConfig` (lines 21–35)
Dataclass containing every hyperparameter. Defaults are the paper's "base"
model: `d_model=512, n_heads=8, n_layers=6, d_ff=2048` ([Paper Table 3](https://arxiv.org/abs/1706.03762)).
`__post_init__` asserts `d_model % n_heads == 0` because every head gets a
slice of size `d_k = d_model / h` (Paper §3.2.2). Forget this and the
view/reshape in attention silently produces garbage.

### §2 `PositionalEncoding` (lines 41–69)
Sinusoidal table built **once** at init, registered as a buffer (no params,
but `.to(device)` follows the module).

Formula (Paper §3.5):
```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

The `div_term` is computed in log-space for numerical stability:
`exp(-2i * log(10000) / d_model)` is identical to `1 / 10000^(2i/d_model)`
but avoids overflow at large `d_model`.

Forward: `x + PE[:, :T]` — sinusoids are **added** to the token embedding.
Phase 2c replaces this whole approach with RoPE, which rotates Q/K inside
attention instead of adding to the embedding stream.

### §3 `scaled_dot_product_attention` (lines 75–99)
Paper eq. (1):
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

Shapes: `q,k,v: (B, h, T, d_k)` → `scores: (B, h, T_q, T_k)` →
`out: (B, h, T_q, d_v)`. The `√d_k` divisor (Paper §3.2.1, footnote 4)
prevents `QKᵀ` from growing in magnitude as `d_k` grows, which would push
softmax into saturated regions where gradients vanish.

The mask uses the convention **True = KEEP, False = MASK OUT**. We invert
it with `~mask` and `masked_fill` to `-inf`, which becomes 0 after softmax.

Phase 2 replaces this whole function with one call to
`F.scaled_dot_product_attention`, which dispatches to FlashAttention-2 on
compatible hardware — ~10× less VRAM at the same numerical answer.

### §4 `MultiHeadAttention` (lines 105–153)
Multi-head attention = run `n_heads` parallel attentions, concat, project.
Paper §3.2.2:
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O
head_i           = Attention(Q W^Q_i, K W^K_i, V W^V_i)
```

The trick in implementation: instead of `h` separate `(d_model → d_k)` Linears,
we use ONE `(d_model → d_model)` Linear and `view + transpose` into heads.
Three reshapes:
1. `(B, T, d_model) → (B, T, h, d_k)` via `view`
2. `(B, T, h, d_k) → (B, h, T, d_k)` via `transpose(1, 2)` so the head axis
   broadcasts with the batch axis during the matmul.
3. After attention: reverse with `transpose(1, 2).contiguous().view(B, T, d_model)`.

`.contiguous()` is required: `transpose` returns a non-contiguous view,
and `view` (unlike `reshape`) refuses non-contiguous tensors.

### §5 `PositionwiseFeedForward` (lines 159–172)
```
FFN(x) = ReLU(xW_1 + b_1) W_2 + b_2
```
Two Linears (`d_model → d_ff → d_model`) with ReLU between. Applied
**position-wise** — same FFN at every token, no information mixing across
positions (mixing is attention's job).

Phase 2d replaces ReLU+`d_ff=4·d_model` with SwiGLU+`d_ff=8/3·d_model`.

### §6 `EncoderLayer` (lines 178–201)
Two sublayers (self-attn, FFN), each wrapped in Post-LN residual:
```
x = LayerNorm(x + Dropout(sublayer(x)))
```

### §7 `DecoderLayer` (lines 207–246)
Three sublayers:
1. Masked self-attention on target prefix (causal mask).
2. **Cross-attention**: `Q` from decoder, `K`/`V` from encoder `memory`.
   This is the architectural feature that Phase 1 deletes.
3. Position-wise FFN.

### §8 `Encoder` / `Decoder` stacks (lines 252–294)
Embedding + positional encoding + a list of N=6 layers. The embedding is
scaled by `√d_model` (Paper §3.4) — this counteracts the variance
contribution of the PE that's about to be added. Phase 2 drops this scale
because RoPE doesn't perturb the embedding stream.

### §9 Mask helpers (lines 300–313)
- `make_pad_mask(seq, pad_idx)` → `(B, 1, 1, T)` boolean: True for non-pad.
  Broadcasts across heads and queries.
- `make_causal_mask(size)` → `(1, 1, T, T)` lower-triangular boolean:
  position i can attend to j ≤ i.

Combine them with `&` to get the decoder's self-attention mask.

### §10 `Transformer` (lines 319–356)
Brings it all together. Two notable details:

- **Weight tying** (Paper §3.4): `self.generator.weight = self.decoder.embed.weight`.
  Same matrix used to embed input and project to output logits. Saves
  ~`vocab_size · d_model` params (a lot for big vocabs), tiny quality
  improvement.
- **Xavier init** for any param with `dim > 1`. Phase 2 switches to GPT-2
  init (N(0, 0.02) with residual-projection scaling).

`forward(src, tgt)`: builds three masks, runs encoder, runs decoder with
cross-attention to encoder memory, generator projects to logits.

---

## [train.py](train.py) — section-by-section

### §1–§3 Vocab pipeline (lines 31–96)
- Regex tokenizer (`\w+|[^\w\s]`) — splits on punctuation, lowercases.
  Works for German and English; not for code (Phase 3 switches to byte-BPE).
- `Vocab` with `min_freq=2`: tokens seen fewer than twice → `<unk>`.
- Special tokens go in **first** so their ids are deterministic 0–3.
  Every downstream component (loss, masks, generation) hardcodes these ids.

### §4–§5 `TranslationDataset` + `collate_fn` (lines 102–167)
- Each example is `(src_ids, tgt_ids)`, both wrapped as `[BOS, …, EOS]`.
- `collate_fn` pads to the longest in the **batch** (not in the dataset).
  This is dynamic padding — wastes less compute than padding to a global max.

### §6 `NoamScheduler` (lines 173–204)
Paper §5.3, eq. (3):
```
lr = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))
```

Two regimes:
- **Warmup** (step < warmup_steps): `lr` grows linearly. Important because
  Post-LN gradients are unstable at high LR early in training.
- **Decay** (after warmup): `lr ∝ 1/√step`. Slow decay; over a long run
  the effective LR keeps shrinking.

`step()` is called **per batch**, not per epoch. The scheduler tracks its
own step counter; `.current_lr` is exposed for logging.

Phase 2 swaps this for cosine warmup with AdamW.

### §7 `LabelSmoothingLoss` (lines 210–256)
Paper §5.4: ε=0.1. Instead of `[0,…,1,…,0]` targets, use
`[ε/(V−2), …, 1−ε, …, ε/(V−2)]` (excluding pad and the correct token from
the uniform spread).

Implemented via `KLDivLoss` because:
```
KL(p ‖ q) = Σ p log(p/q) = Σ p log p − Σ p log q
```
The first term is constant w.r.t. model params, so minimising KL ≡
cross-entropy with soft targets, and `KLDivLoss(reduction="sum")` is the
canonical way to express it.

Why it helps: hard 0/1 targets push the model to be overconfident. Soft
targets keep some probability on plausible alternatives and improve BLEU.
Phase 3 drops this for plain cross-entropy because at LM-pretraining scale
it slightly hurts perplexity (LLaMA, GPT-3 also use plain CE).

### §8 `load_multi30k` (lines 262–299)
Downloads `bentrevett/multi30k` via HuggingFace `datasets`. Builds vocabs
from **train split only** — never peek at val/test when building the
tokenizer or vocabulary.

### §9 `train_epoch` / `evaluate` (lines 305–390)
Teacher-forcing recipe:
- `tgt_in  = tgt[:, :-1]` — fed to the decoder.
- `tgt_out = tgt[:, 1:]`  — what we expect it to predict.
- Loss is per-token, normalised by `(tgt_out != PAD_IDX).sum()`.

This shift-by-1 pattern is universal across all later phases.

### §10 `greedy_decode` (lines 396–435)
Step-by-step inference: encode source once (memory is fixed), then loop:
1. Run decoder on the prefix so far (starting with `[BOS]`).
2. Argmax the last position's logits.
3. Append; stop on `EOS` or `max_len`.

**O(T²)** — every step re-runs decoder over the full prefix. Phase 2 fixes
this with a KV cache.

### §11 `main` (lines 441–555)
Trains with `d_model=256, n_layers=3, d_ff=512` (smaller than the paper's
512/6/2048 — Multi30k is tiny, a paper-sized model would overfit). Saves
on best val loss. Visual sanity-check sample translation each epoch.

---

## Conceptual gotchas to keep in mind

- **Padding ≠ masking**: pad tokens are inserted to make the batch
  rectangular; the **mask** is what tells attention to ignore them. If you
  forget the pad mask, attention will average across pad positions and
  contaminate every other token.
- **Mask convention**: this code uses `True = KEEP`. Some libraries use the
  opposite (`True = MASK OUT`). Phase 2 hands the causal mask off to SDPA
  via `is_causal=True` and never materialises a mask tensor — be aware of
  what convention each consumer expects.
- **Embedding scaling**: `embed(x) * √d_model` only makes sense when you're
  about to **add** a positional embedding with its own scale. Drop it the
  moment you switch to RoPE.
- **Teacher forcing**: training feeds the ground-truth prefix; inference
  feeds the model's own predictions. The mismatch is "exposure bias",
  worsened by greedy decoding. Beam search and sampling help; we keep
  greedy here because the goal is sanity checking, not BLEU SOTA.

---

## What changes next (Phase 1 preview)

- Encoder + cross-attention sublayer disappear.
- Decoder layer goes from **3** sublayers to **2** (self-attn + FFN).
- Single vocab, single sequence, single mask.
- Same primitives reused — `MultiHeadAttention`, `PositionwiseFeedForward`,
  `PositionalEncoding`, mask helpers, `LabelSmoothingLoss`, `NoamScheduler`
  all import directly from this phase.
