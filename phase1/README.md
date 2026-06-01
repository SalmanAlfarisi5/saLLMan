# Phase 1 — Decoder-Only GPT

Strips the encoder and cross-attention out of the [Phase 0](../phase0/) Transformer
to produce a **GPT-style language model**: one stack of (masked self-attention +
FFN) blocks, single vocabulary, next-token-prediction objective.

References:
- Radford et al. 2018, "Improving Language Understanding by Generative Pre-Training" (GPT-1)
- Radford et al. 2019, GPT-2 — byte-level BPE
- Brown et al. 2020, "Language Models are Few-Shot Learners" ([GPT-3](https://arxiv.org/abs/2005.14165))
- Merity et al. 2016, [WikiText-2 dataset](https://arxiv.org/abs/1609.07843)
- Sennrich et al. 2016, [BPE](https://arxiv.org/abs/1508.07909)

This is the **sanity-check** phase: prove the architectural surgery works
(loss falls, samples get more coherent) before swapping primitives in Phase 2.

---

## Files

| File | Role |
|------|------|
| [decoder_only.py](decoder_only.py) | GPTConfig, GPTBlock, GPT (model). Reuses Phase 0 primitives. |
| [train_lm.py](train_lm.py) | BPE tokenizer, block-packed dataset, WikiText-2 training loop. |
| [decoder_only.ipynb](decoder_only.ipynb) / [train_lm.ipynb](train_lm.ipynb) | Notebook companions. |
| `tok_cache/wikitext2_bpe.json` | Cached 8k-vocab BPE tokenizer (rebuilt if missing). |
| `checkpoints_lm/best_model.pt` | Saved by `train_lm.py` on best val loss. |

---

## The single architectural change

```
   Phase 0 DecoderLayer              Phase 1 GPTBlock
   ─────────────────────             ──────────────────
   1. masked self-attention          1. masked self-attention
   2. CROSS-ATTENTION (over memory)  — removed —
   3. FFN                            2. FFN
```

That's the entire conceptual change. Everything else (mask logic, embedding
scaling, weight tying, init) is identical to Phase 0. The point of the
phase is to verify that removing cross-attention doesn't break anything
before we start changing the *remaining* sublayers in Phase 2.

---

## [decoder_only.py](decoder_only.py) — section-by-section

### §1 `GPTConfig` (lines 56–68)
Two changes vs `TransformerConfig`:
- One `vocab_size` instead of `(src_vocab_size, tgt_vocab_size)`. A
  decoder-only LM consumes and produces tokens from the **same** vocabulary,
  so input and output embeddings can be tied directly.
- `max_len` is for the single sequence (no src/tgt distinction).

Otherwise identical: `d_model=512, n_heads=8, n_layers=6, d_ff=2048`.

### §2 `GPTBlock` (lines 80–103)
Two sublayers, both Post-LN residual (same convention as Phase 0):
```python
x = LayerNorm(x + Dropout(sublayer(x)))
```

The self-attention call is `self_attn(x, x, x, mask=…)` — `Q=K=V=x`.
There is no separate memory tensor. Pad-and-causal mask is the only mask
this block sees.

### §3 `GPT` (lines 109–207)

**Init**:
- One `nn.Embedding` (size = `vocab_size · d_model`).
- One `PositionalEncoding` (sinusoidal — replaced by RoPE in Phase 2c).
- `n_layers` GPTBlocks.
- `lm_head = nn.Linear(d_model, vocab_size, bias=False)`.
- **Weight tying**: `self.lm_head.weight = self.embed.weight`. The same
  matrix is used to (a) embed input tokens and (b) project hidden states
  to vocab logits. This is the standard GPT-2/LLaMA pattern; the only cost
  is that the matrix has to serve both jobs well, which Xavier/0.02 init
  handles fine.

**`make_attn_mask(input_ids)`**: builds one `(B, 1, T, T)` mask that combines
non-pad (`make_pad_mask`) with causal (`make_causal_mask`) via `&`. Broadcasts
across heads. Compare to Phase 0 which returned three masks (`src_mask`,
`tgt_mask`, `memory_mask`) — with no encoder, only one survives.

**`forward(input_ids)`** (3 lines that do the work):
1. `x = embed(input_ids) * √d_model` — same scaling trick as Phase 0,
   needed because sinusoidal PE is about to be added.
2. `x = pos_enc(x)` — adds sinusoidal positional embedding.
3. Loop over blocks, then `lm_head(x)` → `(B, T, vocab_size)` logits.

**`generate(input_ids, max_new_tokens, eos_id)`**: naïve greedy. For each
new token: truncate context to `max_len`, run a full forward pass, argmax
the last position. **O(T²)** in time because every step re-processes the
entire prefix. Phase 2 adds a KV cache that turns this into O(T).

---

## [train_lm.py](train_lm.py) — section-by-section

### §1 Special token ids (lines 67–70)
`PAD=0, BOS=1, EOS=2, UNK=3`. Same convention as Phase 0; the BPE trainer
below produces them in this order.

### §2 BPE tokenizer (lines 73–105)
The first new concept in this phase.

**Why BPE instead of regex word-splitting?**
A word vocab works for Multi30k (5k unique German words). For raw text it
has a **long tail** — rare words, numbers, hyphenated terms, names — that
either explodes the vocab size or floods inputs with `<unk>`. **Subword**
tokenization solves this: rare words get split into known pieces.

Example: `"unbelievably"` → `["un", "believ", "ably"]`.

**Byte-level BPE** (Radford et al. 2019, used by GPT-2/3/LLaMA): the base
alphabet is the 256 possible bytes, not Unicode characters. This means:
- No `<unk>` ever fires — every input can be decomposed into bytes.
- Vocabulary contains arbitrary byte sequences, not just nicely-printable
  strings. This is essential for code (Phase 3) where indentation and
  unusual characters matter.

`add_prefix_space=False` is the right setting for plain text where words
are space-separated. Phase 3 keeps it `False` for code (whitespace is
structurally meaningful — Python indentation).

`special_tokens=[<pad>, <bos>, <eos>, <unk>]` in this exact order — the
BpeTrainer assigns ids 0..3 in registration order. Downstream code
depends on this.

### §3 `BlockDataset` (lines 123–142)
The second new concept in this phase, and arguably more important than BPE.

**Block packing**: concatenate the entire corpus into one long stream of
ids, then chop into fixed-length blocks of `block_size + 1`. Each block
becomes a training example:
- input  = `block[:-1]`   shape `(block_size,)`
- target = `block[1:]`    shape `(block_size,)` (shifted by one)

Why this is the standard LM recipe (nanoGPT, GPT-2, LLaMA, every modern
codebase):
- **Zero padding**: every block is full. ~100% of tokens contribute to
  loss. Compare to sentence-pair batching where ~30%+ of tokens are pad.
- **Cheap shuffling**: just shuffle block indices.
- **Implicit "<eos>" handling**: inserting `<eos>` between documents in
  the stream means the model gets trained on the natural "after the
  document ends, a new topic starts" signal.

Trade-off: blocks can span document boundaries. A 256-token block may
contain the end of one Wikipedia article and the start of another. This
is fine and standard — the model learns to look at `<eos>` to reset
context.

Phase 3 uses the same recipe but with `np.memmap` so the token stream
never has to fit in RAM.

### §4 `collate_blocks` (lines 145–148)
All blocks are the same length — just `torch.stack`. No padding needed.
Compare to Phase 0's `collate_fn`, which had to dynamic-pad.

### §5 `load_wikitext2` (lines 154–207)

End-to-end data pipeline:
1. Load WikiText-2 raw via HF datasets.
2. Train BPE tokenizer on **train split only** (val is held out as
   intended). Cache to `tok_cache/wikitext2_bpe.json` — building takes
   ~30s so we don't want to redo it every run.
3. Encode the entire corpus into a flat `torch.LongTensor`, inserting
   `<eos>` between documents.
4. Wrap in `BlockDataset(block_size=256)`.

### §6 `train_epoch` / `evaluate` (lines 218–292)
Mostly identical to Phase 0's training loop, with three small changes:

1. **Single batch tensor**: `(x, y)` instead of `(src, tgt)`. No more
   `tgt[:, :-1] / tgt[:, 1:]` split — `BlockDataset.__getitem__` already
   produces input/target pair.
2. **Mixed precision** via `torch.amp.autocast(dtype=torch.bfloat16)`.
   bf16 on Ampere (RTX 30-series and up) is essentially free: same
   dynamic range as fp32 (so no GradScaler needed, unlike fp16), half the
   memory bandwidth, ~2× the throughput on matmul-heavy code.
3. Token counting against `PAD_IDX` is symmetric with Phase 0, but with
   block-packed data the count equals `B·T` (no pad tokens exist). Kept
   for safety and shared code paths with Phase 0.

### §7 `generate_sample` (lines 298–306)
Wraps `model.generate(...)` for sample logging each epoch. Encodes a
prompt, generates 60 tokens, decodes back to text.

### §8 `main`
Trains the model on WikiText-2 for 5 epochs.
- `d_model=256, n_layers=6, n_heads=8` (~6.8M params): same dims as
  Phase 0's translation experiment for direct comparison with Phase 2,
  which uses identical dims.
- `BLOCK_SIZE=256, BATCH_SIZE=32` → 8192 tokens per step.
- **Noam scheduler** with warmup=2000 — reused from Phase 0 because
  Post-LN still needs warmup. Phase 2 (Pre-LN) drops Noam for cosine.

Per-epoch: train pass → val pass → log loss and PPL → save on best val →
greedy-sample a fixed prompt to eyeball quality.

---

## Result you should get

WikiText-2, 5 epochs, ~6.8M params (256/6/8):

| Metric | Value |
|--------|-------|
| Val loss | ~4.27 |
| Val perplexity | ~71 |

Phase 2 hits ~64 PPL with **the same parameter count and training budget** —
the entire 10% improvement comes from architectural changes (Pre-LN,
RMSNorm, RoPE, SwiGLU). That clean A/B comparison is the whole point of
sizing Phase 1 and Phase 2 identically.

---

## What changes next (Phase 2 preview)

The block architecture stays the same shape (self-attn → FFN with
residuals), but every primitive inside changes:

- LayerNorm → **RMSNorm** (drop centering, drop bias).
- Post-LN → **Pre-LN** (norm *before* sublayer, not after).
- Sinusoidal PE → **RoPE** (rotate Q/K inside attention).
- ReLU FFN → **SwiGLU** FFN (gated, with `d_ff = 8/3 · d_model`).
- Manual scaled-dot-product → `F.scaled_dot_product_attention` (FlashAttn).
- Naïve generation → **KV-cache** generation.
- Adam + Noam → AdamW + cosine warmup with selective weight decay.

All of these are independent — they could be applied one-at-a-time if you
wanted to ablate each. Phase 2 applies them together (the LLaMA recipe).
