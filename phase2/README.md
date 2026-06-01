# Phase 2 — LLaMA-Class Modernised Decoder

Applies every post-2020 architectural improvement that LLaMA, Mistral,
Qwen, GPT-NeoX, and DeepSeek share. Same parameter count and training
budget as [Phase 1](../phase1/) — what changes is *quality per parameter*.

References:
- Xiong et al. 2020, [Pre-LN](https://arxiv.org/abs/2002.04745)
- Zhang & Sennrich 2019, [RMSNorm](https://arxiv.org/abs/1910.07467)
- Su et al. 2021, [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- Shazeer 2020, [GLU Variants — SwiGLU](https://arxiv.org/abs/2002.05202)
- Touvron et al. 2023, [LLaMA](https://arxiv.org/abs/2302.13971)
- Radford et al. 2019, GPT-2 §2.3 (init recipe)
- Dao et al. 2022, [FlashAttention](https://arxiv.org/abs/2205.14135) /
  Dao 2023, [FlashAttention-2](https://arxiv.org/abs/2307.08691)

The headline result: same 6.8M params, same WikiText-2 5-epoch budget,
**9.9 % lower perplexity** than Phase 1 (64.1 vs 71.2). All of it comes
from the architectural changes; data and compute are held constant.

---

## Files

| File | Role |
|------|------|
| [decoder_only_v2.py](decoder_only_v2.py) | Self-contained model: GPTConfigV2, RMSNorm, RoPE helpers, CausalSelfAttention, SwiGLU, TransformerBlock, GPTv2. |
| [train_lm_v2.py](train_lm_v2.py) | AdamW + cosine warmup + selective weight decay trainer. Reuses Phase 1's data pipeline. |

Note: this is the **only phase where the model file is self-contained** —
it doesn't import attention/FFN/norm primitives from Phase 0. Reason: every
primitive changed, and copying them gave room to comment each new design
decision next to the code.

---

## The five changes at a glance

| # | Change | Why | Cost |
|---|--------|-----|------|
| 2a | Post-LN → **Pre-LN** | Stable gradients without warmup tricks | Needs a final norm before lm_head |
| 2b | LayerNorm → **RMSNorm** | Same regularisation, simpler op, faster | None |
| 2c | Sinusoidal PE → **RoPE** | Relative positions, FlashAttn compatible | Precompute cos/sin tables |
| 2d | ReLU FFN → **SwiGLU** | Gated FFN strictly outperforms in ablations | 3 weight matrices instead of 2 → set `d_ff = 8/3·d_model` to keep param count |
| + | Manual SDPA → `F.scaled_dot_product_attention` | FlashAttention kernel: ~10× less VRAM, faster | None — PyTorch picks the implementation |
| + | No bias on Linears | LLaMA convention; small param savings | None |
| + | GPT-2 init | Stable depth scaling | One pass over params at init |
| + | KV-cache generation | O(T) per token instead of O(T²) | Cache memory grows with seq len |

---

## [decoder_only_v2.py](decoder_only_v2.py) — section-by-section

### §1 `GPTConfigV2` (lines 53–75)
Two new fields:
- `rope_base: float = 10000.0` — frequency base for RoPE. Standard value
  in Su et al. 2021 and what LLaMA uses for short contexts (≤4k tokens).
  Larger values support longer contexts.
- `norm_eps: float = 1e-5` — RMSNorm epsilon. LLaMA uses `1e-5` or `1e-6`.

`d_ff` is now `int | None`. If left `None`, `__post_init__` computes
`8/3 · d_model` rounded **up** to the next multiple of 64:
```python
target = int(8 * d_model / 3)
d_ff = ((target + 63) // 64) * 64
```

The 8/3 ratio (Shazeer 2020 §3.3) keeps total FFN params constant when
moving from a 2-matrix ReLU FFN with `d_ff = 4d` to a 3-matrix SwiGLU FFN.
The "multiple of 64" rounding is for GPU kernel efficiency — most CUDA
matmul tiles are 64-aligned.

`dropout=0.0` by default. LLaMA-style: with enough data, dropout actively
hurts. Phase 1 used 0.1 because WikiText-2 is tiny.

### §2 `RMSNorm` (lines 88–100) — Change 2b

```
LayerNorm:  y = γ · (x - mean(x)) / √(var(x) + ε)  + β
RMSNorm:    y = γ ·  x            / √(mean(x²) + ε)
```

Two things removed:
1. **Mean subtraction** (centering).
2. **Learnable bias β**.

Zhang & Sennrich 2019's empirical finding: the rescaling-by-RMS is what's
doing the work; centering is approximately free to drop. The op becomes
one mean and one rsqrt instead of two means, one rsqrt, and a centering
subtraction. LLaMA, T5, Gemma, Qwen, Mistral all use RMSNorm.

**Implementation detail**: the variance estimate is done in **fp32** even
if the input is bf16/fp16. Variance is the one place where norms can lose
precision and produce NaNs:
```python
dtype = x.dtype
x_f32 = x.float()
rms = x_f32.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
return (x_f32 * rms).to(dtype) * self.weight
```

Init: `γ = 1` so the module starts as approximate identity, important for
stable early training.

### §3 RoPE (lines 123–165) — Change 2c

The geometric idea, in one sentence: **pairs of dimensions in Q and K are
treated as 2D vectors and rotated by an angle proportional to their
position**.

For dimension pair `(2i, 2i+1)` at position `m`:
```
[q_{2i}, q_{2i+1}] → [q_{2i} cos(m·θ_i) - q_{2i+1} sin(m·θ_i),
                      q_{2i} sin(m·θ_i) + q_{2i+1} cos(m·θ_i)]
where θ_i = base^(-2i / d_head)
```

Why this is better than sinusoidal PE (Phase 0/1):

1. **Relative**. The dot product `q_m · k_n` after rotation depends only
   on `(m - n)`, not on absolute positions. That's exactly the property
   attention wants — "how far apart are these tokens", not "what's the
   absolute index".
2. **Composes with FlashAttention**. Sinusoidal PE adds to the embedding
   stream, RoPE rotates Q and K *before attention*. FlashAttention only
   sees Q/K/V, doesn't care that they were rotated first.
3. **Length generalisation** (with caveats). The rotation formula extends
   to any sequence length; with rope-scaling tricks you can use a model
   trained at 2k context at 32k+ context.

**`precompute_rope_cache(d_head, max_seq_len, base, device)`**: builds
`cos, sin` tables of shape `(max_seq_len, d_head/2)` once. Each row is
the precomputed `cos`/`sin` of every angle at that position. Stored as
a non-persistent buffer on the model so it moves with `.to(device)` but
doesn't bloat checkpoints.

**`apply_rope(x, cos, sin)`**:
- `x: (B, n_heads, T, d_head)`
- Split last dim into evens and odds:
  - `x1 = x[..., 0::2]` — `(B, h, T, d_head/2)`
  - `x2 = x[..., 1::2]` — `(B, h, T, d_head/2)`
- Compute rotated pair `(rx1, rx2) = (x1·cos - x2·sin, x1·sin + x2·cos)`.
- `torch.stack([rx1, rx2], dim=-1)` → `(..., d_head/2, 2)` then `.flatten(-2)`
  to get back to `(..., d_head)` with the original interleaving.

V is **not** rotated. V carries content; only Q and K need positional
information for the attention dot product.

### §4 `CausalSelfAttention` (lines 180–245)

Differences from Phase 1's `MultiHeadAttention`:

1. **Fused QKV projection**. One `Linear(d_model, 3·d_model)` then
   `chunk(3, dim=-1)`. Mathematically identical to three separate Linears
   but one matmul instead of three — faster on GPU.
2. **`bias=False`** everywhere. LLaMA convention: biases on attention
   projections add ~`d_model` params per linear, contribute nothing
   measurable to quality, and slow inference slightly.
3. **RoPE applied to Q and K**. The cache slice depends on whether we're
   in KV-cache mode (see below).
4. **`F.scaled_dot_product_attention`** in place of manual matmul +
   softmax. PyTorch dispatches this to FlashAttention-2 on Ampere+ when
   shapes/dtypes are compatible. **Memory drops from O(T²) to O(T)** for
   the attention itself — the single biggest win for an 8 GB GPU.
5. **`is_causal=True`** when no cache: no need to build a `(T, T)` mask
   tensor. SDPA applies the triangular mask internally and skips the
   masked matmul region entirely.

**KV-cache logic** (lines 213–229):

During generation, we feed one new token at a time. Each layer caches
its K and V tensors from previous steps and concatenates the new K, V:
```python
if past_kv is not None:
    past_k, past_v = past_kv
    k = torch.cat([past_k, k], dim=2)   # (B, h, past_T + T, d_head)
    v = torch.cat([past_v, v], dim=2)
new_kv = (k, v)
```

Two subtleties:

- **RoPE positions during caching**: when feeding one new token at the
  position `past_len`, we slice cos/sin at `[past_len : past_len + T]`,
  not `[0 : T]`. The cached K's were already rotated when they were
  stored, so we don't re-rotate them.
- **`is_causal` is False during decoding**: when `T_q=1` and `T_k=past_len+1`,
  the new query has to attend to **all** cached keys. The default causal
  mask would mask everything past position 0 — wrong. We explicitly set
  `is_causal = past_kv is None`.

### §5 `SwiGLU` (lines 263–272) — Change 2d

```
ReLU FFN:    FFN(x) = ReLU(xW₁) W₂                       (2 matrices, d_ff = 4·d)
SwiGLU FFN:  FFN(x) = (Swish(xW₁) ⊙ xW₃) W₂              (3 matrices, d_ff = 8/3·d)
                     ↑ gate        ↑ value
where Swish(x) = x · sigmoid(x)  (also called SiLU)
```

The gating — element-wise multiply of `Swish(xW₁)` and `xW₃` — is the
substantive change. Each FFN dimension can be independently amplified
or suppressed by its gate value. Shazeer 2020 ablated GLU, ReGLU, GEGLU,
SwiGLU; SwiGLU wins consistently.

LLaMA naming: `w1 = gate_proj`, `w3 = up_proj`, `w2 = down_proj`. Same
names appear in HuggingFace `transformers` for any LLaMA-class model.

The cost is one extra matmul. The 8/3 ratio compensates so total FFN
params match a ReLU FFN at `d_ff = 4·d`.

### §6 `TransformerBlock` (lines 287–307) — Change 2a (Pre-LN)

```
Post-LN (Phase 1):  x = LN(x + sublayer(x))
Pre-LN (Phase 2):   x = x + sublayer(LN(x))
```

In Post-LN, every layer normalises the residual stream, which means
information from early layers can be magnitudinally squashed before it
reaches deeper layers. This is why Post-LN networks need warmup — the
training is fragile in the first few thousand steps.

Pre-LN preserves a clean additive residual path from input to output —
the residual stream itself is never normalised, only the input to each
sublayer is. Xiong et al. 2020 showed Pre-LN trains stably without
warmup. Every modern decoder-only LM uses Pre-LN.

Two pre-norms per block, one before each sublayer:
```python
attn_out, new_kv = self.attn(self.attn_norm(x), cos, sin, past_kv=past_kv)
x = x + attn_out
x = x + self.ffn(self.ffn_norm(x))
```

A practical consequence: Pre-LN **requires a final norm** before `lm_head`,
otherwise the lm_head sees the un-normalised residual stream and outputs
can blow up. See §7.

### §7 `GPTv2` (lines 313–445)

**Init differences vs Phase 1's `GPT`**:
- No `PositionalEncoding` module — RoPE lives inside attention.
- `final_norm = RMSNorm(...)` before `lm_head`. Required for Pre-LN.
- Tied `lm_head.weight = embed.weight` (same as Phase 1).
- RoPE cos/sin precomputed once, registered as non-persistent buffers.
- **GPT-2 init** (Radford et al. 2019 §2.3):
  - All matrix weights: `N(0, 0.02)`.
  - **Residual-feeding** projections (`out_proj` in attention, `w2` in
    SwiGLU): `N(0, 0.02 / √(2N))` where N is `n_layers`. The intuition:
    each residual add can grow the activation magnitude by a factor
    related to `√layer_index`; scaling the output projection by `1/√(2N)`
    keeps the residual stream bounded as depth grows.
  - Norm weights stay at γ=1 from `RMSNorm.__init__`.

**Forward** (lines 365–397):
- **No `√d_model` embedding scaling**. With sinusoidal PE you needed
  this to balance the PE magnitude. RoPE doesn't perturb the embedding
  stream, so LLaMA et al. drop the scale.
- Slice the RoPE cache to `[: past_len + T]` so the cache covers all
  positions referenced this forward pass (cached + new).
- Run blocks; final norm; lm_head.
- Returns `(logits, new_past_kvs)` so the caller can chain calls during
  generation.

**KV-cache generation** (lines 404–445):
- First call: feed the entire prompt, get logits at every position and
  the K, V cache for every layer.
- Subsequent calls: feed only the new token (`T=1`). Layers concatenate
  with cached K, V. Per-token cost becomes O(T) instead of O(T²).

Sampling supports:
- `temperature=0` → deterministic greedy (argmax).
- `temperature>0, top_k=None` → temperature-scaled multinomial.
- `top_k=K` → mask logits outside the top-K before softmax. Combined
  with `temperature>0` this is the standard "top-k sampling" recipe.

---

## [train_lm_v2.py](train_lm_v2.py) — section-by-section

Most of the trainer is reused from [Phase 1](../phase1/) — same WikiText-2
data pipeline (`BlockDataset`, `collate_blocks`, `load_wikitext2`), same
`LabelSmoothingLoss`. Three things change.

### §1 `CosineWarmupScheduler` (lines 73–101)

```
step < warmup:        lr = max_lr · (step / warmup)
warmup ≤ step < T:    lr = min_lr + 0.5·(max_lr - min_lr)·(1 + cos(π·progress))
                            progress = (step - warmup) / (T - warmup)
step ≥ T:             lr = min_lr
```

Linear warmup then cosine decay to `min_lr` (typically 10% of `max_lr`).
The GPT-3 / Chinchilla / LLaMA schedule. Replaces Phase 0/1's Noam
schedule, which was tuned for Post-LN's gradient instability — Pre-LN
doesn't need the aggressive 1/√step decay.

### §2 Selective weight decay (in `main()`)

AdamW's weight decay pulls parameters toward zero. Good regulariser for
**matrix weights**, harmful for:
- **Norm gains (γ)**: pulling toward zero kills the layer.
- **Biases**: same (we have none in this model, but the rule still holds).
- **Embedding rows**: empirically helps to not decay them (LLaMA does this).

The convention: any parameter with `dim < 2` (norms, biases) or whose
name contains `embed` goes into a no-decay param group. Everything else
(linear weights) gets `weight_decay=0.1`.

This shows up as two param groups in the optimizer init:
```python
optimizer = torch.optim.AdamW(
    [{"params": decay,    "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=max_lr, betas=(0.9, 0.95), eps=1e-8,
)
```

`(beta1, beta2) = (0.9, 0.95)` is the LLaMA / Chinchilla setting.
Phase 0/1 used `(0.9, 0.98)` (the Vaswani et al. 2017 setting).

### §3 KV-cache generation in sample logging
Each epoch logs a generated sample. With KV-cache the cost is O(T) instead
of O(T²), so we get this for free.

---

## Result

WikiText-2, 5 epochs, ~6.8M params, identical training budget:

| Phase | Val loss | Val perplexity |
|-------|---------:|---------------:|
| Phase 1 (Post-LN, sinusoidal, ReLU, Noam, Adam) | 4.27 | 71.2 |
| Phase 2 (Pre-LN, RMSNorm, RoPE, SwiGLU, cosine, AdamW) | 4.16 | **64.1** |

That's the LLaMA architecture's gain at the smallest possible scale,
isolated from any data or compute differences.

---

## What changes next (Phase 3 preview)

Phase 3 doesn't touch the architecture at all — every config knob is
already exposed by `GPTConfigV2`. What Phase 3 adds:

- **Gradient checkpointing** wrapper (Chen et al. 2016) — trade ~30%
  compute for ~4–5× less activation memory. Needed once `d_model` and
  `n_layers` push past what fits on an 8 GB GPU.
- **Memory-mapped binary data loader** — train.bin/val.bin as raw
  uint16 arrays, zero-RAM startup, billions of tokens supported.
- **Step-level training loop** with full resume (model + optimizer +
  RNG + step counter).
- **Domain shift**: WikiText-2 → Python code from The Stack.

The architecture file in Phase 3 (`decoder_only_v3.py`) inherits from
`GPTConfigV2` and `TransformerBlock` directly — only the top-level model
adds a `gradient_checkpointing` flag.
