# saLLMan — Transformer Progression from Scratch

A from-scratch PyTorch implementation tracing the evolution of Transformer
architectures from the 2017 "Attention Is All You Need" paper through
LLaMA-class modern decoder-only models.

Each phase is a complete, runnable training pipeline. The goal is to make
every architectural upgrade legible: what changed, why, and what it buys you.

---

## Project Structure

```
saLLMan/
├── phase0/          Vanilla encoder-decoder Transformer (Vaswani et al. 2017)
│   ├── transformer.py   Model definition
│   └── train.py         DE→EN translation on Multi30k
│
├── phase1/          Decoder-only GPT-style LM (Brown et al. 2020)
│   ├── decoder_only.py  Model definition (reuses phase0 primitives)
│   └── train_lm.py      Next-token prediction on WikiText-2
│
└── phase2/          Modernized LM: Pre-LN + RMSNorm + RoPE + SwiGLU
    ├── decoder_only.py  Self-contained model definition
    └── train_lm.py      Training with AdamW + cosine LR + KV-cache generation
```

---

## Architecture Progression

### Phase 0 — Vanilla Transformer

Reference implementation of Vaswani et al. 2017, §3 ("Attention Is All You
Need"). Encoder-decoder architecture trained on German→English translation.

| Component | Implementation |
|-----------|---------------|
| Attention | Scaled dot-product, eq. (1) |
| PE | Sinusoidal, §3.5 |
| Norm | Post-LN: `LN(x + sublayer(x))` |
| FFN | ReLU, d_ff = 4 × d_model |
| Loss | Label smoothing, ε = 0.1 (§5.4) |
| LR | Noam schedule, warmup = 4000 (§5.3) |

### Phase 1 — Decoder-only GPT

Removes the encoder and cross-attention. A single stack of masked
self-attention + FFN blocks trained on WikiText-2 language modelling.

Changes from Phase 0:
- No encoder; no cross-attention sublayer
- Single vocabulary (no src/tgt split)
- BPE tokenizer (byte-level, vocab = 8 000) instead of regex word tokeniser
- Block-packed dataset (no padding, every token contributes to loss)
- Mixed-precision training with `torch.amp.autocast` (bfloat16)

### Phase 2 — Modernized Decoder (LLaMA-class)

Applies every post-2020 improvement used in GPT-NeoX, LLaMA, Mistral, and
DeepSeek. Same data pipeline as Phase 1 so loss curves are directly comparable.

| Change | Reference |
|--------|-----------|
| Pre-LN: `x = x + sublayer(norm(x))` | Xiong et al. 2020 |
| RMSNorm (no centering, no bias) | Zhang & Sennrich 2019 |
| RoPE (rotary position embeddings) | Su et al. 2021 |
| SwiGLU FFN, d_ff = ⌈8/3 × d_model⌉₆₄ | Shazeer 2020 |
| FlashAttention via `F.scaled_dot_product_attention` | PyTorch 2.0 |
| KV-cache O(T) generation | — |
| GPT-2 init: N(0, 0.02), residual scaling 1/√(2N) | Radford et al. 2019 |
| Bias-free Linear layers | LLaMA convention |
| Selective weight decay (AdamW) | GPT-3 / LLaMA recipe |
| Cosine warmup LR schedule | Chinchilla / LLaMA |

---

## Setup

```bash
pip install torch datasets tokenizers
```

Python 3.10+ is required (uses `X | Y` union type hints).

---

## Running

All scripts can be run from the **project root** or from within their phase
directory. The `sys.path` setup at the top of each file handles both cases.

### Phase 0 — Translation

```bash
# Smoke-test the model (no training):
python phase0/transformer.py

# Full training on Multi30k DE→EN (~15 epochs, ~5 min on GPU):
python phase0/train.py
```

Checkpoints are saved to `phase0/checkpoints/best_model.pt`.

### Phase 1 — Language Modelling

```bash
# Smoke-test the model:
python phase1/decoder_only.py

# Full training on WikiText-2 (~5 epochs):
python phase1/train_lm.py
```

The BPE tokeniser is trained on first run and cached to
`tok_cache/wikitext2_bpe.json` (relative to the working directory).
Checkpoints are saved to `phase1/checkpoints_lm/best_model.pt`.

### Phase 2 — Modernized LM

```bash
# Smoke-test the model (forward pass + KV-cache generation):
python phase2/decoder_only.py

# Full training on WikiText-2 (~5 epochs):
python phase2/train_lm.py
```

Checkpoints are saved to `phase2/checkpoints_lm_v2/best_model.pt`.
Phase 2 reuses the BPE tokeniser trained in Phase 1 (same cache path).

---

## Cross-phase Imports

Each phase directory has an `__init__.py` that re-exports its public API,
so you can import from any phase as a package from the project root:

```python
from phase0.transformer import Transformer, TransformerConfig
from phase1.decoder_only import GPT, GPTConfig
from phase2.decoder_only import GPTv2, GPTConfigV2

# Shared utilities
from phase0.train import LabelSmoothingLoss, NoamScheduler
from phase1.train_lm import BlockDataset, collate_blocks, load_wikitext2
```

---

## Results

Both phases trained on WikiText-2 (~2 M tokens), 5 epochs, same model size
(~6.8 M parameters, d_model = 256, 6 layers, 8 heads), block size = 256.

| Phase | Architecture | Best val loss | Perplexity |
|-------|-------------|:-------------:|:----------:|
| Phase 1 | GPT (Post-LN, sinusoidal PE, ReLU FFN, Noam LR) | 4.266 | 71.2 |
| Phase 2 | LLaMA-style (Pre-LN, RMSNorm, RoPE, SwiGLU, cosine LR) | 4.161 | 64.1 |

Phase 2 achieves **9.9 % lower perplexity** than Phase 1 at identical parameter
count and training budget, attributable to the combined effect of Pre-LN
training stability, RoPE's relative position encoding, and the SwiGLU gated FFN.

---

## Key Design Decisions

**Why separate phases instead of one configurable model?**
Each phase is intentionally self-contained so you can read `phase1/decoder_only.py`
and understand the architecture without chasing imports. The shared utilities
(loss, scheduler, data) are pulled in explicitly so the dependency graph is
always visible.

**Why WikiText-2 and not something bigger?**
WikiText-2 (~2 M tokens) trains in minutes on a single GPU and is large
enough to clearly distinguish architectures by perplexity. Phase 2 reaches
~10 % lower perplexity than Phase 1 at the same parameter count (see Results).

**Why label smoothing in Phase 2?**
Kept for a fair comparison with Phase 1. Most production LMs use plain
cross-entropy once the dataset is large enough.

---

## References

- Vaswani et al. 2017 — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Radford et al. 2018 — [GPT-1](https://openai.com/research/language-unsupervised)
- Radford et al. 2019 — [GPT-2](https://openai.com/research/better-language-models)
- Brown et al. 2020 — [GPT-3](https://arxiv.org/abs/2005.14165)
- Xiong et al. 2020 — [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- Zhang & Sennrich 2019 — [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Su et al. 2021 — [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Shazeer 2020 — [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Touvron et al. 2023 — [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
