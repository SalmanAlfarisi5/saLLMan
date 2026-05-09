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
├── .env                 HuggingFace token (git-ignored — you create this)
├── phase0/              Vanilla encoder-decoder Transformer (Vaswani et al. 2017)
│   ├── transformer.py   Model definition
│   └── train.py         DE→EN translation on Multi30k
│
├── phase1/              Decoder-only GPT-style LM (Brown et al. 2020)
│   ├── decoder_only.py  Model definition (reuses phase0 primitives)
│   └── train_lm.py      Next-token prediction on WikiText-2
│
├── phase2/              Modernized LM: Pre-LN + RMSNorm + RoPE + SwiGLU
│   ├── decoder_only.py  Self-contained model definition
│   └── train_lm.py      Training with AdamW + cosine LR + KV-cache generation
│
└── phase3/              Production-scale 75M-param code LM
    ├── decoder_only.py  GPTv3 — adds gradient checkpointing to GPTv2
    ├── data_prep.py     Tokenizer training + binary corpus preparation
    └── pretrain.py      Full pretraining loop with memmap data + resumable checkpoints
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
pip install torch datasets tokenizers python-dotenv
# Phase 3 also needs:
pip install numpy tqdm
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

### Phase 3 — Code LM Pretraining

Phase 3 is a two-step process: data preparation (one-time) then training.

**Step 1 — Prepare the corpus** (~30–60 min, mostly download time):

```bash
cd phase3
python data_prep.py
```

This produces `phase3/pretrain_data/` containing `train.bin`, `val.bin`,
`bpe_tokenizer.json`, and `meta.json`. It only needs to run once — subsequent
runs detect the output files and skip.

**Step 2 — Pretrain** (~hours on a single GPU):

```bash
python pretrain.py
```

Resume a crashed or interrupted run from the last checkpoint:

```bash
python pretrain.py --resume
```

If you run out of VRAM, enable gradient checkpointing (~30% slower, ~4× less
activation memory):

```bash
python pretrain.py --gradient_checkpointing
```

Other CLI overrides (useful for quick sanity checks):

```bash
python pretrain.py --micro_batch 2 --grad_accum 4 --total_steps 100
```

Checkpoints are saved to `phase3/checkpoints_pretrain/`. `last.pt` is
overwritten every 1 000 steps; `best.pt` is updated whenever validation loss
improves. Progress is also appended to `checkpoints_pretrain/log.jsonl`.

**Smoke-test the model** (forward + checkpointed backward, no data needed):

```bash
python decoder_only.py
```

---

## Cross-phase Imports

Each phase directory has an `__init__.py` that re-exports its public API,
so you can import from any phase as a package from the project root:

```python
from phase0.transformer import Transformer, TransformerConfig
from phase1.decoder_only import GPT, GPTConfig
from phase2.decoder_only import GPTv2, GPTConfigV2
from phase3.decoder_only import GPTv3, GPTConfigV3

# Shared utilities
from phase0.train import LabelSmoothingLoss, NoamScheduler
from phase1.train_lm import BlockDataset, collate_blocks, load_wikitext2
```

### Phase 3 — Production-scale Code LM (75M params)

Scales from the ~7M-parameter WikiText-2 models to a 75M-parameter code LM
trained on a mixed Python corpus. Every component is designed for a long
multi-hour run on a single 8 GB GPU (RTX 3060 Ti).

#### Model — GPTv3 (`decoder_only.py`)

Same architecture as Phase 2 (Pre-LN, RMSNorm, RoPE, SwiGLU, FlashAttention,
KV-cache) with one addition: **gradient checkpointing** (Chen et al. 2016).

| Hyperparameter | Value |
|---------------|-------|
| d_model | 512 |
| n_heads | 8 |
| n_layers | 12 |
| d_ff | 1408 (= ⌈8/3 × 512⌉₆₄, SwiGLU) |
| Parameters | ~75M |
| max_len | 512 |

**Why gradient checkpointing?** Activation memory scales as
`O(n_layers × batch × seq_len × d_model)`. At this scale (12 layers, batch 4,
seq 512, d_model 512), activations alone can consume ~6 GB. Gradient
checkpointing discards intermediate activations during the forward pass and
recomputes them on-the-fly during backprop — trading ~30% extra compute for a
~4–5× reduction in peak activation memory, making the model trainable on 8 GB.
It is automatically disabled during inference (`model.eval()`) and during
KV-cache generation, where checkpointing and caching are incompatible.

#### Data — `data_prep.py`

A one-time preprocessing step that produces memory-mappable binary token files
for the main training loop.

**Corpus mix** (all permissively licensed):

| Dataset | Content | Size |
|---------|---------|------|
| `bigcode/the-stack-smol` (Python) | Real-world Python files | ~10k files |
| `code_search_net` (Python) | Docstring + function pairs | ~450k examples |
| `codeparrot/apps` | Algorithmic problem + solution pairs | 10k problems |

The mix is intentional: `the-stack-smol` provides general Python fluency;
CodeSearchNet teaches the model to associate natural-language descriptions with
code; APPS exposes it to the exact "problem statement → solution" format that
is the end goal of saLLMan.

Each `(docstring, code)` pair from CodeSearchNet is formatted as:
```python
"""
<docstring>
"""
<code>
```
Each APPS entry as:
```python
# Problem
<question>

# Solution
<solution>
```
These formatting conventions appear verbatim during pretraining, so the model
internalizes the natural-language-to-code prompt structure before fine-tuning.

**Tokenizer:** A fresh byte-level BPE tokenizer with **vocab size 16,000**
(2× the Phase 1/2 vocab). The larger vocab is necessary for code — Python
keywords (`def`, `self`, `return`, `range`) are high-frequency and should be
single tokens; a natural-language BPE wastes slots on whitespace run patterns.

**Output files** (written to `phase3/pretrain_data/`):

| File | Description |
|------|-------------|
| `bpe_tokenizer.json` | Trained tokenizer |
| `train.bin` | uint16 token stream for training |
| `val.bin` | uint16 token stream for validation |
| `meta.json` | Vocab size, token counts, special token indices |

Tokens are stored as `uint16` (16 000 < 65 535), halving disk and memory cost
versus `int32`. This is the same trick used by nanoGPT.

> **Note:** `bigcode/the-stack-smol` is a gated dataset. See
> [Authentication](#authentication) below before running `data_prep.py`.

#### Training — `pretrain.py`

**Recipe summary:**

| Setting | Value | Reasoning |
|---------|-------|-----------|
| micro_batch | 4 | Fits in 8 GB with gradient checkpointing |
| grad_accum_steps | 16 | Effective batch = 64, 32 768 tokens/step |
| total_steps | 20 000 | ~640M tokens seen (~3–4× dataset) |
| max_lr | 3e-4 | LLaMA recipe at this scale |
| min_lr | 3e-5 | 10% of max_lr |
| warmup_steps | 1 000 | Linear ramp before cosine decay |
| weight_decay | 0.1 | GPT-3 / LLaMA value |
| grad_clip | 1.0 | Standard |

**Key engineering details:**

- **Memory-mapped data loading** — `train.bin` / `val.bin` are opened with
  `numpy.memmap` in read-only mode. The OS pages in only the slices that are
  actually accessed, so startup is instant and the full dataset never occupies
  RAM. Essential when the corpus is larger than available memory.

- **Random window sampling** — each training step samples random start offsets
  into the token stream rather than iterating sequentially. This provides
  effective infinite shuffling with no index bookkeeping and prevents
  consecutive steps from seeing highly correlated contexts.

- **Gradient accumulation** — `backward()` is called on each of the 16
  micro-batches before a single `optimizer.step()`. Loss is divided by
  `grad_accum_steps` before each backward so gradient magnitude matches a true
  large-batch run. The gradient norm clip is applied once, across the fully
  accumulated gradient.

- **Fused AdamW** — when available (PyTorch 2.x on CUDA), a single fused
  kernel executes the entire Adam parameter update, replacing 4–5 separate
  element-wise kernel launches per parameter group with one.

- **TF32 matmuls** — `torch.set_float32_matmul_precision("high")` enables
  Ampere's TF32 mode for fp32 matrix multiplications: effectively fp16 compute
  with fp32 accumulation, giving ~2× throughput on Ampere with no accuracy loss.

- **Plain cross-entropy** — label smoothing is dropped for pretraining.
  Smoothing helps regularize small-data tasks (translation, classification) but
  slightly hurts perplexity at scale by spreading probability mass away from the
  true next token. GPT-3 and LLaMA use plain cross-entropy; we follow suit.

- **Resumable training** — checkpoints include model state, optimizer state,
  current step, and both numpy and PyTorch RNG states. Saves are atomic
  (write to `last.tmp`, then `os.replace` → `last.pt`) so a crash mid-save
  never corrupts the checkpoint. Resume with `python pretrain.py --resume`.

- **JSONL training log** — every log and eval event is appended to
  `checkpoints_pretrain/log.jsonl` for offline plotting.

---

## Authentication

`bigcode/the-stack-smol` is a gated dataset on HuggingFace Hub. Two steps:

**1. Accept the terms** — log in at [huggingface.co](https://huggingface.co)
and click **Access repository** on the
[bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol)
dataset page.

**2. Add your token to `.env`** at the project root:

```
HF_TOKEN=hf_your_token_here
```

Generate a Read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
All training scripts load this file automatically via `python-dotenv`.
The `.env` file is git-ignored and never committed.

---

## Results

**Phase 1 vs Phase 2** — WikiText-2 (~2 M tokens), 5 epochs, same model size
(~6.8 M parameters, d_model = 256, 6 layers, 8 heads), block size = 256.

| Phase | Architecture | Best val loss | Perplexity |
|-------|-------------|:-------------:|:----------:|
| Phase 1 | GPT (Post-LN, sinusoidal PE, ReLU FFN, Noam LR) | 4.266 | 71.2 |
| Phase 2 | LLaMA-style (Pre-LN, RMSNorm, RoPE, SwiGLU, cosine LR) | 4.161 | 64.1 |

Phase 2 achieves **9.9 % lower perplexity** than Phase 1 at identical parameter
count and training budget, attributable to the combined effect of Pre-LN
training stability, RoPE's relative position encoding, and the SwiGLU gated FFN.

**Phase 3** — Code + DSA corpus, 75 M parameters, 20 000 steps (~640 M tokens).
Training in progress; results will be added here on completion.

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

**Why drop label smoothing in Phase 3?**
At pretraining scale, smoothing slightly hurts perplexity by redistributing
probability away from the true next token toward the uniform noise floor.
GPT-3 and LLaMA both use plain cross-entropy. Phase 3 follows that convention,
and the `ignore_index=PAD_IDX` argument to `F.cross_entropy` handles padding
without needing a custom loss class.

**Why a separate 16k BPE tokenizer in Phase 3?**
The 8k vocab from Phase 1/2 was trained on Wikipedia prose. Python code has a
different token distribution: keyword tokens (`def`, `self`, `return`, `class`,
`None`) are extremely high frequency and should be single tokens. A prose BPE
wastes slots on whitespace n-gram patterns and will fragment common Python
identifiers. Training a fresh tokenizer on the code corpus gives better
compression (fewer tokens per file) and cleaner subword boundaries.

**Why memory-mapped binary files instead of streaming?**
Streaming from HuggingFace re-downloads or re-tokenizes on every epoch.
Binary memmap files are tokenized once (`data_prep.py`), stored as raw
`uint16` on disk, and read in zero-copy fashion during training — the OS only
pages in the slices that are actually needed. At corpus sizes of hundreds of
millions of tokens this is the only practical approach on a single machine.

**Why random window sampling instead of sequential iteration?**
Sequential iteration would make consecutive training steps see highly
correlated contexts (adjacent blocks of the same document). Random sampling
decorrelates steps, avoids the need to shuffle a large index array, and
trivially supports "infinite" training beyond one epoch.

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
- Chen et al. 2016 — [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (gradient checkpointing)
- Kocetkov et al. 2022 — [The Stack: 3 TB of permissively licensed source code](https://arxiv.org/abs/2211.15533)
- Husain et al. 2019 — [CodeSearchNet Challenge](https://arxiv.org/abs/1909.09436)
- Hendrycks et al. 2021 — [Measuring Coding Challenge Competence with APPS](https://arxiv.org/abs/2105.09938)
