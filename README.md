# saLLMan — Transformer Progression from Scratch

From-scratch PyTorch implementation tracing the evolution of Transformers from
Vaswani et al. 2017 through LLaMA-class code models. Each phase is a complete,
runnable training pipeline where every architectural change is annotated.

---

## Project Structure

```
saLLMan/
├── .env                    HuggingFace token (git-ignored)
├── phase0/                 Vanilla encoder-decoder Transformer
│   ├── transformer.py      Model (Vaswani et al. 2017)
│   └── train.py            DE→EN translation on Multi30k
├── phase1/                 Decoder-only GPT
│   ├── decoder_only.py     Model (reuses phase0 primitives)
│   └── train_lm.py         WikiText-2 language modelling
├── phase2/                 LLaMA-class modernised decoder
│   ├── decoder_only_v2.py  Self-contained model
│   └── train_lm_v2.py      AdamW + cosine LR + KV-cache
└── phase3/                 75M-param code LM
    ├── decoder_only_v3.py  GPTv3 (adds gradient checkpointing)
    ├── data_prep.py        Tokeniser training + binary corpus
    └── pretrain.py         Full pretraining loop with resume
```

---

## Architecture Progression

| | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| **Task** | DE→EN translation | WikiText-2 LM | WikiText-2 LM | Code pretraining |
| **Structure** | Encoder-decoder | Decoder-only | Decoder-only | Decoder-only |
| **Norm** | Post-LN | Post-LN | Pre-LN + RMSNorm | Pre-LN + RMSNorm |
| **Position** | Sinusoidal | Sinusoidal | RoPE | RoPE |
| **FFN** | ReLU | ReLU | SwiGLU | SwiGLU |
| **Attention** | Manual SDPA | Manual SDPA | FlashAttention | FlashAttention |
| **LR schedule** | Noam | Noam | Cosine warmup | Cosine warmup |
| **Optimizer** | Adam | Adam | AdamW | AdamW (fused) |
| **Loss** | Label smoothing | Label smoothing | Label smoothing | Cross-entropy |
| **Params** | ~8M | ~6.8M | ~6.8M | ~75M |

### Phase 0 — Vanilla Transformer

Reference implementation of Vaswani et al. 2017. Encoder-decoder with
sinusoidal PE, scaled dot-product attention, ReLU FFN, Post-LN, Noam schedule,
and label smoothing (ε = 0.1). Trained on Multi30k German→English.

### Phase 1 — Decoder-only GPT

Drops the encoder and cross-attention sublayer. Key additions: byte-level BPE
tokeniser (8k vocab), block-packed dataset (zero padding waste), and bfloat16
mixed precision. Reuses `MultiHeadAttention`, `PositionwiseFeedForward`, and
`PositionalEncoding` directly from Phase 0.

### Phase 2 — Modernised Decoder (LLaMA-class)

Every post-2020 improvement used in GPT-NeoX, LLaMA, and Mistral applied to
the Phase 1 architecture:

| Change | Reference |
|--------|-----------|
| Pre-LN: `x = x + sublayer(norm(x))` | Xiong et al. 2020 |
| RMSNorm — drop centering + bias | Zhang & Sennrich 2019 |
| RoPE — rotary position embeddings | Su et al. 2021 |
| SwiGLU FFN, d_ff = ⌈8/3 × d_model⌉₆₄ | Shazeer 2020 |
| FlashAttention via `F.scaled_dot_product_attention` | PyTorch 2.0 |
| KV-cache — O(T) generation | — |
| GPT-2 init: N(0, 0.02), residual ×1/√(2N) | Radford et al. 2019 |
| Bias-free Linear layers | LLaMA convention |
| Selective weight decay (AdamW) | GPT-3 / LLaMA |
| Cosine warmup LR | Chinchilla / LLaMA |

### Phase 3 — Production-scale Code LM

Scales to 75M parameters and shifts domain to Python code. Built on the Phase 2
architecture with one addition: **gradient checkpointing** (Chen et al. 2016)
— recomputes activations during backward instead of storing them, trading ~30%
extra compute for ~4–5× less peak activation memory. Essential at this scale on
an 8 GB GPU.

**Model config:** d_model=512, n_heads=8, n_layers=12, d_ff=1408, max_len=512

**Data (`data_prep.py`):** Downloads `bigcode/the-stack-smol` Python subset
(~10k files, ~24M tokens) via `hf_hub_download`, bypassing deprecated dataset
loading scripts. Trains a 16k byte-level BPE tokeniser on the training split,
then writes `train.bin` / `val.bin` as raw `uint16` arrays (nanoGPT pattern)
for zero-copy memory-mapped loading. The larger vocab (vs 8k in Phase 1/2) is
necessary for code — Python keywords must be single tokens.

**Training (`pretrain.py`):**

| Setting | Value |
|---------|-------|
| micro_batch / accum | 4 / 16 → effective batch 64, 32k tokens/step |
| total_steps | 20 000 (~640M tokens, ~3–4× corpus) |
| LR | 3e-4 → 3e-5 cosine, 1k warmup steps |
| weight_decay | 0.1, selective (no decay on norms/embeddings) |

Key engineering: memmap data loading (zero RAM), random window sampling
(implicit shuffle), fused AdamW, TF32 matmuls on Ampere, atomic checkpoint
saves (tmp → rename), JSONL training log.

---

## Setup

```bash
pip install torch datasets tokenizers python-dotenv numpy tqdm huggingface_hub
```

Python 3.10+ required (`X | Y` union type hints).

**HuggingFace auth** — required for `bigcode/the-stack-smol` (gated):
1. Accept terms at [huggingface.co/datasets/bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol)
2. Create a Read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Add to `.env` at project root: `HF_TOKEN=hf_your_token_here`

All scripts load `.env` automatically via `python-dotenv`.

---

## Running

All scripts work from the **project root** or from within the phase directory.

### Phase 0
```bash
python phase0/transformer.py   # smoke test
python phase0/train.py         # train (~15 epochs on Multi30k)
```
Checkpoints → `phase0/checkpoints/best_model.pt`

### Phase 1
```bash
python phase1/decoder_only.py  # smoke test
python phase1/train_lm.py      # train (~5 epochs on WikiText-2)
```
BPE tokeniser cached to `tok_cache/wikitext2_bpe.json`. Checkpoints → `phase1/checkpoints_lm/best_model.pt`

### Phase 2
```bash
python phase2/decoder_only.py  # smoke test
python phase2/train_lm.py      # train (~5 epochs on WikiText-2)
```
Checkpoints → `phase2/checkpoints_lm_v2/best_model.pt`

### Phase 3
```bash
# Step 1 — one-time data prep (run from phase3/)
cd phase3
python data_prep.py            # outputs to pretrain_data/ by default

# Step 2 — pretrain
python pretrain.py
python pretrain.py --resume                          # resume from last checkpoint
python pretrain.py --gradient_checkpointing          # if OOM (~30% slower, ~4× less VRAM)
python pretrain.py --micro_batch 2 --total_steps 100 # quick sanity check
```
`pretrain_data/` contains `bpe_tokenizer.json`, `train.bin`, `val.bin`, `meta.json`.
Checkpoints → `checkpoints_pretrain/last.pt` (every 1k steps) and `best.pt` (best val).
Training log → `checkpoints_pretrain/log.jsonl`.

---

## Cross-phase Imports

Each phase has an `__init__.py`. From the project root:

```python
from phase0.transformer import Transformer, TransformerConfig
from phase1.decoder_only import GPT, GPTConfig
from phase2.decoder_only import GPTv2, GPTConfigV2
from phase3.decoder_only_v3 import GPTv3, GPTConfigV3

from phase0.train import LabelSmoothingLoss, NoamScheduler
from phase1.train_lm import BlockDataset, collate_blocks, load_wikitext2
```

---

## Results

WikiText-2, 5 epochs, ~6.8M params (d_model=256, 6 layers, 8 heads), block=256.

| Phase | Architecture | Val loss | PPL |
|-------|-------------|:--------:|:---:|
| Phase 1 | GPT (Post-LN, sinusoidal, ReLU, Noam) | 4.266 | 71.2 |
| Phase 2 | LLaMA-style (Pre-LN, RMSNorm, RoPE, SwiGLU, cosine) | 4.161 | 64.1 |

Phase 2 achieves **9.9% lower perplexity** at identical parameter count and
training budget — attributable to Pre-LN stability, RoPE, and SwiGLU combined.

Phase 3 (75M params, code corpus, 20k steps): training in progress.

---

## References

- Vaswani et al. 2017 — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Radford et al. 2019 — [GPT-2](https://openai.com/research/better-language-models)
- Brown et al. 2020 — [GPT-3](https://arxiv.org/abs/2005.14165)
- Xiong et al. 2020 — [On Layer Normalization in Transformers](https://arxiv.org/abs/2002.04745)
- Zhang & Sennrich 2019 — [RMSNorm](https://arxiv.org/abs/1910.07467)
- Su et al. 2021 — [RoPE](https://arxiv.org/abs/2104.09864)
- Shazeer 2020 — [SwiGLU](https://arxiv.org/abs/2002.05202)
- Touvron et al. 2023 — [LLaMA](https://arxiv.org/abs/2302.13971)
- Chen et al. 2016 — [Gradient Checkpointing](https://arxiv.org/abs/1604.06174)
- Kocetkov et al. 2022 — [The Stack](https://arxiv.org/abs/2211.15533)
