# saLLMan — Transformer Progression from Scratch

From-scratch PyTorch implementation tracing the evolution of Transformers from
Vaswani et al. 2017 through LLaMA-class code models. Each phase is a complete,
runnable training pipeline where every architectural change is annotated.

Final target: **saLLMan** — *Step-aware Large Language Model for Algorithm
Navigation* — a decoder-only LLM specialised in DSA (Data Structures &
Algorithms) reasoning. Built on an RTX 3060 Ti (8 GB VRAM).

> Each phase folder has its own README with line-by-line code walkthroughs
> and paper cross-references. Use those as study notes; this file is the
> map.
>
> → [phase0/README.md](phase0/README.md) — vanilla Transformer
> → [phase1/README.md](phase1/README.md) — decoder-only GPT
> → [phase2/README.md](phase2/README.md) — Pre-LN + RMSNorm + RoPE + SwiGLU
> → [phase3/README.md](phase3/README.md) — code pretraining at scale

---

## Project Structure

```
saLLMan/
├── .env                          HuggingFace token (git-ignored)
├── phase0/                       Vanilla encoder-decoder Transformer
│   ├── README.md                 Concept walkthrough
│   ├── transformer.py            Model (Vaswani et al. 2017)
│   └── train.py                  DE→EN translation on Multi30k
├── phase1/                       Decoder-only GPT
│   ├── README.md
│   ├── decoder_only.py           Model (reuses phase0 primitives)
│   └── train_lm.py               WikiText-2 language modelling
├── phase2/                       LLaMA-class modernised decoder
│   ├── README.md
│   ├── decoder_only_v2.py        Self-contained model
│   └── train_lm_v2.py            AdamW + cosine LR + KV-cache
└── phase3/                       Code pretraining
    ├── README.md
    ├── decoder_only_v3.py        GPTv3 (Phase 2 + gradient checkpointing)
    ├── data_prep.py              The Stack Python → train.bin / val.bin
    ├── pretrain.py               Full pretraining loop with resume
    ├── pretrain_data/            v1 (smol-baseline) data — preserved
    ├── pretrain_data_v2/         v2 (~2B-token) data — current target
    ├── checkpoints_pretrain/     v1 model checkpoints — preserved
    └── checkpoints_pretrain_v2/  v2 model checkpoints — current target
```

---

## Architecture Progression

|                  | Phase 0           | Phase 1               | Phase 2                   | Phase 3 (v2)             |
|------------------|-------------------|-----------------------|---------------------------|--------------------------|
| **Task**         | DE→EN translation | WikiText-2 LM         | WikiText-2 LM             | Python code LM           |
| **Structure**    | Encoder-decoder   | Decoder-only          | Decoder-only              | Decoder-only             |
| **Norm**         | Post-LN           | Post-LN               | Pre-LN + RMSNorm          | Pre-LN + RMSNorm         |
| **Position**     | Sinusoidal        | Sinusoidal            | RoPE                      | RoPE                     |
| **FFN**          | ReLU              | ReLU                  | SwiGLU                    | SwiGLU                   |
| **Attention**    | Manual SDPA       | Manual SDPA           | FlashAttn (`F.SDPA`)      | FlashAttn (`F.SDPA`)     |
| **Bias on Linear** | yes             | yes                   | no (LLaMA conv.)          | no                       |
| **LR schedule**  | Noam              | Noam                  | Cosine + linear warmup    | Cosine + linear warmup   |
| **Optimizer**    | Adam              | Adam                  | AdamW (selective WD)      | AdamW (fused)            |
| **Loss**         | Label smoothing   | Label smoothing       | Label smoothing           | Cross-entropy            |
| **Precision**    | fp32              | bf16 autocast         | bf16 autocast             | bf16 autocast            |
| **Generation**   | greedy (no cache) | greedy (no cache)     | KV-cache                  | KV-cache                 |
| **Memory tricks** | —                | —                     | —                         | Grad checkpointing (opt) |
| **Data loading** | DataLoader        | DataLoader            | DataLoader                | `np.memmap` + random window |
| **Params**       | ~8M               | ~6.8M                 | ~6.8M                     | ~97M                     |

### Phase 0 — Vanilla Transformer

Reference implementation of [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
Encoder-decoder, sinusoidal PE, scaled dot-product attention, ReLU FFN,
Post-LN, Noam schedule, label smoothing ε=0.1. Trained on Multi30k DE→EN.
Establishes all primitives (`MultiHeadAttention`, `PositionwiseFeedForward`,
`PositionalEncoding`, mask helpers, `LabelSmoothingLoss`, `NoamScheduler`)
that later phases reuse.

→ Deep dive: [phase0/README.md](phase0/README.md)

### Phase 1 — Decoder-Only GPT

Drops the encoder and cross-attention sublayer. Adds byte-level BPE
tokeniser (8k vocab), block-packed dataset (zero padding waste), bfloat16
mixed precision. Same Phase 0 primitives reused unchanged — the point of
the phase is to **isolate the decoder-only transition** and prove it
trains.

→ Deep dive: [phase1/README.md](phase1/README.md)

### Phase 2 — Modernised Decoder (LLaMA-class)

Every post-2020 improvement, applied together. Same parameter count as
Phase 1, **9.9 % lower val PPL** (64.1 vs 71.2) on WikiText-2 — pure
architectural gain.

| Change | Reference |
|--------|-----------|
| Pre-LN: `x = x + sublayer(norm(x))` | Xiong et al. 2020 |
| RMSNorm — drop centering + bias | Zhang & Sennrich 2019 |
| RoPE — rotary position embeddings | Su et al. 2021 |
| SwiGLU FFN, `d_ff = ⌈8/3 · d_model⌉₆₄` | Shazeer 2020 |
| FlashAttention via `F.scaled_dot_product_attention` | Dao 2022/2023 |
| KV-cache — O(T) generation | — |
| GPT-2 init: N(0, 0.02), residual ×1/√(2N) | Radford et al. 2019 |
| Bias-free Linear layers | LLaMA convention |
| Selective weight decay (AdamW) | GPT-3 / LLaMA |
| Cosine warmup LR | Chinchilla / LLaMA |

→ Deep dive: [phase2/README.md](phase2/README.md)

### Phase 3 — Production-Scale Code Pretraining

Scales to ~**97M parameters** and Python code. Built on the Phase 2
architecture with one addition: **gradient checkpointing** (Chen et al.
2016). Trades ~30 % compute for ~4–5× less peak activation memory —
essential at this scale on an 8 GB GPU. The flag is off by default
because the current dims fit unaided; turn it on with `--gradient_checkpointing`.

**Model config (v2):** `d_model=768, n_heads=12, n_layers=12, max_len=512`
→ ~97M params with tied embeddings.

**Data (`data_prep.py`):** Streams `bigcode/the-stack-dedup` Python until
a configurable `--target-tokens` budget (default 2B). Stages docs to
`docs.jsonl` on disk so the corpus never lives in RAM. Trains a 16k
byte-level BPE on a 200k-doc sample, then writes `train.bin` / `val.bin`
as raw `uint16` arrays (nanoGPT pattern) for zero-copy memory-mapped
loading.

**Training (`pretrain.py`):**

| Setting | v2 (current) | v1 (preserved) |
|---------|--------------|----------------|
| micro_batch / accum | 4 / 16 → effective 64 → 32 k tok/step | same |
| total_steps | 60 000 (~1.97 B tokens) | 20 000 (~640 M) |
| warmup | 2 000 | 1 000 |
| LR | 3e-4 → 3e-5 cosine | same |
| weight_decay | 0.1, selective (no decay on norms/embeds) | same |
| params | ~97 M | 46.7 M |
| corpus | the-stack-dedup Python, ~2 B unique | the-stack-smol, ~22 M unique |

**Why v2 exists:** the v1 run produced clear overfitting (train PPL 1.34
vs val PPL 68 at step 20 000) because the token/parameter ratio was ~40×
below Chinchilla's lower bound. v2 fixes both axes — more data, larger
model — within an ~40–50 h budget on the 3060 Ti.

Key engineering: `np.memmap` data loading (zero RAM), random window
sampling (implicit shuffle), fused AdamW, TF32 matmuls on Ampere, atomic
checkpoint saves (tmp → rename), JSONL training log.

→ Deep dive: [phase3/README.md](phase3/README.md)

---

## Setup

```bash
pip install torch datasets tokenizers python-dotenv numpy tqdm huggingface_hub
```

Python 3.10+ required (`X | Y` union type hints).

**HuggingFace auth** — required for gated datasets:

1. Accept dataset terms:
   - [bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol) (for the v1 baseline)
   - [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup) (for v2)
2. Create a Read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Add to `.env` at project root: `HF_TOKEN=hf_your_token_here`

All scripts load `.env` automatically via `python-dotenv`.

---

## Running

All scripts work from the **project root** or from within the phase
directory.

### Phase 0
```bash
python phase0/transformer.py        # smoke test (forward pass on tiny model)
python phase0/train.py              # train (~15 epochs on Multi30k)
```
Checkpoints → `phase0/checkpoints/best_model.pt`

### Phase 1
```bash
python phase1/decoder_only.py       # smoke test
python phase1/train_lm.py           # train (~5 epochs on WikiText-2)
```
BPE tokeniser cached to `phase1/tok_cache/wikitext2_bpe.json`.
Checkpoints → `phase1/checkpoints_lm/best_model.pt`

### Phase 2
```bash
python phase2/decoder_only_v2.py    # smoke test
python phase2/train_lm_v2.py        # train (~5 epochs on WikiText-2)
```
Checkpoints → `phase2/checkpoints_lm_v2/best_model.pt`

### Phase 3
```bash
# Step 1 — data prep (run from phase3/)
cd phase3
python data_prep.py                                  # → pretrain_data_v2/ (~2 B tokens)
python data_prep.py --target-tokens 200_000_000      # smaller smoke run

# Step 2 — pretrain
python pretrain.py                                   # ~40-50 h on RTX 3060 Ti
python pretrain.py --resume                          # resume from last.pt
python pretrain.py --gradient_checkpointing          # OOM rescue (~30 % slower, ~4× less VRAM)
python pretrain.py --micro_batch 2 --total_steps 100 # quick sanity check
```

`pretrain_data_v2/` contains `bpe_tokenizer.json`, `train.bin`, `val.bin`,
`meta.json`, and a kept-by-default `docs.jsonl` staging file.
Checkpoints → `phase3/checkpoints_pretrain_v2/last.pt` (every 1 000 steps)
and `best.pt` (on best val).
Training log → `phase3/checkpoints_pretrain_v2/log.jsonl`.

> Re-running the **v1 baseline** is still possible from git history:
> `git show <pre-v2-commit>:phase3/data_prep.py` produces the smol-pipeline
> script. The v1 outputs (`pretrain_data/`, `checkpoints_pretrain/`) are
> preserved on disk.

---

## Cross-Phase Imports

Each phase has an `__init__.py`. From the project root:

```python
from phase0.transformer import Transformer, TransformerConfig
from phase1.decoder_only import GPT, GPTConfig
from phase2.decoder_only_v2 import GPTv2, GPTConfigV2
from phase3.decoder_only_v3 import GPTv3, GPTConfigV3

from phase0.train import LabelSmoothingLoss, NoamScheduler
from phase1.train_lm import BlockDataset, collate_blocks, load_wikitext2
```

Phase N reuses Phase (N-1)'s vocabulary-agnostic primitives wherever
possible — only the primitives that change are redefined.

---

## Results

WikiText-2, 5 epochs, ~6.8M params (d_model=256, 6 layers, 8 heads),
block=256. Same data + compute for Phase 1 and Phase 2.

| Phase | Architecture | Val loss | PPL |
|-------|-------------|:--------:|:---:|
| Phase 1 | GPT (Post-LN, sinusoidal, ReLU, Noam) | 4.266 | 71.2 |
| Phase 2 | LLaMA-style (Pre-LN, RMSNorm, RoPE, SwiGLU, cosine) | 4.161 | **64.1** |

Phase 2 achieves **9.9 % lower perplexity** at identical parameter count
and training budget — attributable to Pre-LN stability, RoPE, and SwiGLU
combined.

Phase 3 v1 (46.7M params, 22M-token smol corpus, 20k steps): finished
with train loss 0.29 / val loss 4.16 — clearly overfit, exactly what
motivated the v2 redo.

Phase 3 v2 (97M params, 2B-token Stack corpus, 60k steps): training run
in progress.

---

## Roadmap Beyond Phase 3

| Phase | Goal | Status |
|-------|------|--------|
| 3 (fine-tune) | SFT on LeetCode + USACO + Codeforces with CoT traces (GPT-4o distillation where missing) | not started |
| 4 (RL) | GRPO with code-execution test-case reward (DeepSeek-R1 recipe) | not started |
| 5 (eval) | pass@1 / pass@k on held-out LeetCode + HumanEval | not started |

None of these require architecture changes — they layer on top of the
pretrained checkpoint.

---

## References

- Vaswani et al. 2017 — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Radford et al. 2019 — [GPT-2](https://openai.com/research/better-language-models)
- Brown et al. 2020 — [GPT-3](https://arxiv.org/abs/2005.14165)
- Xiong et al. 2020 — [On Layer Normalization in Transformers](https://arxiv.org/abs/2002.04745)
- Zhang & Sennrich 2019 — [RMSNorm](https://arxiv.org/abs/1910.07467)
- Su et al. 2021 — [RoPE / RoFormer](https://arxiv.org/abs/2104.09864)
- Shazeer 2020 — [SwiGLU](https://arxiv.org/abs/2002.05202)
- Dao et al. 2022 — [FlashAttention](https://arxiv.org/abs/2205.14135)
- Touvron et al. 2023 — [LLaMA](https://arxiv.org/abs/2302.13971)
- Chen et al. 2016 — [Gradient Checkpointing](https://arxiv.org/abs/1604.06174)
- Hoffmann et al. 2022 — [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)
- Kocetkov et al. 2022 — [The Stack](https://arxiv.org/abs/2211.15533)
- Sennrich et al. 2016 — [BPE](https://arxiv.org/abs/1508.07909)
- Chen et al. 2021 — [HumanEval](https://arxiv.org/abs/2107.03374)
- Shao et al. 2024 — [GRPO](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI 2025 — [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
