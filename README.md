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
├── phase3/                       Code pretraining + SFT
│   ├── README.md
│   ├── decoder_only_v3.py        GPTv3 (Phase 2 + gradient checkpointing)
│   ├── data_prep.py              The Stack Python → train.bin / val.bin
│   ├── pretrain.py               Full pretraining loop with resume
│   ├── finetune_data_prep.py     codeforces-cots → SFT JSONL
│   ├── finetune.py               Masked-loss supervised fine-tune
│   └── checkpoints_finetune_v2/  SFT checkpoints (Phase 4 starts here)
└── phase4/                       GRPO reinforcement learning
    ├── README.md                 Includes the reward-hacking story
    ├── audit_tests.py            Test-coverage audit / problem pool
    ├── code_executor.py          Sandbox runner + verifiable reward
    ├── rollouts.py               generate_group (the GRPO group)
    ├── build_curriculum.py       Find the learnable problem subset
    ├── grpo.py                   The GRPO training loop
    ├── compare_grpo.py           Pre-vs-post qualitative side-by-side
    ├── curriculum.jsonl          Pool scored under raw reward (200 SOLVABLE)
    ├── curriculum_v2.jsonl       Pool re-scored, anti-hack (24 SOLVABLE)
    ├── checkpoints_grpo_v1/      First run — reward hack baked in
    └── checkpoints_grpo_v2/      Re-calibration under honest reward
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
| **Context**      | 200               | 256                   | 256                       | **2048**                 |
| **Memory tricks** | —                | —                     | —                         | Grad checkpointing (on, required at 2048) |
| **Data loading** | DataLoader        | DataLoader            | DataLoader                | `np.memmap` + random window |
| **Params**       | ~8M               | ~6.8M                 | ~6.8M                     | ~97M                     |

For a side-by-side comparison of the Phase 3 pretrain and fine-tune
recipes, see [Appendix B](#appendix-b--pretrain-vs-fine-tune-recipe-comparison).

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

### Phase 3 — Production-Scale Code Pretraining + Fine-Tune

Scales to ~**97M parameters**, **2048 context**, Python code. Built on
the Phase 2 architecture with one addition: **gradient checkpointing**
(Chen et al. 2016). Trades ~30 % compute for ~4–5× less peak activation
memory — required at 2048 context on an 8 GB GPU.

**Model config:** `d_model=768, n_heads=12, n_layers=12, max_len=2048`
→ ~97M params with tied embeddings.

#### Pretraining

**Data (`data_prep.py`):** Streams `bigcode/the-stack-dedup` Python until
a configurable `--target-tokens` budget (default 2B). Stages docs to
`docs.jsonl` on disk so the corpus never lives in RAM. Trains a 16k
byte-level BPE on a 200k-doc sample, then writes `train.bin` / `val.bin`
as raw `uint16` arrays (nanoGPT pattern) for zero-copy memory-mapped
loading. Verified actual yield: **2.20 B train tokens**, 10.8 M val
tokens.

**Training (`pretrain.py`) — final from-scratch 2048-context recipe:**

| Setting | Value |
|---------|-------|
| context (`max_len` / `block_size`) | 2048 |
| micro_batch · accum → effective | 1 · 64 = **64** (131 k tokens / step) |
| total_steps | 60 000 → ~7.87 B tokens seen |
| epochs over corpus | ~3.6 (over 2.20 B train tokens) |
| warmup | 2 000 |
| LR | 3e-4 → 3e-5 cosine |
| weight_decay | 0.1, selective (no decay on norms / embeds) |
| gradient checkpointing | **on** (required at 2048) |
| params | ~97 M |
| corpus | the-stack-dedup Python, ~2.2 B unique |

The 512-context v2 attempt was abandoned in favour of this 2048 redo so
the fine-tune source's typical prompt + response (p90 ≈ 2420 tokens)
fits. The 46.7M smol-baseline v1 run is preserved on disk for the
overfitting lesson.

Key engineering: `np.memmap` data loading (zero RAM), random window
sampling (implicit shuffle), fused AdamW, TF32 matmuls on Ampere, atomic
checkpoint saves (tmp → rename), JSONL training log, `--force` guard so a
from-scratch run can't silently clobber an existing `last.pt`.

#### Supervised fine-tune

**Data (`finetune_data_prep.py`):** Source is `open-r1/codeforces-cots`,
config `solutions_py_decontaminated`. Three substantive choices, all
forced by inspection of the actual data (see [Lessons](#lessons--schemalength-surprises-caught-by-inspection)):

- **Reasoning = the dataset's `editorial` field** (concise, human-written),
  NOT the model-generated `<think>...</think>` trace. R1's `<think>`
  blocks have a ~15 k-token median — far past 2048.
- **Problem = `description` + `input_format` + `output_format`** joined
  with blank lines. Drops the Codeforces flavor narrative in
  `messages[0].content` that would otherwise blow the budget.
- **Code** is still parsed from the first ```` ```python ```` block
  after `</think>` in the assistant output. `</think>` is used only as
  a positional marker for the code search — we don't extract anything
  between the `<think>` tags.

`finish_reason="length"` is **counted but not filtered** — editorial is
self-contained, so a length-truncated R1 generation with an intact code
block is still usable.

Yield after inspection (`finetune_data_v2/meta.json`):

| Setting | Value |
|---------|-------|
| Source | `open-r1/codeforces-cots:solutions_py_decontaminated` |
| Examples | **3 519 train / 185 val** (val_ratio = 0.05) |
| Context | 2048 |
| Prompt / response boundary | after opening `<reasoning>\n` tag |
| Loss | masked cross-entropy (response tokens only) |
| LR | 2e-5 → 2e-6 cosine |
| Warmup | 3 % of total steps |
| Schedule | **2 epochs** (small set → overfitting risk) |
| micro_batch · accum → effective | 2 · 16 = 32 |
| Gradient checkpointing | off by default (toggle if OOM at 2048) |

A standalone verification script (`verify_finetune_data.py`) checks two
invariants over every record — `len(input_ids) == len(loss_mask)` and
"single 0→1 flip" — and prints decoded previews so the masked region
can be eyeballed before training starts.

**LeetCode (`greengerong/leetcode`) is NOT a training source** —
it has no reasoning traces. Reserved for Phase 5 evaluation.

→ Deep dive: [phase3/README.md](phase3/README.md)

### Phase 4 — GRPO Reinforcement Learning

Applies **GRPO** (Shao et al. 2024 / DeepSeek-R1) with a **verifiable
code-execution reward** on top of the SFT checkpoint: roll out G solution
attempts per problem, run them against the dataset's test cases, and push
the policy toward correctness. Two resident models (trainable policy +
frozen reference for the KL term) — no PPO critic, the key 8 GB win.

**The headline result is a negative one, and it's the most instructive
finding in the project.** The first run looked like a textbook success —
held-out reward climbed **0.022 → 0.238**. But reading the actual generated
code revealed it was almost entirely **reward hacking**: the model learned
to print constant outputs (`-1`, `0`) on problems whose test set is
dominated by one answer, passing many tests without solving anything.

Building a hack-resistant reward — `advantage = max(0, pass_fraction −
constant_baseline)` plus a guard zeroing constant-output completions —
and re-scoring the problem pool, the genuinely-learnable rate collapsed
**25.4% → 1.8%**. A clean re-calibration then held flat at ~0.01: **a 97M
model has essentially no real RL headroom on this data once the exploit is
blocked.**

The GRPO machinery, reward design, curriculum construction, anti-hack
guard, and held-out methodology all work correctly — the limiting factor
is model scale × dataset difficulty, not the RL implementation.

| Run | Reward | Held-out | Reading |
|-----|--------|----------|---------|
| v1 (1500 steps) | raw pass fraction | 0.022 → 0.238 | mostly reward hacking |
| v2 (200 steps) | advantage (anti-hack) | 0.010 → 0.010 | hack blocked → ~no real headroom |

→ Deep dive: [phase4/README.md](phase4/README.md)

---

## Setup

```bash
pip install torch datasets tokenizers python-dotenv numpy tqdm huggingface_hub
```

Python 3.10+ required (`X | Y` union type hints).

**HuggingFace auth** — required for gated datasets:

1. Accept dataset terms:
   - [bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol) (for the v1 baseline)
   - [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup) (for v2 pretrain)
   - [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots) (for SFT)
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
# Step 1 — pretrain data prep (run from phase3/)
cd phase3
python data_prep.py                                  # → pretrain_data_v2/ (~2 B tokens)
python data_prep.py --target-tokens 200_000_000      # smaller smoke run

# Step 2 — pretrain (97M params, 2048 context, ~7.87 B tokens seen)
python pretrain.py                                   # several days on RTX 3060 Ti
python pretrain.py --resume                          # resume from last.pt
python pretrain.py --force                           # from-scratch even though last.pt exists
python pretrain.py --micro_batch 1 --total_steps 100 # quick sanity check

# Step 3 — SFT data prep (CPU-only, safe to run while pretrain still going)
python finetune_data_prep.py                         # → finetune_data_v2/ (~3.5 k train examples)
python verify_finetune_data.py                       # sanity-check loss mask + decode previews

# Step 4 — supervised fine-tune (uses pretrain best.pt)
python finetune.py                                   # 2 epochs, ~minutes-to-hours
python finetune.py --resume                          # resume from FT last.pt
python finetune.py --gradient_checkpointing          # OOM rescue
```

`pretrain_data_v2/` contains `bpe_tokenizer.json`, `train.bin`, `val.bin`,
`meta.json`, and a kept-by-default `docs.jsonl` staging file.
Pretrain checkpoints → `phase3/checkpoints_pretrain_v2/last.pt` (every
1 000 steps) and `best.pt` (on best val).
SFT checkpoints → `phase3/checkpoints_finetune_v2/{last,best}.pt`.
Both runs emit `log.jsonl` alongside their checkpoints.

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

**Phase 3 v2 — pretraining (final).** 97M params, 2048 context, 2.20 B
train tokens, 60 000 steps (~7.87 B tokens seen, ~3.6 epochs). Final:

| Step | Train loss | Val loss |
|-----:|-----------:|---------:|
| 60 000 | **1.13** | **1.15** |

Train↔val gap of ~0.02 nats throughout — Chinchilla-class training paid
off; no overfitting. Throughput ~22.5 k tok/s on the 3060 Ti.

**Phase 3 v2 — supervised fine-tune (final).** 3 519 train / 185 val
examples from `open-r1/codeforces-cots:solutions_py_decontaminated`,
2 epochs, effective batch 64, masked cross-entropy. Final:

| Step | Train loss | Val loss |
|-----:|-----------:|---------:|
| 220 (end of epoch 2) | **1.32** | **1.38** |

Healthy 0.05-nat gap; loss dropped sharply in the first ~40 steps
(model adapting to the tag scaffolding) then plateaued.
[phase3/test_sft_generation.py](phase3/test_sft_generation.py) confirms
the model produces structured `<problem>...<reasoning>...<code>...</code>`
output and stops cleanly at `<eos>` on held-out DSA problems.

---

## Roadmap Beyond Phase 3

| Phase | Goal | Status |
|-------|------|--------|
| 3 (pretrain) | 97M-param decoder-only LM at 2048 context on `the-stack-dedup` Python | **done** (val_loss 1.15 at step 60 000) |
| 3 (fine-tune) | SFT on `open-r1/codeforces-cots:solutions_py_decontaminated`, editorial-as-reasoning (3 519 train / 185 val examples at 2048 context) | **done** (val_loss 1.38 at end of epoch 2; structural smoke test via [phase3/test_sft_generation.py](phase3/test_sft_generation.py)) |
| 4 (RL) | GRPO with code-execution test-case reward (`public_tests`/`private_tests`); two-model policy+reference, anti-hack advantage reward | **done** — see [phase4/README.md](phase4/README.md). Established a documented negative result: v1 gain was reward hacking; 97M has ~no real headroom under an honest reward |
| 5 (eval) | pass@1 / pass@k on held-out LeetCode (`greengerong/leetcode`) + HumanEval | **not started** |

The architecture is frozen from Phase 3 on — Phases 4–5 layer on top of the
pretrained/SFT checkpoint. Phase 5 (formal pass@k benchmark) is the one
remaining piece of the original plan.

---

## Lessons — schema/length surprises caught by inspection

Three assumptions in the original plan turned out to be wrong about the
actual data. All three were caught by **inspecting the dataset before
writing the loader**, not by debugging silent garbage afterward.

1. **`bigcode/the-stack-smol` layout: parquet shards → a single JSON file.**
   The original plan assumed `data/python/*.parquet` shards. The repo
   actually exposes `data/python/data.json` (one file, no shards). The
   fix in [phase3/data_prep.py](phase3/data_prep.py) uses
   `hf_hub_download` → `load_dataset("json", data_files=…)` instead of
   the parquet / script loaders. The v2 corpus moved to
   `bigcode/the-stack-dedup` and avoids the issue entirely.

2. **`open-r1/codeforces` has no reasoning traces.** The original plan
   named it as the SFT source. On inspection the dataset has only
   problems and solutions, no chain-of-thought. The correct source is
   `open-r1/codeforces-cots` config `solutions_py_decontaminated`, which
   has both R1 traces *and* a separate human-written `editorial` field
   per row. [phase3/finetune_data_prep.py](phase3/finetune_data_prep.py)
   is built against that schema.

3. **R1 `<think>` traces median ~15 k tokens → editorial + 2048 redesign.**
   After loading `codeforces-cots`, the assistant outputs' `<think>`
   sections had a ~15 k-token median — orders of magnitude past any
   reasonable context window. Two redesigns followed: (a) use the
   `editorial` field as reasoning instead of `<think>`; (b) bump pretrain
   context from 512 to 2048 and rerun from scratch so the trimmed
   `problem` + `editorial` + `code` typical example (p50 = 1 278 tokens,
   p90 = 2 420 tokens) actually fits. The 512-context partial pretrain
   run was discarded; the [pretrain.py](phase3/pretrain.py) `--force`
   guard protects against accidentally clobbering prior checkpoints
   during the redo.

Generalised versions of these process lessons live in the knowledge
graph: [Schema verification before coding](Learning/knowledge-graph/Schema%20verification%20before%20coding.md),
[Context length from data](Learning/knowledge-graph/Context%20length%20from%20data.md),
[Loss-mask invariants](Learning/knowledge-graph/Loss-mask%20invariants.md).

### Phase 4 lesson — a rising reward is not evidence of learning

The single biggest lesson of Phase 4: **GRPO's training reward climbed 10×
(0.022 → 0.238) almost entirely via reward hacking.** The model learned to
print constant outputs (`-1`, `0`) on problems whose test set is dominated
by one answer — passing many tests without solving anything. The numbers
looked like success; the *code* told the real story.

Three mitigations followed, all now standard in the Phase 4 reward:
1. **Constant baseline subtraction** — `advantage = max(0, pass_fraction −
   (modal-output frequency))`. A constant nets ~0.
2. **Constant-output guard** — any completion emitting one identical output
   across all distinct inputs is zeroed.
3. **Held-out eval under the honest reward** — the only trustworthy signal;
   it stayed flat (~0.01) where the hackable metric rose.

Generalised in the knowledge graph:
[Reward hacking](Learning/knowledge-graph/Reward%20hacking.md),
[Verifiable reward has a baseline](Learning/knowledge-graph/Verifiable%20reward%20has%20a%20baseline.md),
[Read the outputs, not just the metric](Learning/knowledge-graph/Read%20the%20outputs%20not%20just%20the%20metric.md).

---

## Appendix B — Pretrain vs Fine-tune recipe comparison

Side-by-side of the two Phase 3 training stages. Both use the same model
(`GPTv3`, 97M params, 2048 context) and the same `pretrain_data_v2/bpe_tokenizer.json`.

|                                  | Pretrain (`pretrain.py`)                          | Fine-tune (`finetune.py`)                           |
|----------------------------------|---------------------------------------------------|------------------------------------------------------|
| **Source**                       | `bigcode/the-stack-dedup` Python                  | `open-r1/codeforces-cots:solutions_py_decontaminated` |
| **Examples / tokens**            | ~2.20 B train tokens (uint16 memmap)              | 3 519 train / 185 val examples (JSONL)               |
| **Objective**                    | next-token prediction                             | next-token, **response-masked**                      |
| **Loss**                         | plain cross-entropy                               | plain cross-entropy w/ `ignore_index=-100` on prompt |
| **Context** (`max_len`)          | 2048                                              | 2048                                                 |
| **LR (max → min)**               | 3e-4 → 3e-5 cosine                                | 2e-5 → 2e-6 cosine                                   |
| **Warmup**                       | 2 000 steps                                       | **3 %** of total steps                               |
| **Optimiser**                    | AdamW fused, β=(0.9, 0.95), wd=0.1, selective     | AdamW, β=(0.9, 0.95), wd=0.1, selective              |
| **micro_batch · accum → effective** | 1 · 64 = **64**                              | 2 · 16 = **32**                                      |
| **Tokens / step**                | 131 k                                             | varies (padded batches)                              |
| **Schedule**                     | 60 000 steps (~7.87 B tokens, ~3.6 epochs)        | **2 epochs** (overfit risk on the small set)         |
| **Precision**                    | bf16 autocast                                     | bf16 autocast                                        |
| **Gradient checkpointing**       | **on** (required at 2048)                         | off (toggleable; first OOM fallback)                 |
| **Sampler**                      | random window (memmap)                            | shuffled DataLoader, dynamic padding                 |
| **Generation during sampling**   | KV-cache, top-k=40, temperature=0.8               | KV-cache, top-k=40, temperature=0.7                  |
| **Output dir**                   | `checkpoints_pretrain_v2/`                        | `checkpoints_finetune_v2/`                           |

Why the LR drops 15× from pretrain to FT: pretrained weights are already
in a good basin; SFT adapts them rather than searching anew. Standard
practice across LLaMA, Mistral, Qwen fine-tunes.

Why effective batch shrinks (64 → 32): fewer training examples and
shorter run, so each gradient should be more aggressive at finding the
local minimum without averaging out across too many examples.

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
