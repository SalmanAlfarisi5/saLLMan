# Phase 3 — Production-Scale Code Pretraining

Scales from a 6.8M-param toy LM (Phases 1–2) to a real pretraining run on
Python code, on an 8 GB GPU. The architecture file ([decoder_only_v3.py](decoder_only_v3.py))
inherits the [Phase 2](../phase2/) LLaMA-class block unchanged and adds **one**
thing: gradient checkpointing. Everything else — bigger model, more data,
proper resume support — happens in [data_prep.py](data_prep.py) and
[pretrain.py](pretrain.py).

References:
- Chen et al. 2016, [Gradient checkpointing](https://arxiv.org/abs/1604.06174)
- Karpathy [nanoGPT](https://github.com/karpathy/nanoGPT) — memory-mapped uint16 pattern
- Hoffmann et al. 2022, [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)
- Kocetkov et al. 2022, [The Stack](https://arxiv.org/abs/2211.15533) /
  Lozhkov et al. 2024, [the-stack-v2 + dedup](https://arxiv.org/abs/2402.19173)

---

## What changed: v1 → v2

The first Phase 3 pretrain (46.7M params, `the-stack-smol` ~22M tokens,
20k steps) overfit badly: train PPL 1.34 vs val PPL 68. Root cause: ~28
epochs over a tiny corpus on an under-capacity model. The current files
are the v2 redo:

| | v1 (preserved in `pretrain_data/` + `checkpoints_pretrain/`) | v2 (current code) |
|---|---|---|
| Source | `bigcode/the-stack-smol` Python — 10k files | `bigcode/the-stack-dedup` Python, streamed |
| Token budget | ~22M unique → 640M seen (~28 epochs) | ~2B unique → ~2B seen (~1 epoch) |
| Params | 46.7M (d=512, n=8, L=12) | ~97M (d=768, n=12, L=12) |
| Steps | 20k | 60k |
| Warmup | 1k | 2k |

Token-to-parameter ratio went from ~0.5 to ~20 — Chinchilla guideline
satisfied for the first time.

---

## Files

### Pretraining
| File | Role |
|------|------|
| [decoder_only_v3.py](decoder_only_v3.py) | GPTConfigV3 + GPTv3. Inherits Phase 2 block; adds gradient-checkpointing wrapper around the forward pass. |
| [data_prep.py](data_prep.py) | Stream The Stack Python, train BPE, write `train.bin`/`val.bin`/`bpe_tokenizer.json`/`meta.json`. |
| [pretrain.py](pretrain.py) | Memory-mapped data loader, full training loop with resume, AMP, checkpointing. |
| `pretrain_data/` (legacy) | v1 data outputs — kept for reproducing the smol-baseline run. |
| `pretrain_data_v2/` | v2 data outputs (created by current `data_prep.py`). |
| `checkpoints_pretrain/` (legacy) | v1 model checkpoints (final val_loss 4.16). |
| `checkpoints_pretrain_v2/` | v2 model checkpoints. |

### Fine-tuning (SFT)
| File | Role |
|------|------|
| [finetune_data_prep.py](finetune_data_prep.py) | Build the SFT corpus from `open-r1/codeforces` + `greengerong/leetcode`; emit `train.jsonl` / `val.jsonl` with input_ids + loss_mask. |
| [finetune.py](finetune.py) | Load the pretrained checkpoint, run masked-loss SFT with dynamic-padded batches, save best/last. |
| `finetune_data_v2/` | SFT data outputs (created by `finetune_data_prep.py`). |
| `checkpoints_finetune_v2/` | SFT model checkpoints. |

---

## [decoder_only_v3.py](decoder_only_v3.py) — the model

### `GPTConfigV3` (dataclass)
Inherits `GPTConfigV2` and adds:
```python
gradient_checkpointing: bool = False
```
That's the entire schema diff. Every dim, RoPE base, dropout, norm
epsilon — all from `GPTConfigV2`.

### `GPTv3` model
Same modules as `GPTv2` (`embed`, `n_layers` `TransformerBlock`s, `final_norm`,
`lm_head`, RoPE cos/sin buffers). The forward pass adds three lines:

```python
use_ckpt = (self.cfg.gradient_checkpointing
            and self.training
            and past_kvs is None)
```

Gradient checkpointing is enabled only when **all three** hold:

1. **Config says so** — opt-in flag.
2. **`self.training`** — checkpointing is a backward-pass trick. In eval
   we're in `no_grad` and there are no activations to save.
3. **No KV cache** — caching plus checkpointing don't compose. The
   re-computation during backward would re-rotate K/V at wrong positions
   if the cache is in play. So checkpointing turns off automatically
   during generation.

When `use_ckpt=True`, each block is wrapped with `torch.utils.checkpoint.checkpoint`:
```python
def run(x_, cos_, sin_):
    out, _ = blk(x_, cos_, sin_, past_kv=None)
    return out
x = checkpoint(run, x, cos, sin, use_reentrant=False)
```

`use_reentrant=False` is the modern (PyTorch 2.x) setting — works
correctly with autograd graph tracking and avoids the legacy reentrant
pitfalls.

### Why gradient checkpointing matters for 8 GB
Activation memory scales as `O(n_layers · batch · seq_len · d_model)`.
For the 97M v2 model at `batch=4, seq=512, d_model=768, n_layers=12`,
storing all activations costs roughly:
```
12 · 4 · 512 · 768 · 6_factors · 2_bytes(bf16) ≈ 230 MB just for activations
```
manageable. But scaling to `d_model=1024` or `seq=2048` quickly pushes past
6 GB. Checkpointing stores only the **input** to each block and re-computes
intermediates during backward: ~30% extra compute for ~4–5× less activation
memory.

Default is `False` because the v2 config fits comfortably without it.
Use the `--gradient_checkpointing` CLI flag when scaling further.

---

## [data_prep.py](data_prep.py) — the data pipeline

Goal: turn `bigcode/the-stack-dedup` Python into four files
(`bpe_tokenizer.json`, `train.bin`, `val.bin`, `meta.json`) that
`pretrain.py` consumes via `np.memmap`.

The whole script is **streaming-friendly**: it never holds the full
corpus in RAM. The pipeline is five passes over progressively reduced
data:

```
HuggingFace stream → docs.jsonl (disk)
                       │
                       ├─→ BPE trained on a sample (≤200k docs)
                       │
                       └─→ tokenise + pack → train.bin / val.bin
```

### §1 `stage_corpus(staging_path, target_tokens)`
Streams `load_dataset(HF_REPO, data_dir=HF_DATA_DIR, split="train", streaming=True)`
and appends each filtered doc to `docs.jsonl` until total chars hit
`target_tokens · 3.8` (rough chars-per-token ratio for Python BPE@16k).

Filters:
- `MIN_CHARS=64` — drop near-empty stubs.
- `MAX_CHARS=100_000` — drop generated files / huge notebooks. These
  would dominate batches and rarely contain useful patterns.

If `staging_path` already exists, the function counts what's there and
returns — makes the script **resumable**. Delete the file to force a
re-stream.

`HF_TOKEN` is loaded from `.env` via `python-dotenv` and passed to
`load_dataset` for the gated dataset.

### §2 `train_bpe(staging_path, vocab_size, n_total_docs)`
Trains a 16k-vocab byte-level BPE on at most `BPE_TRAIN_MAX_DOCS=200_000`
sampled docs (~600 MB of code) — enough for a converged vocabulary
without holding the full 8 GB corpus live in the trainer.

Critical settings:
- `BPE(unk_token="<unk>")` + `ByteLevel(add_prefix_space=False)`: byte-level
  is essential for code (any character is potentially meaningful), and
  `add_prefix_space=False` keeps Python indentation literal.
- `special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]`: BpeTrainer assigns
  IDs in registration order. We **verify after training** that they came
  out as 0..3, because every downstream component depends on it
  (PAD/BOS/EOS/UNK constants).
- `initial_alphabet=ByteLevel.alphabet()`: seeds all 256 byte chars so
  the tokenizer has a closed vocabulary even on bytes that never appeared
  in the training sample.

### §3 `tokenise_and_pack(...)`
Streams `docs.jsonl` again, encodes in chunks of 1024 docs via
`tokenizer.encode_batch(...)` (the fast Rust path), appends `<eos>` after
each doc, and writes raw uint16 bytes into `train.bin` or `val.bin`.

Train/val routing: hash-shuffle doc indices up-front with a fixed seed,
take first `val_ratio · n_docs` for val. This is **deterministic** given
the seed and lets us write incrementally without buffering the whole
output.

Memory bound: peak is one chunk's worth of encoded ids (~10 MB),
regardless of corpus size.

**uint16 safety check**: asserts `vocab_size ≤ 65535`. If you ever bump
vocab past 64k, you must switch the dtype here AND in pretrain.py's
`np.memmap` call — silent overflow into id 0 (PAD) would be a particularly
nasty bug to debug.

### §4 `write_meta(...)`
Writes `meta.json` with: vocab size, train/val token counts, special-token
ids, source list. `pretrain.py` reads this to know vocab and pad_idx
without having to load the tokenizer.

### §5 `verify_outputs(...)`
Memory-maps `train.bin`, decodes the first 100 tokens, prints both raw
ids and decoded text. Cheap sanity check that the file is well-formed
and the tokenizer round-trips.

### Resumability properties
- If `docs.jsonl` exists, `stage_corpus` skips re-streaming.
- BPE training is fast (~5 min for 200k docs); not separately cached.
- Tokenisation runs from disk and is deterministic given the seed.

So a typical re-run pattern is: edit a hyperparameter (e.g., vocab size)
and re-run; the slow streaming step is skipped.

---

## [pretrain.py](pretrain.py) — the training loop

The longest file in the project. Walks through 10 sections, all
self-contained.

### §1 `TrainConfig` (dataclass)
All hyperparameters. Current v2 defaults:
```
data_dir = "pretrain_data_v2"
d_model=768, n_heads=12, n_layers=12, max_len=512  → ~97M params
block_size=512, micro_batch=4, grad_accum=16       → 32k tokens/step
total_steps=60_000, warmup_steps=2_000             → ~1.97B tokens seen
max_lr=3e-4, min_lr=3e-5, weight_decay=0.1
gradient_checkpointing=False  (turn on if OOM)
out_dir="checkpoints_pretrain_v2"
```

Effective batch = `micro_batch · grad_accum = 64`. Each optimiser step
consumes `64 · 512 = 32_768` tokens. Over 60k steps that's ~1.97B tokens.

### §2 `MemmapTokenDataset`
The key data-loading idea of the phase. `np.memmap("train.bin", dtype=np.uint16, mode="r")`
returns a file-backed array. The OS pages in **only the slices we
actually touch**, so:
- Startup is instant (no upfront read).
- Memory footprint stays ~0 RAM regardless of train.bin size.
- Works for billions of tokens / multi-GB files.

`sample_batch(batch_size, rng, device)`:
1. Sample `batch_size` random start offsets in `[0, n_tokens - block_size - 1)`.
2. Slice and stack into `(B, block_size+1)`.
3. Split into `x = chunk[:, :-1]` and `y = chunk[:, 1:]`.
4. Transfer to device with `non_blocking=True`.

Random window sampling (vs sequential iteration) gives an implicit
infinite shuffle — consecutive steps see uncorrelated contexts. Standard
nanoGPT pattern.

### §3 `get_lr(step, cfg)` — cosine warmup
```
step < warmup:    lr = max_lr · step / warmup
step ≥ total:     lr = min_lr
otherwise:        lr = min_lr + 0.5·(max_lr-min_lr)·(1 + cos(π·progress))
                       progress = (step - warmup) / (total - warmup)
```
Identical to Phase 2's `CosineWarmupScheduler`, inlined as a function
because it has zero state — we drive `optimizer.param_groups[k]["lr"]`
directly each step.

### §4 `make_optimizer(model, cfg)` — selective weight decay
Same LLaMA recipe as Phase 2: norms, biases, and embeddings go into a
no-decay group; everything else gets `weight_decay=0.1`. Uses fused
AdamW (`fused=True`) on CUDA — a single CUDA kernel for the entire
Adam update, non-trivial speedup on Ampere.

### §5 `lm_loss(logits, targets, pad_idx)`
Plain cross-entropy with `ignore_index=pad_idx`. We **drop label
smoothing** here (Phases 0/1/2 used ε=0.1):

> Label smoothing helps in bounded-output tasks (translation, classification)
> but for next-token prediction at scale it slightly hurts perplexity by
> smearing the target distribution. LLaMA, GPT-3 use plain cross-entropy.

Block-packed binary data has no padding tokens, but `ignore_index=PAD_IDX`
is kept for safety and so the same loss function survives the move to
fine-tuning.

### §6 `evaluate(model, val_ds, cfg, rng, device)`
Samples `cfg.eval_iters` (=100) batches from `val_ds`, computes per-batch
loss under bf16 autocast, returns mean. Restores `model.train()` mode
on exit.

### §7 `generate_samples(model, tokenizer, device, prompts)`
Runs three fixed prompts (`def quicksort(arr):`, a two-sum problem
prompt, `def binary_search(...):`) with `max_new_tokens=80, temperature=0.8,
top_k=40`. Used as a visual sanity check between val runs — if val loss
is dropping but samples look like noise, something is wrong.

### §8 `save_checkpoint / load_checkpoint`

A complete checkpoint includes:
- `model_state` — `model.state_dict()`
- `optim_state` — `optimizer.state_dict()` (includes per-param Adam m, v)
- `step` — the global step counter
- `best_val_loss` — for the best.pt save-on-improvement logic
- `train_cfg` / `model_cfg` — as dicts (via `asdict`)
- `np_rng_state` — numpy RNG state dict
- `torch_rng_state` — torch RNG state tensor

We do **not** save the dataloader position because the dataset is sampled
randomly — there's no position. We DO save the RNG state so future random
samples reproduce bit-exactly.

Atomic save: write to `<path>.tmp`, then `os.replace(tmp, path)`. POSIX
`rename` is atomic so a crash during write leaves the previous
checkpoint intact.

Load: `weights_only=False` because the payload contains Python dicts and
dataclasses, not just tensors. Safe here because we control checkpoint
provenance ourselves (no untrusted files).

### §9 `train(cfg, resume)` — main loop

Setup steps:
1. Device, seeds, TF32 enable for fp32 matmuls on Ampere.
2. Load `meta.json`, instantiate train/val `MemmapTokenDataset`s, load
   tokenizer.
3. Print "tokens/step" and an epochs-equivalent estimate — quick eyeball
   on whether the budget makes sense.
4. Build `GPTv3` on device, build optimiser.
5. If `--resume` and `last.pt` exists: load model, optimiser, step, RNG.

Step loop:
```
while step < total_steps:
    step += 1
    set lr from cosine schedule
    optimizer.zero_grad(set_to_none=True)
    for _ in range(grad_accum_steps):
        x, y = sample batch
        with bf16 autocast:
            logits, _ = model(x)
            loss = cross_entropy(logits, y) / grad_accum_steps
        loss.backward()                       # gradients accumulate
    clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    if step % log_interval == 0:    log train loss / ppl / tok/s
    if step % eval_interval == 0:   val loss; save best.pt on improvement
    if step % sample_interval == 0: generate samples
    if step % checkpoint_interval:  save last.pt
```

Three details worth flagging:

**Gradient accumulation arithmetic**: loss is divided by `grad_accum_steps`
inside the inner loop. Each `.backward()` adds to `.grad`. After
`grad_accum_steps` backward calls, the accumulated gradient equals the
gradient of `(1/N) · Σ loss_i`, which matches what you'd get from one
big-batch step. Without the division you'd be implicitly multiplying the
LR by `grad_accum_steps`.

**`set_to_none=True`**: `optimizer.zero_grad(set_to_none=True)` literally
sets `.grad = None` rather than zeroing the tensor. Saves a small kernel
call per step and triggers re-allocation on next `.backward()` — which
is what we want with mixed precision.

**`clip_grad_norm_` outside the accumulation loop**: we clip the **full
accumulated gradient**, not each micro-batch's. Correct because gradients
are already summed across micro-batches by the time we clip.

### §10 CLI (`main()`)
Three CLI overrides:
- `--resume` — pick up from `last.pt`.
- `--gradient_checkpointing` — enable activation recomputation.
- `--micro_batch N`, `--grad_accum N`, `--total_steps N` — quick
  knob-tweaks without editing `TrainConfig`.

---

## Running

```bash
# One-time data prep (takes hours — streaming + tokenising 2B tokens)
cd phase3
python data_prep.py                       # → pretrain_data_v2/
python data_prep.py --target-tokens 200_000_000   # smoke run with 200M tokens

# Pretrain
python pretrain.py                        # ~40–50 h on RTX 3060 Ti
python pretrain.py --resume               # resume from last.pt
python pretrain.py --gradient_checkpointing   # OOM rescue
python pretrain.py --micro_batch 2 --total_steps 100   # smoke run
```

Checkpoints go to `checkpoints_pretrain_v2/`:
- `last.pt` — every `checkpoint_interval=1000` steps; also at end of training.
- `best.pt` — overwritten on every val-loss improvement.
- `log.jsonl` — train loss, val loss, tok/s, every log_interval steps.

---

## Diagnostics: what "healthy" looks like

The v1 run overfit. After v2's data fix, healthy training should show:

- **Train and val loss tracking within ~0.5** through most of training.
- Val loss **still falling** at the end (not bottomed out / rising).
- Generated samples increasingly Python-shaped at step ~1k, increasingly
  syntactically valid by ~10k, increasingly *semantically* sensible by
  ~30k.

If you see val loss flat or rising while train falls, you're back to the
v1 failure mode — either the corpus is too small or the model is too big
for it. Either grow `--target-tokens` in data_prep or shrink `d_model`.

---

---

## Supervised fine-tuning (SFT)

The pretrained checkpoint speaks Python but not "given a problem, produce
reasoning + code". SFT teaches that distribution. Two new files:

### [finetune_data_prep.py](finetune_data_prep.py)

Builds an instruction-tuning corpus from datasets that **already** contain
chain-of-thought reasoning traces (no GPT-4o distillation needed for the
first pass):

- `open-r1/codeforces` — Codeforces problems with DeepSeek-R1 reasoning.
- `greengerong/leetcode` — LeetCode problems. Used only for entries
  where a reasoning/editorial-style section is detectable; otherwise
  skipped (per the "only datasets with reasoning" preference).

**Output format**: each example is JSONL,
`{"input_ids": [int, ...], "loss_mask": [0/1, ...]}`. The `loss_mask` is
0 on prompt tokens and 1 on response tokens. At training time, label
positions with `loss_mask=0` are replaced with `-100` so
`F.cross_entropy(ignore_index=-100)` skips them — i.e., the model is
graded only on what it had to *generate*, not on copying the problem.

**Prompt/response shape** (tagged sections — no new special tokens needed):
```
<problem>
{problem text}
</problem>
<reasoning>          ← prompt ends here, response begins
{chain-of-thought}
</reasoning>
<code>
{solution}
</code><eos>
```
Putting the opening `<reasoning>` tag in the **prompt** means at inference
time you build the same prefix and the model picks up from there. Loss
gradient flows through everything after that tag, including the closing
`</reasoning>` and the `<code>...</code><eos>` block — that's how it
learns to emit the boundary markers and stop cleanly.

**Source-schema handling**: each dataset's schema is probed at load time
(`extract_open_r1_codeforces`, `extract_greengerong_leetcode`). Multiple
common column names are tried, with clear errors when none match. The
LeetCode loader uses a heuristic split on the first fenced code block
to separate prose-reasoning from code; if a row has no detectable prose
it's dropped (when `--require-reasoning` is on, which is the default).

**Why tokenize prompt and response separately**: building the loss mask
by hand is much safer than searching for a delimiter token afterward.
BPE may merge `<reasoning>` differently than `</reasoning>` depending on
neighboring whitespace; separate encodes guarantee bit-exact boundary
alignment.

**Length filtering**: examples whose total token count exceeds
`--max-len` (default 512, the pretrained model's context) are dropped
rather than truncated — a truncated CoT trace is worse than no trace.
A length histogram (`min, p50, p95, max`) is printed for visibility.

**Run**:
```bash
cd phase3
python finetune_data_prep.py                       # → finetune_data_v2/
python finetune_data_prep.py --no-require-reasoning   # relax filter
python finetune_data_prep.py --max-len 1024 \\
                              --out finetune_data_v2_long
```

### [finetune.py](finetune.py) — section-by-section

Loop structure mirrors `pretrain.py` but with three substantive changes:
**masked loss**, **dynamic-padded batches**, and a **fresh optimiser**
(we don't resume pretrain's Adam state).

#### §1 `FTConfig`
SFT-appropriate defaults:
- `max_lr=2e-5, min_lr=2e-6` — one-to-two orders below pretrain. Why:
  pretrained weights are already in a good basin; SFT adapts them rather
  than searching anew. Standard practice across LLaMA, Mistral, Qwen
  fine-tunes.
- `warmup_ratio=0.03` — 3% of total steps. Pre-LN models don't need
  much warmup; this is just to soften the initial step.
- `epochs=3, micro_batch=8, grad_accum=4` → effective batch 32. Small
  enough to step often on a few-thousand-example corpus.
- `weight_decay=0.1` with the same selective recipe as pretrain (norms,
  biases, embeddings exempt).

#### §2 `FinetuneDataset` + `make_collate`
`FinetuneDataset` loads `train.jsonl`/`val.jsonl` into RAM at startup —
SFT corpora are small (~10s of MB after tokenization), no point memmapping.

`make_collate(pad_idx)` returns the collate function that:
1. Pads `input_ids` to the longest in the batch with `pad_idx=0`.
2. Builds the **labels tensor** by shifting `input_ids` by 1 and gating
   on `loss_mask`. Positions where the shifted-mask is 0 → `-100`.
   Pad positions → `-100`.
3. Builds `attn_keep` (1 = real token, 0 = pad). Currently not consumed
   by the model — the causal mask plus `ignore_index=-100` is sufficient
   — but kept for future use (e.g. truncating during KV-cached inference).

**The shift-by-1 detail**: at training position `t`, the model sees
`input_ids[t]` and must predict `input_ids[t+1]`. So `labels[t] = input_ids[t+1]`
*if* `loss_mask[t+1] == 1`, else `IGNORE`. The very last position has no
next-token, so its label stays `IGNORE`.

#### §3 `masked_lm_loss`
Plain `F.cross_entropy` with `ignore_index=-100`. Mean is over non-ignored
positions only — each response token contributes equally regardless of
the prompt length preceding it.

#### §4 `get_lr(step, ...)`
Same cosine-with-warmup function as pretrain, parameterised by
`total_steps` and `warmup_steps`. Both are computed from the dataset
size and `cfg.epochs` — no pre-fixed step budget.

#### §5 `make_optimizer`
Identical to pretrain (LLaMA selective weight decay, fused AdamW on CUDA).

#### §6 `load_pretrained(ckpt_path, ...)`
Three steps:
1. Load the pretrain checkpoint (`weights_only=False` — payload contains
   dicts and dataclasses).
2. Reconstruct `GPTConfigV3` from the saved `model_cfg` dict, allowing
   CLI override of `gradient_checkpointing`.
3. Build a fresh `GPTv3`, load `model_state`. **Do not** load
   `optim_state` — fine-tune starts a new cosine schedule.

Sanity check after load: assert `model_cfg.vocab_size == meta["vocab_size"]`.
If they differ, the pretrain tokenizer and FT data tokenizer aren't the
same file → silent garbage.

#### §7 `evaluate`
Standard token-weighted average loss over the val loader. Properly
accounts for variable-length batches by multiplying loss by token count
each batch and dividing by total tokens at the end (instead of plain
mean-over-batches, which would over-weight short batches).

#### §8 `generate_sample`
Builds the same `<problem>...</problem>\n<reasoning>\n` prefix used during
training and calls `model.generate(...)`. Logged every `sample_interval`
steps so quality is visible without re-running an eval pass.

`temperature=0.7, top_k=40` — modest randomness, prevents the visual
sample from being purely greedy / repetitive.

#### §9 Training loop
Standard `for epoch in epochs: for batch in train_loader:` with manual
gradient accumulation: collect `grad_accum_steps` micro-batches in a
buffer, run forward+backward on each, then a single `optimizer.step()`.

Subtle bit at epoch end: if the loader ran out before filling the accum
buffer, we **still flush** the remaining micro-batches as one final
optimiser step, with the loss scale adjusted to `1/len(buffer)` instead
of `1/grad_accum_steps`. Otherwise the last few micro-batches per epoch
would silently never reach the optimiser.

**Resume semantics** (`--resume`):
- Resume from `<out_dir>/last.pt`, NOT from the pretrain checkpoint.
- Restores model state, optimizer state, step count, epoch, best val,
  torch RNG. The dataloader doesn't have a position concept (we shuffle
  each epoch), so we don't try to resume mid-epoch — we restart the
  current epoch from the top.

#### §10 CLI
- `--data DIR` — override `finetune_data_v2/`.
- `--pretrain PATH` — point at a different pretrain checkpoint.
- `--out DIR` — override `checkpoints_finetune_v2/`.
- `--epochs N`, `--micro_batch N`, `--grad_accum N`, `--lr F` — quick
  knob tweaks.
- `--gradient_checkpointing` — enable if SFT OOMs (shouldn't at micro_batch=8
  + max_len=512 + 97M params, but headroom is good).
- `--resume` — restart from FT `last.pt`.

### Running the SFT pipeline

```bash
# 1. Wait for pretrain to finish (or use an intermediate checkpoint).
# 2. Build the FT corpus (~minutes, no GPU needed):
cd phase3
python finetune_data_prep.py

# 3. Fine-tune (~minutes-to-hours depending on dataset size):
python finetune.py
python finetune.py --resume
python finetune.py --epochs 5 --lr 1e-5            # gentler schedule
python finetune.py --pretrain checkpoints_pretrain_v2/last.pt    # use last instead of best
```

Checkpoints → `checkpoints_finetune_v2/last.pt` and `best.pt`. Training
log → `checkpoints_finetune_v2/log.jsonl` (same JSONL format as pretrain).

### Diagnostics for a healthy SFT

- Initial val loss should sit somewhere in the 1.5–3.0 range (pretrain
  loss was ~1.6 on raw code; SFT data has tagged structure the model
  hasn't seen, so first-step loss is higher).
- Val loss should fall **sharply** in the first ~200 steps (model adapts
  to the tag scaffolding), then more gradually as it learns CoT
  patterns.
- Watch the generated samples after step ~400: should produce
  reasoning-like prose followed by code-shaped output even if the code
  doesn't yet solve the problem. If you see only repeated tokens or pure
  prose, the masking is probably wrong (e.g., `loss_mask` was inverted).

### Known gotchas

- **Tokenizer drift**: `finetune.py` reads `tokenizer_path` from
  `finetune_data_v2/meta.json`, which is the **absolute path written
  at data prep time**. If you move directories, edit meta.json or pass
  `--data` to point at a freshly re-prepped corpus.
- **Loss mask shift**: the boundary is on the *shifted* mask, not the
  raw mask. The label at position `t` corresponds to predicting
  `input_ids[t+1]`, so the gating is `loss_mask[t+1]`. Easy to off-by-one
  if you're modifying the collate.

---

## What comes next (Phase 4 + Phase 5)

- **Phase 4 — GRPO RL** — generate N solution attempts per problem,
  execute against test cases, use pass/fail as reward (DeepSeek-R1 recipe).
  Builds on top of the SFT checkpoint.
- **Phase 5 — Evaluation** — pass@1 / pass@k on held-out LeetCode and
  HumanEval (Chen et al. 2021).

Neither requires architecture changes — both layer on top of the SFT
checkpoint.
