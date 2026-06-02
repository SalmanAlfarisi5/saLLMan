"""
saLLMan Phase 3 — Pretraining loop for the ~97M-param decoder-only LM.

Trains GPTv3 on the code+DSA corpus prepared by data_prep.py. Designed
for a single 8GB GPU (RTX 3060 Ti) and runs that may span many hours
or multiple sessions.

What changed from the 46.7M baseline
------------------------------------
The previous default produced a clearly-overfit checkpoint (train PPL=1.34
vs val PPL=68 at step 20k). Two root causes: (a) the corpus was ~22M tokens
trained for ~28 epochs, (b) the model was undersized for any sensible
fine-tuning target. v2 addresses both:

  Architecture target:  ~97M params (d_model=768, n_heads=12, n_layers=12,
                        SwiGLU d_ff=2048). Still fits comfortably on 8GB
                        VRAM at micro_batch=4 / seq=512 without gradient
                        checkpointing.
  Data target:          ~2B tokens from bigcode/the-stack-dedup data/python,
                        prepared by the rewritten data_prep.py — close to
                        Chinchilla optimum (20 tokens/param) for this size.
  Schedule:             total_steps 20k → 60k, warmup 1k → 2k.

The architecture file (decoder_only_v3.py) didn't need to change — every
new knob is already exposed by GPTConfigV3.

Features (unchanged from the smol-baseline pipeline)
----------------------------------------------------
1. Memory-mapped binary token files — zero-copy, instant startup, scales to
   billions of tokens with no RAM cost.
2. Random window sampling — each step samples random offsets into the token
   stream. Implicit infinite shuffle. Standard nanoGPT pattern.
3. Gradient accumulation — simulate larger effective batch size than fits
   in VRAM.
4. Gradient checkpointing — toggle when activations don't fit.
5. bf16 mixed precision.
6. Step-level checkpointing with full resume: model, optimizer, scheduler,
   RNG, step.
7. Cosine LR with linear warmup.
8. JSONL training log.

Recipe summary
--------------
  d_model=768, n_heads=12, n_layers=12 → ~97M params
  block_size=2048, micro_batch=1, accum=64 → effective batch=64, 131k tok/step
  max_lr=3e-4, min_lr=3e-5, weight_decay=0.1
  total_steps=60000 → ~7.87B tokens seen (~3.6 epochs of a 2.2B corpus)
  gradient_checkpointing=True (required for 2048 context on 8 GB)

Wall-clock estimate (RTX 3060 Ti)
---------------------------------
At 512 context the 97M model ran ~22k tok/s. The 2048 context move adds:
  - 4x tokens per step from the longer block,
  - ~30% compute overhead from gradient checkpointing,
  - ~4x larger attention matrix (mitigated by FlashAttention's O(n) memory).
Net: expect roughly 5x the 512-context wall-clock per step. 60k steps at
2048 ≈ several days on a 3060 Ti. Use --total_steps to scale down for
shorter experiments.

Dependencies:
    pip install tokenizers numpy

Run:
    python data_prep.py        # one-time, builds pretrain_data_v2/ (~hours)
    python pretrain.py         # the long run
    python pretrain.py --resume    # picks up from last checkpoint
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tokenizers import Tokenizer

# Ensure project root is on sys.path so cross-phase imports work from anywhere.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decoder_only_v3 import GPTv3, GPTConfigV3


# ===========================================================================
# 1. Configuration
# ===========================================================================
# All hyperparameters in one place. Edit the dataclass for different runs.
@dataclass
class TrainConfig:
    # ── Data ─────────────────────────────────────────────────────────────────
    # New directory so the smol baseline (pretrain_data/) remains reproducible.
    data_dir: str = "pretrain_data_v2"

    # ── Model ────────────────────────────────────────────────────────────────
    # ~97M-param recipe. SwiGLU d_ff = ⌈8/3 · d_model⌉₆₄ = 2048 for d_model=768.
    #   embed (tied with lm_head):  16000 · 768            ≈ 12.29M
    #   per block: 4·768²  (attn QKVO)  +  3·768·2048  (SwiGLU)  ≈ 7.08M
    #   12 blocks:                                              ≈ 84.95M
    #   total                                                   ≈ 97.24M
    # Head dim stays 64 (768 / 12) — same as the 46.7M baseline (512 / 8).
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    max_len: int = 2048       # context length during training (bumped from 512)
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    gradient_checkpointing: bool = False   
    # ── Training ─────────────────────────────────────────────────────────────
    block_size: int = 2048
    micro_batch_size: int = 2   
    grad_accum_steps: int = 32  
    total_steps: int = 60_000    
    eval_interval: int = 500
    eval_iters: int = 100        # number of val batches per eval
    log_interval: int = 20
    checkpoint_interval: int = 1000
    sample_interval: int = 1000  # generate samples every N steps

    # ── Optimizer ────────────────────────────────────────────────────────────
    # max_lr kept at 3e-4 — GPT-3-class LR for the ~100M band (Brown et al.
    # 2020 use 6e-4 at 125M; conservative 3e-4 is well within stable range
    # and matches what we used at 46.7M, simplifying comparison).
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000     # scaled 2x with total_steps for stability
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    out_dir: str = "checkpoints_pretrain_v2"
    use_amp: bool = True


# ===========================================================================
# 2. Memory-mapped data loader
# ===========================================================================
# Why memmap rather than load to RAM:
#   - At 1GB+ token files, loading to RAM would fight with PyTorch for memory.
#   - memmap only pages in the slices we actually touch.
#   - Resume across restarts is trivial — just remap the file.
#
# Why random sampling rather than sequential iteration:
#   - We want each (micro_batch * accum) effective batch to see diverse
#     contexts within the corpus. Sequential would make consecutive steps
#     highly correlated.
#   - Implements implicit infinite shuffling without index bookkeeping.
class MemmapTokenDataset:
    """Memory-mapped uint16 token file. Sample random windows of block_size+1."""

    def __init__(self, path: Path, block_size: int) -> None:
        self.path = path
        self.block_size = block_size
        # mode='r' — read-only memmap. dtype must match what data_prep.py wrote.
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.n_tokens = self.data.shape[0]
        # We need block_size+1 contiguous tokens per sample (input + shifted target).
        assert self.n_tokens > block_size + 1, "dataset smaller than block_size"

    def __len__(self) -> int:
        # "Number of distinct starting positions" — large but bounded.
        return self.n_tokens - self.block_size - 1

    def sample_batch(self, batch_size: int, rng: np.random.Generator,
                     device: torch.device) -> tuple[Tensor, Tensor]:
        # Sample `batch_size` random start offsets, each yields a window
        # of length block_size+1.
        starts = rng.integers(0, len(self), size=batch_size)
        # Slice and stack. np.stack is faster than building a list of arrays
        # and then converting in PyTorch. We then convert ONCE.
        chunks = np.stack([
            self.data[s : s + self.block_size + 1].astype(np.int64)
            for s in starts
        ])
        # Move to device with non_blocking transfer (pinned memory possible
        # but adds complexity for marginal benefit at this scale).
        x = torch.from_numpy(chunks[:, :-1]).to(device, non_blocking=True)
        y = torch.from_numpy(chunks[:, 1:]).to(device, non_blocking=True)
        return x, y


# ===========================================================================
# 3. Cosine LR schedule (same as Phase 2, restated for self-containment)
# ===========================================================================
def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * step / max(1, cfg.warmup_steps)
    if step >= cfg.total_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


# ===========================================================================
# 4. Optimizer with selective weight decay (LLaMA recipe)
# ===========================================================================
def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Skip 1D tensors (norms, biases) and embeddings — LLaMA convention.
        if p.dim() < 2 or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    print(f"  optimizer: {sum(p.numel() for p in decay):,} decayed, "
          f"{sum(p.numel() for p in no_decay):,} not decayed")
    # fused=True dispatches to a single CUDA kernel for the full Adam update.
    # Non-trivial speedup on Ampere. Falls back to non-fused on CPU/older GPUs.
    fused_supported = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    extra = {"fused": True} if fused_supported and torch.cuda.is_available() else {}
    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.max_lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8, **extra,
    )


# ===========================================================================
# 5. Cross-entropy loss
# ===========================================================================
# We DROP label smoothing for pretraining. Label smoothing helps in
# bounded-output tasks (translation, classification) but for next-token-prediction
# at scale it slightly hurts perplexity by smearing the target distribution.
# LLaMA, GPT-3 use plain cross-entropy. (We can revisit for fine-tuning.)
def lm_loss(logits: Tensor, targets: Tensor, pad_idx: int = 0) -> Tensor:
    """Standard next-token cross-entropy. ignore_index skips PAD positions
    (block-packed data has no padding, but kept for safety / fine-tuning reuse)."""
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B * T, V),
        targets.reshape(B * T),
        ignore_index=pad_idx,
    )


# ===========================================================================
# 6. Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate(model: GPTv3, val_ds: MemmapTokenDataset, cfg: TrainConfig,
             rng: np.random.Generator, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = val_ds.sample_batch(cfg.micro_batch_size, rng, device)
        with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp,
                                dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = lm_loss(logits, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# ===========================================================================
# 7. Sample generation for visual sanity check
# ===========================================================================
def generate_samples(model: GPTv3, tokenizer: Tokenizer, device: torch.device,
                     prompts: list[str]) -> list[str]:
    model.eval()
    BOS = 1
    EOS = 2
    out = []
    for p in prompts:
        ids = [BOS] + tokenizer.encode(p).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        gen = model.generate(x, max_new_tokens=80, eos_id=EOS,
                              temperature=0.8, top_k=40)
        out.append(tokenizer.decode(gen[0, 1:].tolist()))
    model.train()
    return out


# ===========================================================================
# 8. Checkpointing — save/load full training state
# ===========================================================================
# A complete checkpoint includes everything needed to resume bit-exactly:
#   model weights, optimizer state, current step, training+eval rng state.
# We do NOT save the dataloader position because we sample randomly — there's
# no position. We DO save the rng so future random samples reproduce.
def save_checkpoint(path: Path, model: GPTv3, optimizer: torch.optim.Optimizer,
                    step: int, best_val: float, train_cfg: TrainConfig,
                    model_cfg: GPTConfigV3, np_rng_state: dict,
                    torch_rng_state: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state":     model.state_dict(),
        "optim_state":     optimizer.state_dict(),
        "step":            step,
        "best_val_loss":   best_val,
        "train_cfg":       asdict(train_cfg),
        "model_cfg":       asdict(model_cfg),
        "np_rng_state":    np_rng_state,
        "torch_rng_state": torch_rng_state,
    }
    # Save to a temp file then rename — atomic, can't be corrupted by interrupt.
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model: GPTv3,
                    optimizer: torch.optim.Optimizer) -> dict:
    print(f"Resuming from {path}")
    # weights_only=False because we save Python dicts/dataclasses, not just tensors.
    # Safe because we control checkpoint provenance ourselves.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optim_state"])
    return payload


# ===========================================================================
# 9. Main training loop
# ===========================================================================
def train(cfg: TrainConfig, resume: bool, force: bool = False) -> None:
    # ── Setup ────────────────────────────────────────────────────────────────
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    # Guard: a from-scratch run (resume=False) that lands in a directory
    # which already contains a last.pt would silently clobber that
    # checkpoint on the first save_checkpoint() call. Require --force to
    # acknowledge this. (--resume bypasses the guard by construction.)
    last_ckpt = out_dir / "last.pt"
    if not resume and last_ckpt.exists() and not force:
        raise SystemExit(
            f"\n{last_ckpt} already exists.\n"
            f"This looks like a from-scratch run that would overwrite an "
            f"existing checkpoint.\n"
            f"  - To continue the existing run, pass --resume.\n"
            f"  - To start fresh anyway (and overwrite), pass --force.\n"
            f"  - To preserve the existing checkpoints, move them aside first:\n"
            f"      mv {out_dir} {out_dir}_old"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    # Performance knobs — these are free wins on Ampere.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # use TF32 for fp32 matmuls

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir = Path(cfg.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    vocab_size = meta["vocab_size"]
    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {meta['n_train_tokens']:,}, "
          f"Val tokens: {meta['n_val_tokens']:,}")

    train_ds = MemmapTokenDataset(data_dir / "train.bin", cfg.block_size)
    val_ds   = MemmapTokenDataset(data_dir / "val.bin",   cfg.block_size)
    tokenizer = Tokenizer.from_file(str(data_dir / "bpe_tokenizer.json"))

    # Quick sanity: how many tokens will we see?
    tokens_per_step = cfg.micro_batch_size * cfg.grad_accum_steps * cfg.block_size
    total_train_tokens = tokens_per_step * cfg.total_steps
    epochs_equivalent = total_train_tokens / meta["n_train_tokens"]
    print(f"Tokens/step: {tokens_per_step:,}, "
          f"Total: {total_train_tokens:,} "
          f"(~{epochs_equivalent:.1f} epochs over corpus)")

    # ── Model ────────────────────────────────────────────────────────────────
    model_cfg = GPTConfigV3(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
        pad_idx=meta["pad_idx"],
        rope_base=cfg.rope_base,
        norm_eps=cfg.norm_eps,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    model = GPTv3(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params/1e6:.1f}M, d_ff={model_cfg.d_ff}, "
          f"checkpointing={cfg.gradient_checkpointing}")

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = make_optimizer(model, cfg)

    # ── Resume? ──────────────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")
    last_ckpt = out_dir / "last.pt"
    if resume and last_ckpt.exists():
        payload = load_checkpoint(last_ckpt, model, optimizer)
        start_step    = payload["step"]
        best_val_loss = payload["best_val_loss"]
        np_rng = np.random.default_rng()
        np_rng.bit_generator.state = payload["np_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        print(f"Resumed at step {start_step}, best_val_loss={best_val_loss:.3f}")

    sample_prompts = [
        "def quicksort(arr):",
        "# Problem\nGiven an array of integers, return indices of the two numbers such that they add up to a target.\n\n# Solution\n",
        "def binary_search(arr, target):",
    ]

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    step = start_step
    t_log = time.time()
    tokens_since_log = 0
    loss_sum = 0.0
    loss_count = 0

    while step < cfg.total_steps:
        step += 1

        # Set LR for this step.
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation: grad_accum_steps micro-batches per optimizer step ──
        # We zero grads ONCE before the accumulation loop, then call .backward()
        # on each micro-batch. Gradients accumulate (PyTorch's default behavior).
        # After accum_steps backward calls, we step optimizer once.
        # Loss is scaled by 1/accum so the gradient magnitude matches a single
        # micro_batch * accum_size big batch.
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_ds.sample_batch(cfg.micro_batch_size, np_rng, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp,
                                    dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = lm_loss(logits, y) / cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        # Clip across the FULL accumulated gradient — this is correct because
        # gradients are already summed across micro-batches.
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        loss_sum += accum_loss
        loss_count += 1
        tokens_since_log += tokens_per_step

        # ── Logging ──────────────────────────────────────────────────────────
        if step % cfg.log_interval == 0:
            elapsed = time.time() - t_log
            tok_per_sec = tokens_since_log / max(elapsed, 1e-6)
            avg_loss = loss_sum / loss_count
            ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
            print(f"step {step:6d}/{cfg.total_steps} | "
                  f"loss {avg_loss:.3f} | ppl {ppl:.1f} | "
                  f"lr {lr:.2e} | {tok_per_sec/1e3:.1f}k tok/s")

            with log_path.open("a") as f:
                f.write(json.dumps({
                    "step": step, "train_loss": avg_loss,
                    "lr": lr, "tok_per_sec": tok_per_sec,
                }) + "\n")

            t_log = time.time()
            tokens_since_log = 0
            loss_sum = 0.0
            loss_count = 0

        # ── Eval ─────────────────────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_ds, cfg, np_rng, device)
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"  >> step {step}: val_loss={val_loss:.3f}, val_ppl={val_ppl:.1f}")
            with log_path.open("a") as f:
                f.write(json.dumps({
                    "step": step, "val_loss": val_loss,
                }) + "\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(out_dir / "best.pt", model, optimizer, step,
                                best_val_loss, cfg, model_cfg,
                                np_rng.bit_generator.state, torch.get_rng_state())
                print(f"  >> saved best (val_loss={val_loss:.3f})")

        # ── Sample generation ────────────────────────────────────────────────
        if step % cfg.sample_interval == 0:
            samples = generate_samples(model, tokenizer, device, sample_prompts)
            for p, s in zip(sample_prompts, samples):
                # Truncate display to avoid log spam.
                short_prompt = p[:60].replace("\n", " ⏎ ")
                short_sample = s[:200].replace("\n", " ⏎ ")
                print(f"  PROMPT: {short_prompt!r}")
                print(f"  SAMPLE: {short_sample!r}")

        # ── Periodic checkpoint (recoverable on crash) ───────────────────────
        if step % cfg.checkpoint_interval == 0:
            save_checkpoint(out_dir / "last.pt", model, optimizer, step,
                            best_val_loss, cfg, model_cfg,
                            np_rng.bit_generator.state, torch.get_rng_state())
            print(f"  >> saved last.pt at step {step}")

    # Final checkpoint at end of training.
    save_checkpoint(out_dir / "last.pt", model, optimizer, step,
                    best_val_loss, cfg, model_cfg,
                    np_rng.bit_generator.state, torch.get_rng_state())
    print(f"\nTraining complete at step {step}. "
          f"Best val loss: {best_val_loss:.3f} "
          f"(ppl {math.exp(best_val_loss):.1f}).")


# ===========================================================================
# 10. Entry point
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints_pretrain_v2/last.pt")
    parser.add_argument("--force", action="store_true",
                        help="Allow a from-scratch run to overwrite an existing "
                             "last.pt in out_dir. Required when starting fresh "
                             "in a directory that already contains a checkpoint.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (slower, less VRAM)")
    parser.add_argument("--micro_batch", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.gradient_checkpointing:
        cfg.gradient_checkpointing = True
    if args.micro_batch is not None:
        cfg.micro_batch_size = args.micro_batch
    if args.grad_accum is not None:
        cfg.grad_accum_steps = args.grad_accum
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps

    train(cfg, resume=args.resume, force=args.force)


if __name__ == "__main__":
    main()
