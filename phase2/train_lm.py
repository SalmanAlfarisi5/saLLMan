"""
saLLMan Phase 2 — Training script for the modernized decoder-only LM.

Same data pipeline as train_lm.py (WikiText-2, BPE, block packing) but with:
  - Model: GPTv2 (Pre-LN + RMSNorm + RoPE + SwiGLU + GPT-2 init + SDPA)
  - Optimizer: AdamW with selective weight decay (LLaMA recipe)
  - Cosine LR schedule with warmup (modern default; Pre-LN doesn't need Noam)
  - bf16 mixed precision (Ampere+)
  - KV-cache generation for fast sample logging

Why AdamW over Adam-with-Noam
-----------------------------
Pre-LN architectures (Xiong et al. 2020) train stably without warmup, so the
1/sqrt(step) Noam decay isn't necessary. Cosine annealing with a short linear
warmup is the de-facto standard since GPT-3 and is what LLaMA, Mistral, and
DeepSeek all use.

Why selective weight decay
--------------------------
AdamW's weight decay pulls parameters toward zero. This is good regularization
for matrix weights but ACTIVELY HARMFUL for:
  - LayerNorm/RMSNorm gains (gamma): pulling toward zero kills the layer
  - Biases: same
  - Embedding rows: empirically helps to NOT decay them (LLaMA does this)
So we partition parameters into two groups: decay and no-decay.

Dependencies:
    pip install datasets tokenizers
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

# Ensure the project root is on sys.path so cross-phase imports work whether
# this file is run directly (`python phase2/train_lm.py`) or imported.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse data pipeline from Phase 1 trainer.
from phase1.train_lm import (
    PAD_IDX, BOS_IDX, EOS_IDX,
    BlockDataset, collate_blocks, load_wikitext2,
)
# Reuse label smoothing — it's vocabulary-agnostic.
from phase0.train import LabelSmoothingLoss

from phase2.decoder_only import GPTv2, GPTConfigV2


# ---------------------------------------------------------------------------
# 1. Cosine LR schedule with linear warmup
# ---------------------------------------------------------------------------
# lr(step) =
#   step < warmup:        max_lr * (step / warmup)
#   warmup <= step < T:   min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
#                         where progress = (step - warmup) / (T - warmup)
#   step >= T:            min_lr
#
# This is the GPT-3 / LLaMA / Chinchilla schedule. The cosine decay from max_lr
# to ~10% of max_lr over the run is what's used in practice.
class CosineWarmupScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer,
                 max_lr: float, min_lr: float,
                 warmup_steps: int, total_steps: int) -> None:
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._step = 0
        self._rate = 0.0

    def step(self) -> None:
        self._step += 1
        s = self._step
        if s < self.warmup_steps:
            lr = self.max_lr * s / max(1, self.warmup_steps)
        elif s >= self.total_steps:
            lr = self.min_lr
        else:
            progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self._rate = lr

    @property
    def current_lr(self) -> float:
        return self._rate


# ---------------------------------------------------------------------------
# 2. Selective weight decay  (LLaMA / GPT-NeoX convention)
# ---------------------------------------------------------------------------
# Decay: 2D tensors (Linear weight matrices) — these benefit from L2 regularization.
# No-decay: 1D tensors (biases, RMSNorm gains) and embeddings.
def make_optimizer(model: GPTv2, lr: float, weight_decay: float, betas=(0.9, 0.95)) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Embedding weight is 2D but we explicitly exclude it (LLaMA convention).
        # Norm weights are 1D so they fall into no_decay naturally.
        if p.dim() < 2 or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    print(f"  decay params:    {sum(p.numel() for p in decay):,} "
          f"({len(decay)} tensors)")
    print(f"  no_decay params: {sum(p.numel() for p in no_decay):,} "
          f"({len(no_decay)} tensors)")

    return torch.optim.AdamW([
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=betas, eps=1e-8)


# ---------------------------------------------------------------------------
# 3. Train / eval steps
# ---------------------------------------------------------------------------
def train_epoch(
    model: GPTv2,
    loader: DataLoader,
    loss_fn: LabelSmoothingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    device: torch.device,
    grad_clip: float = 1.0,
    log_every: int = 100,
    use_amp: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp,
                                dtype=torch.bfloat16):
            logits, _ = model(x)                          # (B, T, V)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), y.reshape(B * T))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        n_tokens = (y != PAD_IDX).sum().item()
        total_loss   += loss.item()
        total_tokens += n_tokens

        if step % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"  step {step:5d} | "
                f"loss/tok {total_loss/total_tokens:.3f} | "
                f"ppl {math.exp(total_loss/total_tokens):.1f} | "
                f"lr {scheduler.current_lr:.2e} | "
                f"{total_tokens/elapsed:,.0f} tok/s"
            )

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model: GPTv2, loader: DataLoader, loss_fn: LabelSmoothingLoss,
             device: torch.device, use_amp: bool = True) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp,
                                dtype=torch.bfloat16):
            logits, _ = model(x)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), y.reshape(B * T))
        n_tokens = (y != PAD_IDX).sum().item()
        total_loss   += loss.item()
        total_tokens += n_tokens
    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# 4. Sample generation
# ---------------------------------------------------------------------------
def generate_sample(model: GPTv2, tokenizer, prompt: str, device: torch.device,
                    max_new_tokens: int = 60, temperature: float = 0.8,
                    top_k: int = 40) -> str:
    """KV-cached sampled generation. Use temperature=0 for greedy."""
    ids = [BOS_IDX] + tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             eos_id=EOS_IDX, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out_ids[0, 1:].tolist())


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ── Hyperparameters ──────────────────────────────────────────────────────
    BLOCK_SIZE     = 256
    BATCH_SIZE     = 32
    N_EPOCHS       = 5
    MAX_LR         = 3e-4    # LLaMA-style peak LR for this scale
    MIN_LR         = 3e-5    # ~10% of max
    WARMUP_STEPS   = 500
    WEIGHT_DECAY   = 0.1     # GPT-3 / LLaMA value
    GRAD_CLIP      = 1.0
    LABEL_SMOOTH   = 0.1
    VOCAB_SIZE     = 8000
    CHECKPOINT_DIR = Path("checkpoints_lm_v2")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"Using device: {device}, amp={use_amp}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds, tokenizer, vocab_size = load_wikitext2(
        block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_blocks, num_workers=2,
                              pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_blocks, num_workers=2,
                            pin_memory=use_amp)

    # ── Model — comparable size to Phase 1 (~10M) so loss curves are comparable ─
    cfg = GPTConfigV2(
        vocab_size = vocab_size,
        d_model    = 256,
        n_heads    = 8,
        n_layers   = 6,
        # d_ff auto-computed: 8/3 * 256 = 682 → rounded to 704
        max_len    = BLOCK_SIZE,
        dropout    = 0.0,
        pad_idx    = PAD_IDX,
    )
    model = GPTv2(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}, d_ff={cfg.d_ff}")

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    print("Building optimizer with selective weight decay:")
    optimizer = make_optimizer(model, lr=MAX_LR, weight_decay=WEIGHT_DECAY,
                                betas=(0.9, 0.95))
    total_steps = len(train_loader) * N_EPOCHS
    scheduler = CosineWarmupScheduler(optimizer, max_lr=MAX_LR, min_lr=MIN_LR,
                                      warmup_steps=WARMUP_STEPS,
                                      total_steps=total_steps)
    print(f"Total steps: {total_steps}, warmup: {WARMUP_STEPS}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(vocab_size=vocab_size, pad_idx=PAD_IDX,
                                 smoothing=LABEL_SMOOTH)

    sample_prompts = [
        "The history of",
        "In the early 20th century",
        "Albert Einstein was",
    ]

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    for epoch in range(1, N_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{N_EPOCHS} ===")
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer,
                                 scheduler, device, GRAD_CLIP, use_amp=use_amp)
        val_loss = evaluate(model, val_loader, loss_fn, device, use_amp=use_amp)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss {train_loss:.3f} (ppl {math.exp(train_loss):.1f}) | "
            f"Val loss {val_loss:.3f} (ppl {math.exp(val_loss):.1f}) | "
            f"LR {scheduler.current_lr:.2e} | "
            f"Time {elapsed:.1f}s"
        )

        for prompt in sample_prompts:
            sample = generate_sample(model, tokenizer, prompt, device,
                                     max_new_tokens=40, temperature=0.8, top_k=40)
            print(f"  PROMPT: {prompt!r}")
            print(f"  SAMPLE: {sample!r}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optim_state":  optimizer.state_dict(),
                "val_loss":     val_loss,
                "cfg":          cfg,
                "vocab_size":   vocab_size,
            }, CHECKPOINT_DIR / "best_model.pt")
            print(f"  ✓ Saved best (val_loss={val_loss:.3f}, ppl={math.exp(val_loss):.1f})")

    print(f"\nDone. Best val loss: {best_val_loss:.3f} "
          f"(ppl {math.exp(best_val_loss):.1f})")


if __name__ == "__main__":
    main()