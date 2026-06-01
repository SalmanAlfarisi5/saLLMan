"""
saLLMan Phase 3 (fine-tune) — Supervised fine-tuning loop.

Loads the pretrained GPTv3 checkpoint and continues training on the
instruction-tuning corpus produced by `finetune_data_prep.py`. Two main
differences from `pretrain.py`:

  1. Variable-length, padded batches with an attention mask — instead of
     fixed-block random window sampling. Fine-tune examples have natural
     boundaries (problem → reasoning → code → <eos>) and we don't want
     to splice unrelated examples together.

  2. Loss is masked. Tokens belonging to the *prompt* are set to -100
     in the labels tensor, so `F.cross_entropy(ignore_index=-100)` skips
     them. The model learns to *generate* reasoning + code conditioned
     on the problem, not to *reproduce* the problem.

Why supervised fine-tuning (SFT) at all
---------------------------------------
The pretrained model has seen ~2B tokens of raw Python from The Stack.
That gives it Python syntax fluency and some idiomatic patterns, but
zero awareness of "given a problem statement, produce a reasoning trace
and a solution". SFT teaches that distribution.

Hyperparameter philosophy
-------------------------
LR is one to two orders of magnitude lower than pretrain (2e-5 vs 3e-4).
Reason: pretrained weights are already in a good basin; we want to
adapt them, not push them out of it. A short cosine warmup (~3% of
training) is plenty.

Epochs: 3 by default. Fine-tune datasets are small (~thousands of
examples) so each epoch is fast, and 3 is enough to start showing
quality without overfitting to the SFT distribution.

Run:
    python finetune.py                               # all defaults
    python finetune.py --data finetune_data_v2 \\
                       --pretrain checkpoints_pretrain_v2/best.pt
    python finetune.py --resume                      # from last.pt
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
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decoder_only_v3 import GPTv3, GPTConfigV3


# ===========================================================================
# 1. Configuration
# ===========================================================================
@dataclass
class FTConfig:
    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir: str = "finetune_data_v2"
    pretrain_ckpt: str = "checkpoints_pretrain_v2/best.pt"
    out_dir: str = "checkpoints_finetune_v2"

    # ── Schedule ─────────────────────────────────────────────────────────────
    epochs: int = 3
    micro_batch_size: int = 8     # fits comfortably at max_len=512 / 97M model
    grad_accum_steps: int = 4     # effective batch 32
    log_interval: int = 20
    eval_interval: int = 200      # eval every N optimizer steps
    sample_interval: int = 400
    checkpoint_interval: int = 500

    # ── Optimizer ────────────────────────────────────────────────────────────
    max_lr: float = 2e-5          # one-to-two orders below pretrain
    min_lr: float = 2e-6
    warmup_ratio: float = 0.03    # 3% of total steps
    weight_decay: float = 0.1     # same selective recipe as pretrain
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    use_amp: bool = True
    gradient_checkpointing: bool = False  # turn on if OOM


# ===========================================================================
# 2. Dataset — JSONL with (input_ids, loss_mask)
# ===========================================================================
class FinetuneDataset(Dataset):
    """
    Each example is a dict {input_ids: [int], loss_mask: [0/1]}.
    Lengths vary; the collate function pads to the longest in the batch.

    We load the whole JSONL into memory — fine-tune corpora are small
    (~10s of MB) and this avoids per-batch I/O.
    """

    def __init__(self, path: Path) -> None:
        self.records: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.records.append({
                    "input_ids": rec["input_ids"],
                    "loss_mask": rec["loss_mask"],
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def make_collate(pad_idx: int):
    """
    Builds a collate function that:
      - Pads input_ids with `pad_idx` to the longest in the batch.
      - Builds the labels tensor: input_ids shifted by 1, with -100 on
        prompt positions and on pad positions.
      - Builds an attention key-padding mask (1 = real, 0 = pad). We don't
        actually pass this into the model in this codebase — the causal
        mask plus padding-as-pad-token works because PAD_IDX=0 never
        appears mid-sequence and the loss skips it via ignore_index.
        Kept for clarity and for any future move to KV-cached inference
        where you'd want to truncate at real length.
    """
    IGNORE = -100

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(r["input_ids"]) for r in batch)

        B = len(batch)
        input_ids = torch.full((B, max_len), pad_idx, dtype=torch.long)
        labels = torch.full((B, max_len), IGNORE, dtype=torch.long)
        # attn_keep is True where the position is a real token (non-pad).
        attn_keep = torch.zeros((B, max_len), dtype=torch.bool)

        for i, rec in enumerate(batch):
            ids = rec["input_ids"]
            mask = rec["loss_mask"]
            L = len(ids)
            ids_t = torch.tensor(ids, dtype=torch.long)
            mask_t = torch.tensor(mask, dtype=torch.long)

            # Model input is the full sequence including the final EOS.
            input_ids[i, :L] = ids_t
            attn_keep[i, :L] = True

            # Standard next-token-prediction shift:
            #   at position t, the model sees ids[t] and must predict ids[t+1].
            # So labels[t] = ids[t+1], gated by loss_mask[t+1].
            # The very last position has no next-token to predict.
            if L >= 2:
                shifted_ids = ids_t[1:]                         # ids[1:L]
                shifted_mask = mask_t[1:].bool()                # mask[1:L]
                # Where the shifted target's mask is 0 → keep IGNORE.
                # Where it is 1 → put the actual target id.
                labels_row = torch.full((L - 1,), IGNORE, dtype=torch.long)
                labels_row[shifted_mask] = shifted_ids[shifted_mask]
                labels[i, : L - 1] = labels_row

        return {"input_ids": input_ids, "labels": labels, "attn_keep": attn_keep}

    return collate


# ===========================================================================
# 3. Loss
# ===========================================================================
def masked_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Plain CE with ignore_index=-100. Loss is averaged over the *non-ignored*
    positions, which is what we want — each token in the response
    contributes equally regardless of how much prompt preceded it.
    """
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B * T, V),
        labels.reshape(B * T),
        ignore_index=-100,
    )


# ===========================================================================
# 4. Cosine LR with warmup (step-driven)
# ===========================================================================
def get_lr(step: int, total_steps: int, warmup_steps: int,
           max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ===========================================================================
# 5. Optimizer — same LLaMA selective-decay recipe as pretrain
# ===========================================================================
def make_optimizer(model: nn.Module, cfg: FTConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    print(f"  optimizer: {sum(p.numel() for p in decay):,} decayed, "
          f"{sum(p.numel() for p in no_decay):,} not decayed")
    fused_ok = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    extra = {"fused": True} if fused_ok and torch.cuda.is_available() else {}
    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.max_lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8, **extra,
    )


# ===========================================================================
# 6. Load pretrained checkpoint and rebuild the model
# ===========================================================================
def load_pretrained(ckpt_path: Path, gradient_checkpointing: bool,
                    device: torch.device) -> tuple[GPTv3, GPTConfigV3]:
    """
    The pretrain checkpoint stores `model_cfg` as a dict (via asdict).
    We rehydrate it into GPTConfigV3, build a fresh model, load weights.
    Optimizer state is intentionally NOT loaded — fine-tuning starts a
    new schedule from a clean optimizer.
    """
    print(f"Loading pretrained checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = payload["model_cfg"]
    # Allow CLI override of the checkpointing flag without retraining.
    cfg_dict["gradient_checkpointing"] = gradient_checkpointing
    model_cfg = GPTConfigV3(**cfg_dict)

    model = GPTv3(model_cfg).to(device)
    model.load_state_dict(payload["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded {n_params/1e6:.1f}M params from step {payload.get('step', '?')}")
    print(f"  model_cfg: d_model={model_cfg.d_model}, n_layers={model_cfg.n_layers}, "
          f"n_heads={model_cfg.n_heads}, max_len={model_cfg.max_len}, "
          f"vocab_size={model_cfg.vocab_size}")
    return model, model_cfg


# ===========================================================================
# 7. Eval + sample generation
# ===========================================================================
@torch.no_grad()
def evaluate(model: GPTv3, val_loader: DataLoader, device: torch.device,
             use_amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp,
                                dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            loss = masked_lm_loss(logits, labels)
        n = (labels != -100).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(1, total_tokens)


def generate_sample(model: GPTv3, tokenizer: Tokenizer, problem: str,
                    device: torch.device, max_new_tokens: int = 200) -> str:
    """
    Build a prompt in the same shape as training (problem + opening
    <reasoning> tag), then call model.generate. Greedy by default for
    visual sanity check.
    """
    from finetune_data_prep import (PROBLEM_OPEN, PROBLEM_CLOSE, REASON_OPEN,
                                     BOS_IDX, EOS_IDX)
    prompt = PROBLEM_OPEN + problem.strip() + PROBLEM_CLOSE + REASON_OPEN
    ids = [BOS_IDX] + tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new_tokens, eos_id=EOS_IDX,
                          temperature=0.7, top_k=40)
    return tokenizer.decode(out[0, 1:].tolist())


# ===========================================================================
# 8. Checkpointing
# ===========================================================================
def save_checkpoint(path: Path, model: GPTv3, optimizer: torch.optim.Optimizer,
                    step: int, epoch: int, best_val: float, ft_cfg: FTConfig,
                    model_cfg: GPTConfigV3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state":   model.state_dict(),
        "optim_state":   optimizer.state_dict(),
        "step":          step,
        "epoch":         epoch,
        "best_val_loss": best_val,
        "ft_cfg":        asdict(ft_cfg),
        "model_cfg":     asdict(model_cfg),
        "torch_rng_state": torch.get_rng_state(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ===========================================================================
# 9. Main loop
# ===========================================================================
def train(cfg: FTConfig, resume: bool) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # ── Data ────────────────────────────────────────────────────────────────
    data_dir = Path(cfg.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    pad_idx = meta["pad_idx"]
    tokenizer = Tokenizer.from_file(meta["tokenizer_path"])

    train_ds = FinetuneDataset(data_dir / "train.jsonl")
    val_ds = FinetuneDataset(data_dir / "val.jsonl")
    print(f"Train examples: {len(train_ds):,}, Val examples: {len(val_ds):,}")

    collate = make_collate(pad_idx=pad_idx)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.micro_batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.micro_batch_size, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # Step accounting. One "step" = one optimizer step, which spans
    # grad_accum_steps micro-batches.
    micro_per_epoch = math.ceil(len(train_ds) / cfg.micro_batch_size)
    steps_per_epoch = math.ceil(micro_per_epoch / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    print(f"Steps per epoch: {steps_per_epoch}, total: {total_steps}, "
          f"warmup: {warmup_steps}")

    # ── Model ───────────────────────────────────────────────────────────────
    model, model_cfg = load_pretrained(
        Path(cfg.pretrain_ckpt), cfg.gradient_checkpointing, device,
    )

    # Sanity: vocab must match
    if model_cfg.vocab_size != meta["vocab_size"]:
        raise RuntimeError(
            f"Vocab mismatch — model: {model_cfg.vocab_size}, "
            f"data: {meta['vocab_size']}. The pretrained tokenizer and the "
            "fine-tune tokenizer must be the same file."
        )

    optimizer = make_optimizer(model, cfg)

    # ── Resume (from last.pt of this FT run, NOT from pretrain) ─────────────
    start_step = 0
    start_epoch = 0
    best_val = float("inf")
    last_ckpt = out_dir / "last.pt"
    if resume and last_ckpt.exists():
        print(f"Resuming FT from {last_ckpt}")
        payload = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optim_state"])
        start_step = payload["step"]
        start_epoch = payload["epoch"]
        best_val = payload["best_val_loss"]
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])

    sample_problems = [
        "Given an integer array nums, return the maximum sum of any contiguous subarray.",
        "Write a function that returns True if a string is a palindrome, ignoring spaces and case.",
    ]

    # ── Train loop ──────────────────────────────────────────────────────────
    model.train()
    step = start_step
    t_log = time.time()
    loss_sum = 0.0
    loss_count = 0
    tokens_since_log = 0

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n=== Epoch {epoch + 1}/{cfg.epochs} ===")
        # iter the train_loader; we manually batch every grad_accum_steps
        accum_buffer: list[dict] = []
        for micro in train_loader:
            accum_buffer.append(micro)
            if len(accum_buffer) < cfg.grad_accum_steps:
                continue

            step += 1
            lr = get_lr(step, total_steps, warmup_steps, cfg.max_lr, cfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_tokens = 0
            for mb in accum_buffer:
                input_ids = mb["input_ids"].to(device, non_blocking=True)
                labels = mb["labels"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp,
                                        dtype=torch.bfloat16):
                    logits, _ = model(input_ids)
                    loss = masked_lm_loss(logits, labels) / cfg.grad_accum_steps
                loss.backward()
                accum_loss += loss.item()
                accum_tokens += (labels != -100).sum().item()
            accum_buffer.clear()

            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            loss_sum += accum_loss
            loss_count += 1
            tokens_since_log += accum_tokens

            # ── Logging ──
            if step % cfg.log_interval == 0:
                elapsed = time.time() - t_log
                tok_per_sec = tokens_since_log / max(elapsed, 1e-6)
                avg_loss = loss_sum / loss_count
                ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                print(f"step {step:5d}/{total_steps} | "
                      f"loss {avg_loss:.3f} | ppl {ppl:.1f} | "
                      f"lr {lr:.2e} | {tok_per_sec/1e3:.1f}k tok/s")
                with log_path.open("a") as f:
                    f.write(json.dumps({
                        "step": step, "epoch": epoch + 1,
                        "train_loss": avg_loss, "lr": lr,
                        "tok_per_sec": tok_per_sec,
                    }) + "\n")
                t_log = time.time()
                tokens_since_log = 0
                loss_sum = 0.0
                loss_count = 0

            # ── Eval ──
            if step % cfg.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, use_amp)
                val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                print(f"  >> step {step}: val_loss={val_loss:.3f}, val_ppl={val_ppl:.1f}")
                with log_path.open("a") as f:
                    f.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(out_dir / "best.pt", model, optimizer,
                                     step, epoch, best_val, cfg, model_cfg)
                    print(f"  >> saved best (val_loss={val_loss:.3f})")

            # ── Sample ──
            if step % cfg.sample_interval == 0:
                for prob in sample_problems:
                    txt = generate_sample(model, tokenizer, prob, device,
                                           max_new_tokens=160)
                    print(f"  PROBLEM: {prob[:80]}")
                    print(f"  SAMPLE : {txt[:300].replace(chr(10), ' ⏎ ')}")
                model.train()

            # ── Periodic save ──
            if step % cfg.checkpoint_interval == 0:
                save_checkpoint(out_dir / "last.pt", model, optimizer,
                                 step, epoch, best_val, cfg, model_cfg)
                print(f"  >> saved last.pt at step {step}")

        # Flush any trailing partial accumulation buffer at epoch end.
        if accum_buffer:
            step += 1
            lr = get_lr(step, total_steps, warmup_steps, cfg.max_lr, cfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            scale = 1.0 / len(accum_buffer)
            for mb in accum_buffer:
                input_ids = mb["input_ids"].to(device, non_blocking=True)
                labels = mb["labels"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp,
                                        dtype=torch.bfloat16):
                    logits, _ = model(input_ids)
                    loss = masked_lm_loss(logits, labels) * scale
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            accum_buffer.clear()

    # Final save
    save_checkpoint(out_dir / "last.pt", model, optimizer, step,
                    cfg.epochs - 1, best_val, cfg, model_cfg)
    val_loss = evaluate(model, val_loader, device, use_amp)
    print(f"\nFine-tune complete. Final val_loss={val_loss:.3f}, "
          f"best={best_val:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                        help="Override data_dir (default: finetune_data_v2).")
    parser.add_argument("--pretrain", type=str, default=None,
                        help="Override pretrain checkpoint path.")
    parser.add_argument("--out", type=str, default=None,
                        help="Override out_dir.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--micro_batch", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                        help="Override max_lr.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = FTConfig()
    if args.data: cfg.data_dir = args.data
    if args.pretrain: cfg.pretrain_ckpt = args.pretrain
    if args.out: cfg.out_dir = args.out
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.micro_batch is not None: cfg.micro_batch_size = args.micro_batch
    if args.grad_accum is not None: cfg.grad_accum_steps = args.grad_accum
    if args.lr is not None: cfg.max_lr = args.lr
    if args.gradient_checkpointing: cfg.gradient_checkpointing = True

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
