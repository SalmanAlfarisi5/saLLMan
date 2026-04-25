"""
saLLMan Phase 1 — Training script for the decoder-only LM.

Trains GPT (decoder_only.py) on WikiText-2 with a next-token-prediction
objective. This is a SANITY-CHECK trainer: its purpose is to verify that
the Phase 1 architecture trains correctly (loss falls, generation gets
more coherent each epoch) before we start swapping in modern components
in Phase 2.

We deliberately reuse Phase 0 components where they're vocabulary-agnostic:
    - LabelSmoothingLoss   (Vaswani et al. 2017 §5.4)
    - NoamScheduler        (Vaswani et al. 2017 §5.3)
    - Adam(β1=0.9, β2=0.98)

What's new vs train.py
----------------------
1. Single vocab (no src/tgt split).
2. BPE tokenizer via HuggingFace `tokenizers` for word-piece-style coverage —
   regex word-tokenization works for translation but produces a huge vocab
   for raw text. We use byte-level BPE like GPT-2 (Radford et al. 2019).
3. Fixed-length block packing instead of sentence-pair batching. Standard LM
   recipe: concatenate the entire corpus, then chop into blocks of `block_size`
   tokens. Each block is one training example; loss is teacher-forced
   next-token-prediction (input = block[:-1], target = block[1:]).
4. Mixed-precision training (torch.amp) since you mentioned the 3060 Ti.
   This roughly halves VRAM use and speeds things up ~1.5–2x on Ampere.

Dataset: WikiText-2 (Merity et al. 2016, https://arxiv.org/abs/1609.07843)
The standard small LM benchmark. ~2M training tokens of clean Wikipedia prose.

Dependencies:
    pip install datasets tokenizers
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# Reuse the parts of Phase 0 that are vocabulary-agnostic.
from train import LabelSmoothingLoss, NoamScheduler

from decoder_only import GPT, GPTConfig


# ---------------------------------------------------------------------------
# 1. Special token ids — single vocabulary now, so just PAD/BOS/EOS/UNK
# ---------------------------------------------------------------------------
# These match the order our BPE tokenizer adds them in (see build_tokenizer).
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ---------------------------------------------------------------------------
# 2. BPE tokenizer  (Sennrich et al. 2016, used by GPT-2/3/LLaMA)
# ---------------------------------------------------------------------------
# Why BPE instead of the regex word tokenizer from train.py?
#
# Word tokenization works fine when the vocabulary is small and bounded
# (Multi30k has ~5k unique German words). Raw text has a long tail —
# rare words, numbers, code — and a word vocab would either be massive or
# OOV-heavy. BPE solves this by splitting rare words into subword pieces:
#   "unbelievably" → ["un", "believ", "ably"]
# so we get a fixed vocab size with full coverage of any input.
def build_tokenizer(text_iter, vocab_size: int = 8000, save_path: Path | None = None):
    """Train a byte-level BPE tokenizer on the provided text iterator."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],   # order = id 0,1,2,3
        show_progress=False,
    )
    tok.train_from_iterator(text_iter, trainer=trainer)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(save_path))

    return tok


# ---------------------------------------------------------------------------
# 3. Block-packed dataset
# ---------------------------------------------------------------------------
# Standard LM training pattern (used by GPT-2/3, LLaMA, every "nanoGPT"-style codebase):
#   1. Tokenize the entire corpus into one giant 1D tensor of token ids.
#   2. Reshape into chunks of length `block_size + 1`.
#   3. For each chunk, input = chunk[:-1], target = chunk[1:].
#
# This is much more efficient than sentence-pair batching:
#   - No padding (every block is full → ~100% of tokens contribute to loss).
#   - No "wasted" short sequences.
# The trade-off: blocks ignore document boundaries — a block can span the
# end of one Wikipedia article and the start of another. This is fine
# (and standard) because the model just learns "after </eos> a new topic
# may start", which is realistic.
class BlockDataset(Dataset):
    def __init__(self, token_ids: Tensor, block_size: int) -> None:
        # token_ids: 1D LongTensor of all corpus tokens concatenated.
        # We need block_size + 1 because input and target are offset by 1.
        self.block_size = block_size
        n_blocks = (token_ids.size(0) - 1) // block_size
        # Truncate to a multiple of block_size for clean reshaping.
        self.data = token_ids[: n_blocks * block_size + 1]
        self.n_blocks = n_blocks

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # Grab block_size + 1 contiguous tokens; split into input/target.
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        x = chunk[:-1]   # (block_size,) — what the model sees
        y = chunk[1:]    # (block_size,) — what it must predict
        return x, y


def collate_blocks(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """All blocks are the same length, so we just stack — no padding needed."""
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------
# 4. Data loading pipeline
# ---------------------------------------------------------------------------
def load_wikitext2(block_size: int, vocab_size: int = 8000, cache_dir: Path = Path("tok_cache")):
    """
    Downloads WikiText-2, trains a BPE tokenizer on the train split,
    and returns block-packed train/val datasets.
    """
    print("Downloading WikiText-2...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Each split is a list of dicts with a "text" field. Many are empty
    # or just whitespace headers — we keep them all because BPE handles
    # whitespace, but filter trivially empty lines.
    def texts(split):
        return [t for t in raw[split]["text"] if t.strip()]

    train_texts = texts("train")
    val_texts   = texts("validation")
    print(f"Train lines: {len(train_texts):,}, Val lines: {len(val_texts):,}")

    # Train tokenizer on train split only. (Touching val with the tokenizer
    # is technically fine since we don't use val labels — but staying clean.)
    tok_path = cache_dir / "wikitext2_bpe.json"
    if tok_path.exists():
        print(f"Loading cached tokenizer from {tok_path}")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
    else:
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        tokenizer = build_tokenizer(train_texts, vocab_size=vocab_size, save_path=tok_path)

    actual_vocab = tokenizer.get_vocab_size()
    print(f"Vocab size: {actual_vocab}")

    # Encode the entire corpus into one long stream of ids.
    # We insert <eos> between documents so the model learns document boundaries.
    def encode_stream(texts: list[str]) -> Tensor:
        ids: list[int] = []
        for t in texts:
            enc = tokenizer.encode(t).ids
            ids.extend(enc)
            ids.append(EOS_IDX)
        return torch.tensor(ids, dtype=torch.long)

    print("Encoding corpus...")
    train_ids = encode_stream(train_texts)
    val_ids   = encode_stream(val_texts)
    print(f"Train tokens: {train_ids.numel():,}")
    print(f"Val   tokens: {val_ids.numel():,}")

    train_ds = BlockDataset(train_ids, block_size=block_size)
    val_ds   = BlockDataset(val_ids,   block_size=block_size)
    print(f"Train blocks: {len(train_ds):,} (block_size={block_size})")
    print(f"Val   blocks: {len(val_ds):,}")

    return train_ds, val_ds, tokenizer, actual_vocab


# ---------------------------------------------------------------------------
# 5. Train / eval steps
# ---------------------------------------------------------------------------
# Almost identical to train.py's train_epoch, with three small differences:
#   - Single (input_ids, targets) batch instead of (src, tgt) → tgt[:,:-1]/tgt[:,1:]
#   - Mixed precision via torch.amp.autocast + GradScaler
#   - We compute n_tokens against PAD_IDX, but block-packed data has no padding
#     so it equals B*T — kept for safety/symmetry with the loss normalization.
def train_epoch(
    model: GPT,
    loader: DataLoader,
    loss_fn: LabelSmoothingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    scaler: "torch.amp.GradScaler | None",
    device: torch.device,
    grad_clip: float = 1.0,
    log_every: int = 100,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    use_amp = scaler is not None

    t0 = time.time()
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)   # (B, T)
        y = y.to(device, non_blocking=True)   # (B, T)

        optimizer.zero_grad(set_to_none=True)

        # Mixed-precision forward. autocast runs ops in bf16/fp16 where safe.
        # We use bf16 on Ampere (3060 Ti supports it natively); falls back to
        # fp32 elsewhere with no GradScaler needed.
        with torch.amp.autocast(device_type=device.type, enabled=use_amp,
                                dtype=torch.bfloat16):
            logits = model(x)                              # (B, T, V)
            B, T, V = logits.shape
            # LabelSmoothingLoss returns SUM not mean; we'll normalize by tokens later.
            loss = loss_fn(logits.view(B * T, V), y.reshape(B * T))

        # bfloat16 doesn't need a GradScaler (its dynamic range matches fp32).
        # If you switch to fp16, wrap loss.backward()/optimizer.step() in scaler calls.
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        n_tokens = (y != PAD_IDX).sum().item()
        total_loss   += loss.item()
        total_tokens += n_tokens

        if step % log_every == 0:
            elapsed = time.time() - t0
            tok_per_sec = total_tokens / elapsed
            print(
                f"  step {step:5d} | "
                f"loss/tok {total_loss/total_tokens:.3f} | "
                f"ppl {math.exp(total_loss/total_tokens):.1f} | "
                f"lr {scheduler.current_lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s"
            )

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, loss_fn: LabelSmoothingLoss,
             device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            logits = model(x)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), y.reshape(B * T))
        n_tokens = (y != PAD_IDX).sum().item()
        total_loss   += loss.item()
        total_tokens += n_tokens
    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# 6. Sample generation — visual sanity check each epoch
# ---------------------------------------------------------------------------
def generate_sample(model: GPT, tokenizer, prompt: str, device: torch.device,
                    max_new_tokens: int = 60) -> str:
    """Encode prompt, run greedy generate, decode back to text."""
    ids = [BOS_IDX] + tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, eos_id=EOS_IDX)
    # out_ids shape: (1, len). Drop BOS for clean printing.
    text = tokenizer.decode(out_ids[0, 1:].tolist())
    return text


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ── Hyperparameters ──────────────────────────────────────────────────────
    # Model is sized for ~10M params — small enough to train in minutes on a
    # 3060 Ti, large enough to clearly see loss drop on WikiText-2.
    # We're sanity-checking the architecture, not training a real LM.
    BLOCK_SIZE   = 256       # context length during training
    BATCH_SIZE   = 32        # 32 * 256 = 8192 tokens per step
    N_EPOCHS     = 5
    WARMUP_STEPS = 1000
    LABEL_SMOOTH = 0.1
    GRAD_CLIP    = 1.0
    VOCAB_SIZE   = 8000
    CHECKPOINT_DIR = Path("checkpoints_lm")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp = (device.type == "cuda")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds, tokenizer, vocab_size = load_wikitext2(
        block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_blocks, num_workers=2, pin_memory=use_amp,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_blocks, num_workers=2, pin_memory=use_amp,
    )

    # ── Model — ~10M params ──────────────────────────────────────────────────
    cfg = GPTConfig(
        vocab_size = vocab_size,
        d_model    = 256,
        n_heads    = 8,
        n_layers   = 6,
        d_ff       = 1024,        # 4 * d_model (paper default; will become 8/3 in Phase 2d)
        max_len    = BLOCK_SIZE,
        dropout    = 0.1,
        pad_idx    = PAD_IDX,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer (Vaswani §5.3) ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=0,
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=WARMUP_STEPS)

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(vocab_size=vocab_size, pad_idx=PAD_IDX,
                                 smoothing=LABEL_SMOOTH)

    # bf16 doesn't need a GradScaler; keeping the variable for symmetry.
    scaler = "bf16" if use_amp else None  # marker; we don't actually scale

    # ── Sample prompts to track generation quality ────────────────────────────
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
                                 scheduler, scaler, device, GRAD_CLIP)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        train_ppl = math.exp(train_loss)
        val_ppl   = math.exp(val_loss)
        print(
            f"Epoch {epoch:02d} | "
            f"Train loss {train_loss:.3f} (ppl {train_ppl:.1f}) | "
            f"Val loss {val_loss:.3f} (ppl {val_ppl:.1f}) | "
            f"LR {scheduler.current_lr:.2e} | "
            f"Time {elapsed:.1f}s"
        )

        # Generate samples to visually track quality.
        for prompt in sample_prompts:
            sample = generate_sample(model, tokenizer, prompt, device, max_new_tokens=40)
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
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.3f}, ppl={val_ppl:.1f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.3f} "
          f"(ppl {math.exp(best_val_loss):.1f})")


if __name__ == "__main__":
    main()