"""
Phase 0 Training Script — German → English translation on Multi30k.
Add this file alongside transformer.py. Run with: python train.py

Dependencies:
    pip install datasets
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from transformer import Transformer, TransformerConfig


# ---------------------------------------------------------------------------
# 1. Special token indices — agreed upon before building vocab
# ---------------------------------------------------------------------------
PAD_IDX = 0
BOS_IDX = 1   # Beginning of Sentence
EOS_IDX = 2   # End of Sentence
UNK_IDX = 3   # Unknown token


# ---------------------------------------------------------------------------
# 2. Tokenizer — pure Python regex, zero dependencies, zero downloads
# ---------------------------------------------------------------------------
# Matches: sequences of unicode letters/digits, OR any single non-whitespace.
# This correctly splits punctuation from words for both German and English.
# e.g. "lädt." → ["lädt", "."]   "don't" → ["don", "'", "t"]
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer:
    """
    Regex-based word tokenizer. Works for both German and English.
    No model files, no internet required.
    """
    def tokenize_de(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def tokenize_en(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# 3. Vocabulary — maps tokens <-> integer ids
# ---------------------------------------------------------------------------
class Vocab:
    """
    Simple frequency-based vocabulary.
    Tokens appearing fewer than `min_freq` times are mapped to UNK.

    self.token2id: dict[str, int]
    self.id2token: list[str]
    """
    SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]   # must match PAD/BOS/EOS/UNK indices above

    def __init__(self, min_freq: int = 2) -> None:
        self.min_freq = min_freq
        self.token2id: dict[str, int] = {}
        self.id2token: list[str] = []

    def build(self, token_lists: list[list[str]]) -> None:
        """Count all tokens across all sentences, then assign ids."""
        from collections import Counter
        counter: Counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        # Special tokens first so their ids are fixed at 0,1,2,3
        self.id2token = self.SPECIALS.copy()
        for token, freq in counter.most_common():
            if freq >= self.min_freq:
                self.id2token.append(token)

        self.token2id = {tok: idx for idx, tok in enumerate(self.id2token)}

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert list of token strings to list of integer ids."""
        return [self.token2id.get(tok, UNK_IDX) for tok in tokens]

    def __len__(self) -> int:
        return len(self.id2token)


# ---------------------------------------------------------------------------
# 4. Dataset — wraps Multi30k pairs into a PyTorch Dataset
# ---------------------------------------------------------------------------
class TranslationDataset(Dataset):
    """
    Each item is a (src_ids, tgt_ids) pair as Python lists of integers.
    BOS and EOS tokens are added here so the model always sees them.

    src_ids: [BOS, w1, w2, ..., wN, EOS]
    tgt_ids: [BOS, w1, w2, ..., wM, EOS]

    During training, the model receives tgt[:-1] as input (teacher forcing)
    and tgt[1:] as the label — so BOS starts the input and EOS ends the label.
    """
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        tokenizer: Tokenizer,
        max_len: int = 100,
    ) -> None:
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Tokenize and encode everything upfront — faster than doing it per batch
        self.data: list[tuple[list[int], list[int]]] = []
        for de_text, en_text in pairs:
            src_tokens = tokenizer.tokenize_de(de_text)
            tgt_tokens = tokenizer.tokenize_en(en_text)
            # Skip sentences that are too long — they'd dominate padding cost
            if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
                continue
            src_ids = [BOS_IDX] + src_vocab.encode(src_tokens) + [EOS_IDX]
            tgt_ids = [BOS_IDX] + tgt_vocab.encode(tgt_tokens) + [EOS_IDX]
            self.data.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.data[idx]


# ---------------------------------------------------------------------------
# 5. Collate function — pads a batch of variable-length sequences
# ---------------------------------------------------------------------------
def collate_fn(batch: list[tuple[list[int], list[int]]]) -> tuple[Tensor, Tensor]:
    """
    Pads src and tgt sequences in a batch to the same length using PAD_IDX.
    Returns:
        src: (B, max_src_len)
        tgt: (B, max_tgt_len)
    """
    src_batch, tgt_batch = zip(*batch)

    max_src = max(len(s) for s in src_batch)
    max_tgt = max(len(t) for t in tgt_batch)

    # Pad each sequence to the max length in this batch
    src_padded = [s + [PAD_IDX] * (max_src - len(s)) for s in src_batch]
    tgt_padded = [t + [PAD_IDX] * (max_tgt - len(t)) for t in tgt_batch]

    return (
        torch.tensor(src_padded, dtype=torch.long),
        torch.tensor(tgt_padded, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# 6. Noam Learning Rate Scheduler — Paper §5.3
# ---------------------------------------------------------------------------
class NoamScheduler:
    """
    Paper §5.3, eq. (3):
        lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    - For the first `warmup_steps` steps: lr increases linearly
    - After warmup: lr decays proportionally to 1/sqrt(step)

    This is NOT a standard PyTorch scheduler — it directly sets the lr
    on the optimizer each step. Call .step() once per batch.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0
        self._rate = 0.0

    def step(self) -> None:
        self._step += 1
        rate = self._compute_lr()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def _compute_lr(self) -> float:
        s = self._step
        return self.d_model ** (-0.5) * min(s ** (-0.5), s * self.warmup_steps ** (-1.5))

    @property
    def current_lr(self) -> float:
        return self._rate


# ---------------------------------------------------------------------------
# 7. Label Smoothing Loss — Paper §5.4
# ---------------------------------------------------------------------------
class LabelSmoothingLoss(nn.Module):
    """
    Paper §5.4: label smoothing with ε = 0.1.
    Instead of training the model to output probability 1.0 for the correct
    token, we target (1 - ε) for the correct token and ε / (V-2) for all
    others (excluding PAD and the correct token itself).

    Why does this help? Hard targets (0 / 1) make the model overconfident.
    Soft targets encourage the model to keep some probability mass on other
    plausible tokens, which acts as a regularizer and improves BLEU.

    Uses KLDivLoss internally because:
        KL(p || q) = sum(p * log(p/q)) = sum(p * log(p)) - sum(p * log(q))
    The first term is constant w.r.t. model params, so minimizing KL is
    equivalent to minimizing cross-entropy with soft targets.
    """
    def __init__(self, vocab_size: int, pad_idx: int = PAD_IDX, smoothing: float = 0.1) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.kl = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        logits:  (N, vocab_size)  — raw scores from generator
        targets: (N,)             — true token ids
        Returns scalar loss.
        """
        # log_softmax is numerically more stable than log(softmax(x))
        log_probs = torch.log_softmax(logits, dim=-1)   # (N, vocab_size)

        # Build smooth target distribution
        # Start with uniform ε / (V - 2) everywhere
        # (V - 2: exclude PAD and the correct token from the uniform spread)
        smooth_val = self.smoothing / (self.vocab_size - 2)
        targets_smooth = torch.full_like(log_probs, smooth_val)

        # Put (1 - ε) on the correct token
        targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Zero out PAD positions — we don't want to learn to predict PAD
        targets_smooth[:, self.pad_idx] = 0.0
        pad_mask = (targets == self.pad_idx)
        targets_smooth[pad_mask] = 0.0

        return self.kl(log_probs, targets_smooth)


# ---------------------------------------------------------------------------
# 8. Data loading helper
# ---------------------------------------------------------------------------
def load_multi30k(tokenizer: Tokenizer, min_freq: int = 2, max_len: int = 100):
    """
    Downloads Multi30k via HuggingFace datasets (no torchtext needed).
    Builds vocabularies from training split only (never peek at val/test).
    Returns train/val datasets and both vocabularies.
    """
    print("Downloading Multi30k...")
    # HuggingFace hosts this as 'bentrevett/multi30k'
    raw = load_dataset("bentrevett/multi30k")

    def extract_pairs(split):
        return [(ex["de"], ex["en"]) for ex in raw[split]]

    train_pairs = extract_pairs("train")
    val_pairs   = extract_pairs("validation")

    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    # Build vocabularies from TRAINING data only
    print("Building vocabularies...")
    de_tokens = [tokenizer.tokenize_de(de) for de, _ in train_pairs]
    en_tokens = [tokenizer.tokenize_en(en) for _, en in train_pairs]

    src_vocab = Vocab(min_freq=min_freq)
    tgt_vocab = Vocab(min_freq=min_freq)
    src_vocab.build(de_tokens)
    tgt_vocab.build(en_tokens)

    print(f"German vocab size:  {len(src_vocab)}")
    print(f"English vocab size: {len(tgt_vocab)}")

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab, tokenizer, max_len)
    val_ds   = TranslationDataset(val_pairs,   src_vocab, tgt_vocab, tokenizer, max_len)

    print(f"Train examples (after length filter): {len(train_ds)}")
    print(f"Val   examples (after length filter): {len(val_ds)}")

    return train_ds, val_ds, src_vocab, tgt_vocab


# ---------------------------------------------------------------------------
# 9. Training and validation steps
# ---------------------------------------------------------------------------
def train_epoch(
    model: Transformer,
    loader: DataLoader,
    loss_fn: LabelSmoothingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """
    One full pass over the training set.
    Returns average loss per token.
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)   # (B, T_src)
        tgt = tgt.to(device)   # (B, T_tgt)

        # Teacher forcing:
        # tgt_in  = tgt without the last token  → [BOS, w1, w2, ..., wN]
        # tgt_out = tgt without the first token → [w1, w2, ..., wN, EOS]
        # The model sees tgt_in and must predict tgt_out at each position.
        tgt_in  = tgt[:, :-1]   # (B, T_tgt - 1)
        tgt_out = tgt[:, 1:]    # (B, T_tgt - 1)

        # Forward pass → logits: (B, T_tgt-1, vocab_size)
        logits = model(src, tgt_in)

        # Flatten for loss: (B * T, vocab_size) and (B * T,)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), tgt_out.reshape(B * T))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients, especially at
        # the start of training before the scheduler has warmed up.
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()   # Noam scheduler updates lr every step

        # Count non-padding tokens for per-token loss normalization
        n_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss   += loss.item()
        total_tokens += n_tokens

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(
    model: Transformer,
    loader: DataLoader,
    loss_fn: LabelSmoothingLoss,
    device: torch.device,
) -> float:
    """
    One full pass over the validation set.
    @torch.no_grad() disables gradient tracking — saves memory and speeds up eval.
    Returns average loss per token.
    """
    model.eval()   # disables dropout
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(src, tgt_in)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), tgt_out.reshape(B * T))

        n_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss   += loss.item()
        total_tokens += n_tokens

    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# 10. Greedy decoding — for sanity-checking translation quality during training
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: Tensor,          # (1, T_src) — single sentence
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int = 50,
) -> str:
    """
    Greedy decoding: always pick the highest-probability token at each step.
    Not the best decoding strategy (beam search is better) but useful for
    quickly checking if the model is learning something sensible.
    """
    model.eval()
    src = src.to(device)

    # Encode source once
    from transformer import make_pad_mask
    src_mask = make_pad_mask(src, PAD_IDX)
    memory = model.encoder(src, src_mask)

    # Start decoder with BOS token
    tgt_ids = [BOS_IDX]

    for _ in range(max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)   # (1, current_len)

        # Build masks for current decoder state
        _, tgt_mask, memory_mask = model.make_masks(src, tgt)
        out = model.decoder(tgt, memory, tgt_mask, memory_mask)           # (1, current_len, 512)
        logits = model.generator(out[:, -1, :])                           # (1, vocab_size) — last position only
        next_token = logits.argmax(dim=-1).item()

        if next_token == EOS_IDX:
            break
        tgt_ids.append(next_token)

    # Convert ids back to words, skipping BOS
    tokens = [tgt_vocab.id2token[i] for i in tgt_ids[1:]]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 11. Main training loop
# ---------------------------------------------------------------------------
def main() -> None:
    # ── Hyperparameters ──────────────────────────────────────────────────────
    BATCH_SIZE    = 128
    N_EPOCHS      = 15
    WARMUP_STEPS  = 4000    # Paper §5.3
    LABEL_SMOOTH  = 0.1     # Paper §5.4
    GRAD_CLIP     = 1.0
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    tokenizer = Tokenizer()
    train_ds, val_ds, src_vocab, tgt_vocab = load_multi30k(tokenizer)  # type: ignore

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # Using smaller dims than the paper (d_model=256 instead of 512)
    # because Multi30k is a small dataset — a full 512-dim model would overfit.
    cfg = TransformerConfig(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model  = 256,
        n_heads  = 8,
        n_layers = 3,
        d_ff     = 512,
        max_len  = 200,
        dropout  = 0.1,
        pad_idx  = PAD_IDX,
    )
    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer — Paper §5.3: Adam with β1=0.9, β2=0.98, ε=1e-9 ──────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0,                    # lr is set by the scheduler each step
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=WARMUP_STEPS)

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=LABEL_SMOOTH,
    )

    # ── A few sample sentences to watch during training ───────────────────────
    sample_de = "eine gruppe von männern lädt baumwolle auf einen lastwagen."
    sample_src = torch.tensor(
        [[BOS_IDX] + src_vocab.encode(tokenizer.tokenize_de(sample_de)) + [EOS_IDX]],
        dtype=torch.long,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, GRAD_CLIP)
        val_loss   = evaluate(model, val_loader, loss_fn, device)

        elapsed = time.time() - t0

        # Perplexity = exp(loss) — a standard MT metric for language models.
        # Lower is better. Perplexity of 1 = perfect, perplexity of V = random.
        train_ppl = math.exp(train_loss)
        val_ppl   = math.exp(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {train_loss:.3f} | Train PPL: {train_ppl:.1f} | "
            f"Val loss: {val_loss:.3f} | Val PPL: {val_ppl:.1f} | "
            f"LR: {scheduler.current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Greedy decode a sample sentence every epoch to visually sanity-check
        translation = greedy_decode(model, sample_src, src_vocab, tgt_vocab, device)
        print(f"  DE: {sample_de}")
        print(f"  EN: {translation}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    val_loss,
                "cfg":         cfg,
                "src_vocab":   src_vocab,
                "tgt_vocab":   tgt_vocab,
            }, CHECKPOINT_DIR / "best_model.pt")
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.3f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.3f}")


if __name__ == "__main__":
    main()