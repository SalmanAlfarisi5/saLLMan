"""
saLLMan Phase 1 — Decoder-only Transformer (GPT-style).

This file imports the proven primitives from my vanilla `transformer.py`
(MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding,
make_causal_mask, make_pad_mask) and assembles them into a decoder-only LM.

References
----------
- Brown et al. 2020, "Language Models are Few-Shot Learners" (GPT-3)
  https://arxiv.org/abs/2005.14165
- Radford et al. 2018, "Improving Language Understanding by Generative
  Pre-Training" (GPT-1) — the original decoder-only LM
- Vaswani et al. 2017, "Attention Is All You Need" (architectural primitives)

Differences from the vanilla Transformer
----------------------------------------
1. No encoder stack.
2. Each block has 2 sublayers (masked self-attn + FFN); cross-attention is gone.
3. Single vocabulary (no src_vocab / tgt_vocab split).
4. Forward signature is forward(input_ids) → logits — no `memory` argument.
5. Mask logic only needs (causal & padding); no memory mask.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# Reuse primitives from the vanilla transformer — they are unchanged.
# A multi-head self-attention block is a multi-head self-attention block,
# regardless of whether it lives in an encoder, an encoder-decoder, or a
# decoder-only model. Same for the FFN and PE.
from transformer import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    make_causal_mask,
    make_pad_mask,
)


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
# Differences from TransformerConfig:
#   - One `vocab_size` instead of (src_vocab_size, tgt_vocab_size). A
#     decoder-only LM consumes and produces tokens from the same vocabulary.
#   - No `max_len` distinction between src/tgt — there's only one sequence.
@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048           # 4 * d_model — will become 8/3 * d_model in Phase 2d (SwiGLU)
    max_len: int = 1024        # GPT-2 used 1024; we'll keep this modest for an 8GB GPU
    dropout: float = 0.1
    pad_idx: int = 0

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0


# ---------------------------------------------------------------------------
# 2. Decoder-only block  (the central architectural change)
# ---------------------------------------------------------------------------
# Compare side-by-side with DecoderLayer in transformer.py:
#
#   ENCODER-DECODER DecoderLayer:        DECODER-ONLY GPTBlock:
#     1) masked self-attn                  1) masked self-attn
#     2) cross-attn (over memory)          ── REMOVED ──
#     3) FFN                               2) FFN
#
class GPTBlock(nn.Module):
    """A single transformer block in a decoder-only LM."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        # Two LayerNorms instead of three — one per sublayer.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        # x: (B, T, d_model)
        # attn_mask: (B, 1, T, T) — causal AND padding combined; True = KEEP
        # Sublayer 1: masked multi-head self-attention.
        # Q, K, V all come from x — there is no separate "memory" tensor.
        attn_out = self.self_attn(x, x, x, mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_out))   # Post-LN (will flip to Pre-LN in Phase 2a)
        # Sublayer 2: position-wise feed-forward.
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# ---------------------------------------------------------------------------
# 3. The model
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    """
    Decoder-only Transformer language model. Predicts the next token given
    the prefix [t_0, ..., t_{i-1}] for every position i.

    Forward signature: forward(input_ids: (B, T)) -> logits: (B, T, vocab_size)

    To train as a language model:
        input_ids = tokens[:-1]   # the model sees positions 0..T-1
        targets   = tokens[1:]    # and must predict positions 1..T at each step
    This is the same teacher-forcing pattern you used in Phase 0, except now
    `input_ids` and `targets` are slices of THE SAME sequence rather than
    separate src/tgt tensors. This is the GPT next-token-prediction objective.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Token embedding — single vocab, no src/tgt split.
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        # Positional encoding — sinusoidal for now; replaced by RoPE in Phase 2c.
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)

        self.blocks = nn.ModuleList(
            [GPTBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )

        # Output head: project hidden states to vocabulary logits.
        # Named `lm_head` by convention (HuggingFace, GPT-2, LLaMA all use this name).
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share weights between the input embedding and output projection.
        # Same trick you used in Phase 0 (generator.weight = decoder.embed.weight).
        # Cuts parameters by ~vocab_size * d_model and is standard in GPT-2/LLaMA.
        self.lm_head.weight = self.embed.weight

        self._init_params()

    def _init_params(self) -> None:
        # Same Xavier init as Phase 0. GPT-2 uses N(0, 0.02) instead;
        # we'll keep Xavier for now to minimize variables while validating
        # the architectural change.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # --- mask construction ----------------------------------------------------
    # We need exactly ONE mask: (B, 1, T, T) where position i can attend to
    # position j iff (j <= i) AND (j is not padding).
    #
    # Compare to Phase 0's make_masks: that returned three masks
    # (src padding, tgt causal+pad, memory padding). With no encoder and no
    # cross-attention, two of those become irrelevant.
    def make_attn_mask(self, input_ids: Tensor) -> Tensor:
        # pad: True for non-pad positions.    shape: (B, 1, 1, T)
        pad = make_pad_mask(input_ids, self.cfg.pad_idx)
        # causal: lower-triangular True.       shape: (1, 1, T, T)
        causal = make_causal_mask(input_ids.size(1), input_ids.device)
        # Combine: True only where BOTH allow attention.
        # Broadcasts cleanly to (B, 1, T, T).
        return pad & causal

    # --- forward --------------------------------------------------------------
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        input_ids: (B, T) integer token ids
        returns:   (B, T, vocab_size) logits
        """
        attn_mask = self.make_attn_mask(input_ids)

        # Embed + scale + positional encoding (same recipe as Phase 0).
        x = self.embed(input_ids) * math.sqrt(self.cfg.d_model)   # (B, T, d_model)
        x = self.pos_enc(x)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        return self.lm_head(x)  # (B, T, vocab_size)

    # --- generation -----------------------------------------------------------
    # Greedy generation. Phase 0's greedy_decode took a `src` and used the
    # encoder; here we just keep extending the same sequence.
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,            # (B, T_prompt) starting context
        max_new_tokens: int,
        eos_id: int | None = None,
    ) -> Tensor:
        """
        Naive (no KV-cache) greedy generation. We re-run the full forward
        pass for every new token, which is O(T^2) but matches what Phase 0
        does and is fine for sanity checks. KV-caching is a later optimization.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Truncate to max_len if the prompt + generated tokens overflow.
            ctx = input_ids[:, -self.cfg.max_len:]
            logits = self(ctx)                    # (B, T, vocab_size)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (B, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return input_ids


# ---------------------------------------------------------------------------
# 4. Smoke test  (mirrors Phase 0's smoke test for parity)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_len=128,
    )
    model = GPT(cfg)

    # Fake batch: 2 sequences of length 10. Note: no src/tgt — one tensor.
    input_ids = torch.tensor([
        [1, 5, 7, 9, 11, 13, 15, 0, 0, 0],     # padded
        [1, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    ])
    logits = model(input_ids)
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"logits shape:    {tuple(logits.shape)}  (expected: (2, 10, 1000))")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")

    # Try generating 5 tokens from a tiny prompt.
    prompt = torch.tensor([[1, 5, 7]])
    out = model.generate(prompt, max_new_tokens=5)
    print(f"generated: {out.tolist()}")