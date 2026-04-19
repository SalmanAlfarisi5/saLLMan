"""
Transformer implementation following Vaswani et al. 2017,
"Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

This is a from-scratch implementation emphasizing clarity over performance.
For production use, swap scaled_dot_product_attention() for
torch.nn.functional.scaled_dot_product_attention (fused FlashAttention kernel).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. Configuration  -  Paper Table 3 "base" model
# ---------------------------------------------------------------------------
@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 512        # Paper §3.1: embedding/hidden dim
    n_heads: int = 8          # Paper §3.2.2: h = 8 parallel heads
    n_layers: int = 6         # Paper §3.1: N = 6 encoder and decoder layers
    d_ff: int = 2048          # Paper §3.3: inner FFN dim (4 * d_model)
    max_len: int = 5000       # Max positions for sinusoidal PE table
    dropout: float = 0.1      # Paper §5.4: P_drop = 0.1
    pad_idx: int = 0          # Used for padding masks

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, \
            "d_model must be divisible by n_heads so each head gets d_k = d_model/h"


# ---------------------------------------------------------------------------
# 2. Positional Encoding  -  Paper §3.5
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Paper §3.5, eqs. for PE(pos, 2i) / PE(pos, 2i+1)):
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    Added to the token embeddings so the model has access to order information.
    Not learned; registered as a buffer so it moves with .to(device) but has no params.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # Numerically stable form of 1 / 10000^(2i/d_model) = exp(-2i/d_model * log(10000))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) -> broadcasts over batch

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 3. Scaled Dot-Product Attention  -  Paper §3.2.1, eq. (1)
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: Tensor,             # (B, h, T_q, d_k)
    k: Tensor,             # (B, h, T_k, d_k)
    v: Tensor,             # (B, h, T_k, d_v)
    mask: Tensor | None = None,   # broadcastable to (B, h, T_q, T_k); True = KEEP
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Paper eq. (1):   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    The sqrt(d_k) scale (§3.2.1 footnote 4) prevents dot products from growing
    large in magnitude, which would push softmax into regions with vanishing
    gradients.
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, T_q, T_k)

    if mask is not None:
        # mask == True means "attend to this position", False means "mask out"
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    out = torch.matmul(attn, v)  # (B, h, T_q, d_v)
    return out, attn


# ---------------------------------------------------------------------------
# 4. Multi-Head Attention  -  Paper §3.2.2
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Paper §3.2.2:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        head_i            = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    In practice the per-head projections W_i^Q/K/V are implemented as one
    Linear layer of shape (d_model -> d_model) and reshaped into h heads.
    Each head operates in dimension d_k = d_v = d_model / h (§3.2.2).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Fused Q, K, V projections as separate Linears for clarity.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # output projection W^O
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,           # (B, T_q, d_model)
        key: Tensor,             # (B, T_k, d_model)
        value: Tensor,           # (B, T_k, d_model)
        mask: Tensor | None = None,  # (B, 1, T_q, T_k) or (B, 1, 1, T_k), True = KEEP
    ) -> Tensor:
        B, T_q, _ = query.shape
        T_k = key.size(1)

        # 1) Linear projections, then split into heads:
        #    (B, T, d_model) -> (B, T, h, d_k) -> (B, h, T, d_k)
        q = self.W_q(query).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)

        # 2) Attention per head (broadcast mask across heads).
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.attn_dropout)

        # 3) Concat heads: (B, h, T_q, d_k) -> (B, T_q, h, d_k) -> (B, T_q, d_model)
        #    .contiguous() is required because transpose produces a non-contiguous view.
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        # 4) Final output projection W^O.
        return self.W_o(out)


# ---------------------------------------------------------------------------
# 5. Position-wise Feed-Forward  -  Paper §3.3, eq. (2)
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """
    Paper eq. (2):  FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    Applied independently to each position. Inner dim d_ff = 2048 (§3.3).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ---------------------------------------------------------------------------
# 6. Encoder Layer  -  Paper §3.1 (left half of Figure 1)
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """
    Each encoder layer (Paper §3.1):
        sublayer 1:  x + Dropout(MultiHeadSelfAttention(LayerNorm(x)))     ... see note
        sublayer 2:  x + Dropout(FFN(LayerNorm(x)))

    NOTE ON POST-LN vs PRE-LN:
    The original paper uses POST-LN: y = LayerNorm(x + Sublayer(x)).
    Most modern implementations use PRE-LN: y = x + Sublayer(LayerNorm(x)),
    which trains more stably without a learning-rate warmup.
    I've coded POST-LN below to match the paper exactly (§5.4 residual dropout).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        # Sublayer 1: Multi-head self-attention
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # Sublayer 2: Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# ---------------------------------------------------------------------------
# 7. Decoder Layer  -  Paper §3.1 (right half of Figure 1)
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """
    Each decoder layer has THREE sublayers (Paper §3.1):
        1) Masked multi-head self-attention (causal mask so position i cannot
           attend to positions > i).
        2) Multi-head cross-attention over encoder output: Q from decoder,
           K and V from encoder (§3.2.3 "encoder-decoder attention").
        3) Position-wise FFN.
    Each with residual + LayerNorm.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,                        # target embeddings (B, T_tgt, d_model)
        memory: Tensor,                   # encoder output (B, T_src, d_model)
        tgt_mask: Tensor | None = None,   # causal + padding mask for decoder self-attn
        memory_mask: Tensor | None = None,  # padding mask for cross-attn (over src)
    ) -> Tensor:
        # 1) Masked self-attention over target prefix
        attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # 2) Cross-attention: Q = decoder, K/V = encoder memory
        cross_out = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout2(cross_out))
        # 3) Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


# ---------------------------------------------------------------------------
# 8. Encoder and Decoder stacks
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
             for _ in range(cfg.n_layers)]
        )
        self.d_model = cfg.d_model

    def forward(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        # Paper §3.4: multiply embeddings by sqrt(d_model)
        x = self.embed(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
             for _ in range(cfg.n_layers)]
        )
        self.d_model = cfg.d_model

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x


# ---------------------------------------------------------------------------
# 9. Mask helpers
# ---------------------------------------------------------------------------
def make_pad_mask(seq: Tensor, pad_idx: int = 0) -> Tensor:
    """
    Returns (B, 1, 1, T) boolean mask where True means "keep / attend".
    Broadcasts across heads and query positions.
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size: int, device: torch.device) -> Tensor:
    """
    Returns (1, 1, size, size) lower-triangular boolean mask.
    Position i can attend to positions [0, i]. Paper §3.2.3 (decoder self-attn).
    """
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# 10. Full Transformer
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.generator = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)

        # Paper §3.4: tie target embedding weights with the output projection ("weight tying").
        self.generator.weight = self.decoder.embed.weight

        self._init_params()

    def _init_params(self) -> None:
        # Xavier init is what the Annotated Transformer (and common practice) uses.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_masks(self, src: Tensor, tgt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Build src padding, tgt causal+padding, and memory padding masks."""
        src_mask = make_pad_mask(src, self.cfg.pad_idx)          # (B, 1, 1, T_src)
        tgt_pad = make_pad_mask(tgt, self.cfg.pad_idx)           # (B, 1, 1, T_tgt)
        causal = make_causal_mask(tgt.size(1), tgt.device)       # (1, 1, T_tgt, T_tgt)
        tgt_mask = tgt_pad & causal                              # (B, 1, T_tgt, T_tgt)
        memory_mask = src_mask                                   # cross-attn uses src padding
        return src_mask, tgt_mask, memory_mask

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        src: (B, T_src) token ids
        tgt: (B, T_tgt) token ids — typically the right-shifted target (start with BOS)
        Returns logits of shape (B, T_tgt, tgt_vocab_size).
        """
        src_mask, tgt_mask, memory_mask = self.make_masks(src, tgt)
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return self.generator(out)  # logits; apply log_softmax in the loss


# ---------------------------------------------------------------------------
# 11. Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = TransformerConfig(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=64,      # small for debugging
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_len=100,
    )
    model = Transformer(cfg)

    # Fake batch: 2 sentences, src of length 7, tgt of length 5.
    src = torch.tensor([[5, 7, 9, 11, 13, 0, 0],
                        [4, 6, 8, 10, 12, 14, 16]])
    tgt = torch.tensor([[1, 3, 5, 7, 0],
                        [1, 2, 4, 6, 8]])
    logits = model(src, tgt)
    print(f"logits shape: {logits.shape}  (expected: [2, 5, 1000])")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")