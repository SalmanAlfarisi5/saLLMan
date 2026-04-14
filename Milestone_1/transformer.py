"""
================================================================================
ATTENTION IS ALL YOU NEED — Full Annotated Implementation
================================================================================
Paper: "Attention Is All You Need" (Vaswani et al., 2017)
Link:  https://arxiv.org/abs/1706.03762

This file implements every core component of the original Transformer
architecture from scratch using PyTorch. Each section maps directly to a
section/equation in the paper and includes inline explanations.

Requires: Python 3.10+, PyTorch 2.x
    pip install torch --break-system-packages

Architecture Overview (Figure 1 in the paper):
───────────────────────────────────────────────
    ┌──────────────────────────────────────────┐
    │              LINEAR + SOFTMAX            │  ← Final projection to vocab
    │         (Output Probabilities)           │
    └────────────────────┬─────────────────────┘
                         │
    ┌────────────────────┴─────────────────────┐
    │           DECODER STACK (N×)             │
    │  ┌─────────────────────────────────────┐ │
    │  │  Feed-Forward Network               │ │
    │  │  Add & Norm                         │ │
    │  │  Cross-Attention (to encoder out)   │ │
    │  │  Add & Norm                         │ │
    │  │  Masked Self-Attention              │ │
    │  │  Add & Norm                         │ │
    │  └─────────────────────────────────────┘ │
    └────────────────────┬─────────────────────┘
                         │
    ┌────────────────────┴─────────────────────┐
    │           ENCODER STACK (N×)             │
    │  ┌─────────────────────────────────────┐ │
    │  │  Feed-Forward Network               │ │
    │  │  Add & Norm                         │ │
    │  │  Multi-Head Self-Attention          │ │
    │  │  Add & Norm                         │ │
    │  └─────────────────────────────────────┘ │
    └────────────────────┬─────────────────────┘
                         │
    ┌────────────────────┴─────────────────────┐
    │  Input Embedding + Positional Encoding   │
    └──────────────────────────────────────────┘
================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Section 3.2.2 — Scaled Dot-Product Attention
# ==============================================================================
# This is the fundamental building block of the Transformer.
#
# Given queries (Q), keys (K), and values (V), attention computes:
#
#   Attention(Q, K, V) = softmax(Q K^T / √d_k) V        — Equation (1)
#
# WHY SCALING?
#   When d_k is large, the dot products Q·K grow in magnitude, pushing the
#   softmax into regions with extremely small gradients (saturation).
#   Dividing by √d_k keeps the variance of the dot products at ~1,
#   preventing gradient vanishing.
#
# WHY THIS FUNCTION?
#   Unlike RNN attention (Bahdanau, 2014) which uses a learned alignment
#   network, dot-product attention is much faster because it can be
#   computed entirely with optimised matrix-multiply hardware (GPUs/TPUs).
# ==============================================================================

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        query:   (batch, heads, seq_len_q, d_k)
        key:     (batch, heads, seq_len_k, d_k)
        value:   (batch, heads, seq_len_k, d_v)
        mask:    Broadcastable mask. 0/False positions are *allowed*,
                 True positions are *blocked* (filled with -inf before softmax).
        dropout: Optional dropout applied to the attention weights.

    Returns:
        output:  (batch, heads, seq_len_q, d_v)
        attn_weights: (batch, heads, seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)

    # Step 1: Compute raw attention scores  →  Q K^T
    # Shape: (batch, heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: Scale by √d_k to stabilise gradients
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask (used for padding and causal/autoregressive decoding)
    # Masked positions get -inf so that softmax assigns them ~0 probability.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 4: Softmax along the key dimension → attention weights sum to 1
    attn_weights = F.softmax(scores, dim=-1)

    # Step 5: Optional dropout on attention weights (Section 5.4 in the paper)
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # Step 6: Weighted sum of values
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


# ==============================================================================
# Section 3.2.2 — Multi-Head Attention
# ==============================================================================
# Instead of performing a single attention function with d_model-dimensional
# keys, values, and queries, the paper finds it beneficial to linearly project
# Q, K, V  →  h times with different learned projections to d_k, d_k, d_v dims.
#
#   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O      — Equation (5)
#   where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)       — Equation (4)
#
# WHY MULTIPLE HEADS?
#   Each head can learn to attend to different aspects of the input:
#   - Head 1 might focus on syntactic dependencies (subject-verb agreement)
#   - Head 2 might focus on positional proximity
#   - Head 3 might capture coreference resolution
#   Multi-head attention allows the model to jointly attend to information
#   from different representation subspaces at different positions.
#
# The paper uses h=8 heads with d_k = d_v = d_model/h = 64.
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism (Section 3.2.2).

    This module is used in three different ways in the Transformer:
      1. Encoder self-attention — Q=K=V come from encoder input
      2. Decoder self-attention — Q=K=V come from decoder input (masked)
      3. Encoder-decoder attention — Q from decoder, K=V from encoder output
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Learned linear projections: W_Q, W_K, W_V, W_O
        # Each is (d_model, d_model) but conceptually splits into h × (d_model, d_k)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # Output projection

        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # Stored for visualisation

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask:  (batch, 1, 1, seq_len_k) or (batch, 1, seq_len_q, seq_len_k)

        Returns:
            output: (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # ── Step 1: Linear projections ──────────────────────────────────────
        # Project and reshape: (batch, seq_len, d_model) → (batch, heads, seq_len, d_k)
        # This splits d_model into num_heads separate d_k-dimensional subspaces.
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # ── Step 2: Apply scaled dot-product attention to all heads in parallel ─
        # Thanks to the batch dimension including heads, this is done in one
        # matmul — no loop over heads needed.
        x, self.attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # ── Step 3: Concatenate heads and apply output projection ───────────
        # (batch, heads, seq_len, d_k) → (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear: W^O in Equation (5)
        return self.W_o(x)


# ==============================================================================
# Section 3.3 — Position-wise Feed-Forward Networks
# ==============================================================================
# Applied to each position separately and identically:
#
#   FFN(x) = max(0, x W_1 + b_1) W_2 + b_2                   — Equation (2)
#
# This is two linear transformations with a ReLU activation in between.
# The inner layer has dimensionality d_ff = 2048 (4× the model dimension).
#
# WHY THIS DESIGN?
#   - The attention layer mixes information *across* positions.
#   - The FFN processes each position *independently*, acting as a learned
#     nonlinear transformation — essentially a 1×1 convolution.
#   - Together they give the model both cross-position communication (attention)
#     and per-position computation (FFN).
#   - The expansion to 4× width and back creates a bottleneck that forces
#     the model to learn compressed, useful representations.
# ==============================================================================

class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network (Section 3.3)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)     # W_1: expand
        self.linear2 = nn.Linear(d_ff, d_model)      # W_2: compress back
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ==============================================================================
# Section 3.1 — Residual Connection + Layer Normalization
# ==============================================================================
# Every sub-layer (attention or FFN) in the Transformer is wrapped with:
#
#   LayerNorm(x + Sublayer(x))                                 — Equation (implicit)
#
# RESIDUAL CONNECTIONS (He et al., 2016):
#   Allow gradients to flow directly through the network, enabling training
#   of very deep models (the paper stacks N=6 layers). Without residuals,
#   gradient signal degrades exponentially with depth.
#
# LAYER NORMALIZATION (Ba et al., 2016):
#   Normalises across the feature dimension (d_model) for each position
#   independently. Unlike BatchNorm, it doesn't depend on batch statistics,
#   making it suitable for variable-length sequences and small batch sizes.
#
# NOTE: The paper uses "post-norm" (norm after residual addition).
#   Many modern implementations use "pre-norm" (norm before the sublayer)
#   for more stable training. We follow the original paper here.
# ==============================================================================

class SublayerConnection(nn.Module):
    """Residual connection followed by layer normalisation (post-norm)."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_fn) -> torch.Tensor:
        """
        Args:
            x:           Input tensor (batch, seq_len, d_model)
            sublayer_fn: A callable (the sub-layer to apply, e.g. attention or FFN)
        """
        return self.norm(x + self.dropout(sublayer_fn(x)))


# ==============================================================================
# Section 3.4 — Positional Encoding
# ==============================================================================
# Since the Transformer contains no recurrence or convolution, it has no
# inherent notion of token order. Positional encodings are added to the
# input embeddings to inject information about the position of each token.
#
# The paper uses sinusoidal functions of different frequencies:
#
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))            — Equation (3)
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))            — Equation (4)
#
# WHY SINUSOIDAL?
#   1. Unique encoding: Each position gets a unique pattern.
#   2. Bounded: Values stay in [-1, 1] regardless of sequence length.
#   3. Relative positions: For any fixed offset k, PE(pos+k) can be
#      expressed as a linear function of PE(pos), so the model can
#      easily learn to attend by relative positions.
#   4. Extrapolation: Unlike learned embeddings, sinusoidal encodings
#      can generalise to sequences longer than those seen during training.
#
# The authors also experimented with learned positional embeddings and
# found nearly identical results (Table 3, row E).
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (Section 3.4)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # Compute the division term: 10000^(2i/d_model)
        # We use the log-space trick for numerical stability:
        #   10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        # Add batch dimension and register as buffer (not a learnable parameter)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)  — token embeddings
        Returns:
            (batch, seq_len, d_model)  — embeddings + positional encoding
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ==============================================================================
# Section 3.4 — Token Embeddings
# ==============================================================================
# The paper uses learned embeddings to convert input/output tokens to vectors
# of dimension d_model. The embedding weights are multiplied by √d_model.
#
# WHY √d_model SCALING?
#   The positional encoding values are in [-1, 1]. Without scaling, the
#   embedding values (initialised with std ≈ 1/√d_model by default) would
#   be much smaller than the positional encodings, drowning out the token
#   identity information. Multiplying by √d_model brings both signals to
#   a comparable scale.
# ==============================================================================

class TokenEmbedding(nn.Module):
    """Learned token embedding with √d_model scaling."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


# ==============================================================================
# Section 3.1 — Encoder Layer (single block)
# ==============================================================================
# Each encoder layer has two sub-layers:
#   1. Multi-head self-attention
#   2. Position-wise feed-forward network
# Each sub-layer is wrapped with a residual connection and layer norm.
# ==============================================================================

class EncoderLayer(nn.Module):
    """Single Encoder Layer (one block of Figure 1, left side)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:        (batch, seq_len, d_model)
            src_mask: (batch, 1, 1, seq_len) — padding mask for source
        """
        # Sub-layer 1: Multi-head self-attention
        # Q = K = V = x  (self-attention: the sequence attends to itself)
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, src_mask))

        # Sub-layer 2: Position-wise FFN
        x = self.sublayer2(x, self.feed_forward)

        return x


# ==============================================================================
# Section 3.1 — Decoder Layer (single block)
# ==============================================================================
# Each decoder layer has THREE sub-layers:
#   1. Masked multi-head self-attention (prevents attending to future positions)
#   2. Multi-head cross-attention (attends to encoder output)
#   3. Position-wise feed-forward network
#
# CAUSAL MASKING (Sub-layer 1):
#   During training, the decoder sees the entire target sequence at once
#   (teacher forcing). The causal mask ensures position i can only attend
#   to positions ≤ i, preserving the autoregressive property — the model
#   cannot "cheat" by looking at future tokens.
#
# CROSS-ATTENTION (Sub-layer 2):
#   Queries come from the decoder, but keys and values come from the
#   encoder output. This is how the decoder reads the source sequence.
# ==============================================================================

class DecoderLayer(nn.Module):
    """Single Decoder Layer (one block of Figure 1, right side)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (batch, tgt_len, d_model) — decoder input
            encoder_output: (batch, src_len, d_model) — from the encoder stack
            src_mask:       Padding mask for source sequence
            tgt_mask:       Combined causal + padding mask for target sequence
        """
        # Sub-layer 1: Masked self-attention (causal)
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # Sub-layer 2: Cross-attention (decoder queries → encoder keys/values)
        x = self.sublayer2(x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask))

        # Sub-layer 3: Position-wise FFN
        x = self.sublayer3(x, self.feed_forward)

        return x


# ==============================================================================
# Section 3.1 — Encoder Stack
# ==============================================================================
# The encoder is a stack of N=6 identical layers. The output of the final
# encoder layer is used as keys and values for cross-attention in every
# decoder layer.
# ==============================================================================

class Encoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(self, layer: EncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # Final layer norm


# ==============================================================================
# Section 3.1 — Decoder Stack
# ==============================================================================

class Decoder(nn.Module):
    """Stack of N decoder layers."""

    def __init__(self, layer: DecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# ==============================================================================
# Section 3.4 — Output Linear Layer + Softmax (Generator)
# ==============================================================================
# The final linear layer projects the decoder output back to the vocabulary
# size, and a log-softmax produces token probabilities.
#
# WEIGHT TYING (Section 3.4):
#   The paper shares the same weight matrix between the two embedding layers
#   (source and target) and the pre-softmax linear transformation.
#   This reduces parameters and has been shown to improve performance
#   (Press & Wolf, 2017).
# ==============================================================================

class Generator(nn.Module):
    """Linear + log-softmax to produce next-token probabilities."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.projection(x), dim=-1)


# ==============================================================================
# Full Transformer Model — Putting It All Together
# ==============================================================================
# The complete encoder-decoder Transformer as described in the paper.
# ==============================================================================

class Transformer(nn.Module):
    """
    The full Transformer model (Figure 1).

    Data flow:
      1. Source tokens → src_embedding + positional_encoding → Encoder stack
      2. Target tokens → tgt_embedding + positional_encoding → Decoder stack
         (Decoder also receives encoder output for cross-attention)
      3. Decoder output → Generator (linear + softmax) → token probabilities
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── Embeddings ──────────────────────────────────────────────────────
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # ── Encoder & Decoder Stacks ────────────────────────────────────────
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, num_layers)
        self.decoder = Decoder(decoder_layer, num_layers)

        # ── Output Projection ──────────────────────────────────────────────
        self.generator = Generator(d_model, tgt_vocab_size)

        # ── Parameter Initialisation (Section 5.4) ─────────────────────────
        # Xavier uniform init is used for all parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialise parameters using Xavier uniform (Glorot) initialisation."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the encoder (used during both training and inference)."""
        x = self.positional_encoding(self.src_embedding(src))
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder (used during both training and inference)."""
        x = self.positional_encoding(self.tgt_embedding(tgt))
        return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full forward pass: encode source, decode target, generate probabilities.

        Args:
            src:      (batch, src_len)  — source token indices
            tgt:      (batch, tgt_len)  — target token indices
            src_mask: Padding mask for source
            tgt_mask: Causal + padding mask for target

        Returns:
            (batch, tgt_len, tgt_vocab_size) — log-probabilities over vocabulary
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.generator(decoder_output)


# ==============================================================================
# Mask Utilities
# ==============================================================================
# Two types of masks are needed:
#   1. PADDING MASK: Ignores <pad> tokens so they don't affect attention.
#   2. CAUSAL MASK:  Prevents the decoder from attending to future positions.
# ==============================================================================

def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create a padding mask.

    Args:
        seq:     (batch, seq_len) — token indices
        pad_idx: Index of the <pad> token

    Returns:
        (batch, 1, 1, seq_len) — broadcastable mask for multi-head attention.
        1 = keep, 0 = mask out.
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size: int) -> torch.Tensor:
    """
    Create a causal (look-ahead) mask for autoregressive decoding.

    Returns an upper-triangular matrix of zeros (blocked) with ones on
    and below the diagonal (allowed).

    Args:
        size: Length of the target sequence

    Returns:
        (1, size, size) — broadcastable causal mask.
    """
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0)  # Lower-triangular
    return mask  # 1 = attend, 0 = block


def create_target_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Combine padding mask and causal mask for the decoder.

    The result blocks both:
      - Attention to padding tokens
      - Attention to future tokens
    """
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)  # (batch, 1, 1, tgt_len)
    tgt_len = tgt.size(1)
    causal_mask = create_causal_mask(tgt_len).to(tgt.device)  # (1, tgt_len, tgt_len)
    # Combine: both conditions must be true to attend
    return tgt_pad_mask & causal_mask.unsqueeze(0).bool()


# ==============================================================================
# Section 5.3 — Learning Rate Schedule (Noam Schedule)
# ==============================================================================
# The paper uses the Adam optimiser with a custom learning rate schedule:
#
#   lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
#
# This increases the learning rate linearly for the first `warmup_steps`
# training steps, then decreases it proportionally to the inverse square
# root of the step number.
#
# WHY WARMUP?
#   Early in training, the model parameters are random and gradients are
#   noisy. A small learning rate prevents large, destructive updates.
#   After warmup, the parameters are more stable and can tolerate larger
#   steps. The decay then prevents overshooting as training converges.
# ==============================================================================

class NoamScheduler:
    """Learning rate scheduler from Section 5.3."""

    def __init__(self, d_model: int, warmup_steps: int, optimizer: torch.optim.Optimizer):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self._step = 0

    def step(self):
        """Update learning rate and advance the step counter."""
        self._step += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _compute_lr(self) -> float:
        step = self._step
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))


# ==============================================================================
# Section 5.4 — Label Smoothing
# ==============================================================================
# The paper uses label smoothing with ε_ls = 0.1. Instead of training on
# hard one-hot targets, the target distribution becomes:
#
#   y_smooth = (1 - ε) · one_hot(target) + ε / vocab_size
#
# WHY?
#   Hard targets make the model overconfident. Label smoothing hurts
#   perplexity (the model is less certain) but improves accuracy and
#   BLEU score because it encourages the model to be more conservative
#   and spread probability mass more evenly — a form of regularisation.
# ==============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss using KL Divergence (Section 5.4).

    Instead of cross-entropy against hard targets, we minimise
    KL divergence against smoothed targets.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (batch * seq_len, vocab_size) — log-probabilities
            target: (batch * seq_len,) — ground-truth token indices
        """
        # Create smoothed target distribution
        true_dist = torch.full_like(pred, self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0  # Never predict padding

        # Zero out rows where target is padding
        mask = (target == self.pad_idx).nonzero(as_tuple=False)
        if mask.dim() > 0 and mask.size(0) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        return self.criterion(pred, true_dist)


# ==============================================================================
# Model Factory — Build with Paper Hyperparameters
# ==============================================================================

def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 2048,
    max_len: int = 5000,
    dropout: float = 0.1,
) -> Transformer:
    """
    Build a Transformer with the paper's default hyperparameters (Table 3).

    Base model:   d_model=512, h=8,  N=6, d_ff=2048, dropout=0.1  (~65M params)
    Big model:    d_model=1024, h=16, N=6, d_ff=4096, dropout=0.3 (~213M params)
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
    )
    return model


# ==============================================================================
# Greedy Decoding (Inference)
# ==============================================================================
# At inference time, the decoder generates one token at a time:
#   1. Encode the entire source sequence once.
#   2. Start with <bos> (beginning of sentence) token.
#   3. Feed current target sequence into decoder.
#   4. Take the argmax of the last position's output distribution.
#   5. Append the predicted token and repeat until <eos> or max_len.
#
# NOTE: This is greedy decoding. The paper reports BLEU scores using
#        beam search (beam size 4, length penalty α=0.6). Beam search
#        maintains multiple candidate sequences and picks the best.
# ==============================================================================

@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    bos_idx: int,
    eos_idx: int,
) -> torch.Tensor:
    """
    Greedy autoregressive decoding.

    Args:
        model:    Trained Transformer
        src:      (1, src_len)  — source token indices
        src_mask: Padding mask for source
        max_len:  Maximum tokens to generate
        bos_idx:  Beginning-of-sentence token index
        eos_idx:  End-of-sentence token index

    Returns:
        (1, generated_len) — generated token indices including <bos>
    """
    model.eval()
    device = src.device

    # Step 1: Encode the source sequence (done once)
    encoder_output = model.encode(src, src_mask)

    # Step 2: Start with <bos> token
    ys = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        # Step 3: Create causal mask for current target length
        tgt_mask = create_causal_mask(ys.size(1)).to(device)

        # Step 4: Decode
        decoder_output = model.decode(ys, encoder_output, src_mask, tgt_mask)

        # Step 5: Get log-probs for the last position only
        log_probs = model.generator(decoder_output[:, -1, :])

        # Step 6: Greedy — pick highest probability token
        next_token = log_probs.argmax(dim=-1, keepdim=True)

        # Step 7: Append to generated sequence
        ys = torch.cat([ys, next_token], dim=1)

        # Step 8: Stop if <eos> is generated
        if next_token.item() == eos_idx:
            break

    return ys


# ==============================================================================
# Demo — Smoke Test with Synthetic Data
# ==============================================================================
# This demo creates a simple copy task: the model must learn to copy
# the source sequence to the target. This verifies all components work
# correctly and the model can overfit a trivial task.
# ==============================================================================

def demo():
    """Run a quick smoke test: learn to copy a sequence."""
    print("=" * 70)
    print("TRANSFORMER SMOKE TEST — Copy Task")
    print("=" * 70)

    # ── Hyperparameters ─────────────────────────────────────────────────
    VOCAB_SIZE = 12        # 0=pad, 1=bos, 2=eos, 3-11=tokens
    D_MODEL = 128          # Small for demo
    NUM_HEADS = 4
    NUM_LAYERS = 2
    D_FF = 256
    BATCH_SIZE = 64
    SEQ_LEN = 5
    EPOCHS = 300
    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Build model ─────────────────────────────────────────────────────
    model = build_transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=0.0,  # No dropout for tiny demo
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # ── Generate synthetic copy data ────────────────────────────────────
    # Source: random tokens from [3, VOCAB_SIZE)
    # Target: same as source, prefixed with <bos>
    def generate_batch():
        src = torch.randint(3, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        tgt_input = torch.cat([torch.full((BATCH_SIZE, 1), BOS_IDX, device=device), src], dim=1)
        tgt_output = torch.cat([src, torch.full((BATCH_SIZE, 1), EOS_IDX, device=device)], dim=1)
        return src, tgt_input, tgt_output

    # ── Training loop ───────────────────────────────────────────────────
    criterion = LabelSmoothingLoss(VOCAB_SIZE, pad_idx=PAD_IDX, smoothing=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        src, tgt_input, tgt_output = generate_batch()

        src_mask = create_padding_mask(src, PAD_IDX)
        tgt_mask = create_target_mask(tgt_input, PAD_IDX)

        log_probs = model(src, tgt_input, src_mask, tgt_mask)

        # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len,)
        loss = criterion(
            log_probs.contiguous().view(-1, VOCAB_SIZE),
            tgt_output.contiguous().view(-1),
        )
        # Normalise by number of non-pad tokens
        n_tokens = (tgt_output != PAD_IDX).sum().item()
        loss = loss / n_tokens

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  |  Loss: {loss.item():.4f}")

    # ── Test with greedy decoding ───────────────────────────────────────
    print("\n" + "-" * 70)
    print("GREEDY DECODING TEST")
    print("-" * 70)
    model.eval()
    test_src = torch.randint(3, VOCAB_SIZE, (1, SEQ_LEN), device=device)
    src_mask = create_padding_mask(test_src, PAD_IDX)
    generated = greedy_decode(model, test_src, src_mask, max_len=SEQ_LEN + 2, bos_idx=BOS_IDX, eos_idx=EOS_IDX)

    print(f"  Source:    {test_src[0].tolist()}")
    print(f"  Generated: {generated[0].tolist()}")
    print(f"  (Expected: [1] + source + [2])")

    src_list = test_src[0].tolist()
    gen_list = generated[0].tolist()
    # Check if the generated tokens (excluding BOS and EOS) match source
    gen_core = [t for t in gen_list if t not in (BOS_IDX, EOS_IDX)]
    match = gen_core == src_list
    print(f"  Match: {'✓ PASS' if match else '✗ FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    demo()
