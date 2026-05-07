"""
saLLMan Phase 2 — Modernized decoder-only Transformer.

Builds on Phase 1 (decoder_only.py) by applying the architectural improvements
that every serious post-2020 decoder-only LM uses (LLaMA, Mistral, Qwen, GPT-NeoX,
DeepSeek). Each change is annotated with the paper/reference that introduced it.

PHASE 2 CHANGES
---------------
2a. Pre-LN          x = x + sublayer(norm(x))
                    (Xiong et al. 2020, https://arxiv.org/abs/2002.04745)
2b. RMSNorm         drop the mean-centering and learnable bias from LayerNorm
                    (Zhang & Sennrich 2019, https://arxiv.org/abs/1910.07467)
2c. RoPE            rotate Q/K by position-dependent complex phase instead of
                    adding a sinusoidal embedding to the input
                    (Su et al. 2021, https://arxiv.org/abs/2104.09864)
2d. SwiGLU          gated FFN: SwiGLU(x) = (Swish(xW_1) * xW_3) W_2
                    with d_ff = 8/3 * d_model to match parameter count
                    (Shazeer 2020, https://arxiv.org/abs/2002.05202)

ADDITIONAL PRODUCTION IMPROVEMENTS (all standard in LLaMA-class models)
-----------------------------------------------------------------------
+ Final RMSNorm before lm_head           (required for Pre-LN architectures)
+ GPT-2 style init: N(0, 0.02) with     (Radford et al. 2019, GPT-2 §2.3)
  residual-projection scaling 1/sqrt(2N)
+ No bias on Linear layers              (LLaMA convention; saves params, no acc cost)
+ F.scaled_dot_product_attention        (FlashAttention kernel; ~10x less VRAM)
  This is the single most impactful change for an 8GB GPU.
+ KV-cache during generation            (O(T) instead of O(T^2) per token)

WHAT'S STILL THE SAME AS PHASE 1
--------------------------------
- Decoder-only structure (one stack of self-attn + FFN blocks)
- Causal masking
- Tied input/output embeddings
- Single vocabulary
- Adam optimizer + Noam scheduler still used in train_lm_v2.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ===========================================================================
# 1. Configuration
# ===========================================================================
@dataclass
class GPTConfigV2:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    # SwiGLU note (Shazeer 2020 §3.3): a SwiGLU FFN has 3 weight matrices instead
    # of 2, so to keep parameter count comparable to a ReLU FFN with d_ff=4*d_model,
    # we use d_ff = (2/3) * 4 * d_model = 8/3 * d_model.
    # Most implementations (LLaMA) round to a multiple of 64 for kernel efficiency.
    d_ff: int | None = None       # auto: round_to_multiple(8/3 * d_model, 64)
    max_len: int = 1024
    dropout: float = 0.0          # LLaMA-style: 0 dropout once data is sufficient
    pad_idx: int = 0
    rope_base: float = 10000.0    # standard RoPE frequency base (Su et al. 2021)
    norm_eps: float = 1e-5        # RMSNorm epsilon (LLaMA uses 1e-5/1e-6)

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0
        if self.d_ff is None:
            # Round 8/3 * d_model up to nearest multiple of 64.
            target = int(8 * self.d_model / 3)
            self.d_ff = ((target + 63) // 64) * 64


# ===========================================================================
# 2. RMSNorm  -  Phase 2b (Zhang & Sennrich 2019)
# ===========================================================================
# LayerNorm:    y = ((x - mean(x)) / sqrt(var(x) + eps)) * gamma + beta
# RMSNorm:      y =  x / sqrt(mean(x^2) + eps)            * gamma
#
# The intuition: the centering step (subtracting mean) is empirically not what's
# doing the heavy lifting — the RESCALING by RMS is. Removing the mean and the
# learnable bias makes the op cheaper and slightly more stable, with no quality
# loss. Used in LLaMA, T5, Gemma, Qwen, Mistral.
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))   # gamma; init to 1 so initial behavior is identity-ish
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Compute in fp32 even when x is bf16/fp16 — variance estimation is the
        # one place norms can lose precision. LLaMA does this dance too.
        dtype = x.dtype
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f32 * rms).to(dtype) * self.weight


# ===========================================================================
# 3. RoPE  -  Phase 2c (Su et al. 2021)
# ===========================================================================
# RoPE replaces "add positional embedding to token embedding" with "rotate the
# query and key vectors of EACH ATTENTION OPERATION by a position-dependent
# angle". Pairs of dimensions in q and k are treated as 2D vectors in the
# complex plane; position m rotates them by m*theta_i where theta_i depends on
# the dimension pair.
#
# Why this is better than sinusoidal PE:
#   1. It's *relative* — the dot product q_m . k_n depends only on (m-n), not
#      on absolute m, n. This is exactly what attention should be.
#   2. It generalizes to longer sequences than seen during training (with some
#      caveats — see "RoPE scaling" / position interpolation).
#   3. It composes with FlashAttention because it's a pre-attention op on Q/K.
#
# The rotation: for dimension pair (2i, 2i+1) at position m,
#   [q_{2i}, q_{2i+1}]  ->  [q_{2i} cos(m*θ_i) - q_{2i+1} sin(m*θ_i),
#                            q_{2i} sin(m*θ_i) + q_{2i+1} cos(m*θ_i)]
# where θ_i = base ^ (-2i / d_head).
def precompute_rope_cache(d_head: int, max_seq_len: int, base: float,
                          device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Returns (cos, sin) tensors of shape (max_seq_len, d_head/2) that can be
    indexed by [position]. d_head must be even (it is, since d_model % n_heads == 0
    and we choose d_model divisible by 2*n_heads).
    """
    assert d_head % 2 == 0, "d_head must be even for RoPE"
    # θ_i = base^(-2i / d_head),  i = 0, 1, ..., d_head/2 - 1
    theta = base ** (-torch.arange(0, d_head, 2, dtype=torch.float32, device=device) / d_head)
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    # Outer product → (max_seq_len, d_head/2)
    angles = torch.outer(positions, theta)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply RoPE rotation to x.
    x:   (B, n_heads, T, d_head)
    cos: (T, d_head/2)
    sin: (T, d_head/2)

    The rotation pairs dimensions (0,1), (2,3), ... and rotates each pair.
    Implementation note: we view x as having a final pair-dimension of 2 to
    do the rotation, then reshape back. Equivalent to the complex-number
    formulation but stays in real arithmetic.
    """
    # Split last dim into pairs: (..., d_head) -> (..., d_head/2, 2)
    x1 = x[..., 0::2]   # (B, h, T, d_head/2)  — even-indexed dims
    x2 = x[..., 1::2]   # (B, h, T, d_head/2)  — odd-indexed dims

    # cos/sin: (T, d_head/2) -> (1, 1, T, d_head/2) for broadcasting.
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Rotate each (x1, x2) pair: (x1', x2') = (x1*cos - x2*sin, x1*sin + x2*cos)
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos

    # Reinterleave back to original layout: stack on a new pair-dim then flatten.
    out = torch.stack([rx1, rx2], dim=-1)        # (..., d_head/2, 2)
    return out.flatten(-2)                       # (..., d_head)


# ===========================================================================
# 4. Multi-head self-attention with RoPE + FlashAttention
# ===========================================================================
# Differences from Phase 1's MultiHeadAttention:
#   - bias=False on all linears (LLaMA convention)
#   - RoPE is applied to Q and K before attention
#   - F.scaled_dot_product_attention used in place of manual matmul+softmax.
#     PyTorch dispatches this to FlashAttention-2 when shapes/dtypes allow.
#     This is THE big speed/memory win.
#   - KV cache support for generation (past_kv argument)
#   - is_causal=True passed to SDPA → no need to materialize a causal mask
#     (FlashAttention handles it internally)
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfigV2) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        # Fused QKV projection — slightly faster than three separate Linears
        # because it's one matmul. Standard in nanoGPT, GPT-NeoX.
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: Tensor,                         # (B, T, d_model)
        cos: Tensor,                       # (T, d_head/2)  RoPE cache slice
        sin: Tensor,                       # (T, d_head/2)
        past_kv: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        B, T, _ = x.shape

        # Project to Q, K, V in one matmul, then split.
        qkv = self.qkv(x)                                   # (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)                      # each (B, T, d_model)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K (NOT V — V carries content, not position).
        # When using KV-cache, we need to rotate the new K/Q at the *correct*
        # absolute positions (which start after the cached length).
        if past_kv is not None:
            past_len = past_kv[0].size(2)
            # Slice cos/sin starting from past_len. The caller passed the full
            # cache up to past_len + T; we want the new positions only.
            cos_new = cos[past_len : past_len + T]
            sin_new = sin[past_len : past_len + T]
        else:
            cos_new, sin_new = cos, sin
        q = apply_rope(q, cos_new, sin_new)
        k = apply_rope(k, cos_new, sin_new)

        # KV cache: concatenate previous K, V (already rotated when stored) with new ones.
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)               # (B, h, past_T + T, d_head)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)

        # FlashAttention via SDPA. is_causal=True applies a causal mask internally.
        # IMPORTANT: when past_kv is non-empty we are doing single-step decoding
        # where T_q < T_k (typically T_q=1). is_causal must be False in that case
        # because the new query attends to ALL cached keys (which are all "past"
        # from its perspective).
        is_causal = past_kv is None
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                                    # (B, h, T, d_head)

        # Concat heads and project.
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), new_kv


# ===========================================================================
# 5. SwiGLU FFN  -  Phase 2d (Shazeer 2020)
# ===========================================================================
# Standard ReLU FFN:    FFN(x) = ReLU(x W_1) W_2
# SwiGLU FFN:           FFN(x) = (Swish(x W_1) * (x W_3)) W_2
#                                ↑ gate              ↑ value
# Swish(x) = x * sigmoid(x)  (also called SiLU)
#
# The "gating" — element-wise multiply between Swish(x W_1) and (x W_3) —
# is the actual win. Each FFN dimension can be independently up- or
# down-weighted depending on the gate. Shazeer 2020 ablates several variants
# (GLU, ReGLU, GEGLU, SwiGLU) and SwiGLU consistently does best.
#
# Cost: 3 matrices instead of 2. We compensate by setting d_ff = 8/3 * d_model
# instead of 4 * d_model, keeping total FFN parameters constant.
class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfigV2) -> None:
        super().__init__()
        # In LLaMA terminology: w1 = gate_proj, w3 = up_proj, w2 = down_proj
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # value
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)   # down

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ===========================================================================
# 6. Pre-LN block  -  Phase 2a
# ===========================================================================
# Post-LN (Phase 1):    x = LN(x + sublayer(x))
# Pre-LN (Phase 2):     x = x + sublayer(LN(x))
#
# Why this matters: in Post-LN, the residual stream gets normalized at every
# layer, which means signal magnitudes from early layers can be effectively
# wiped out as you go deeper. Pre-LN preserves a clean additive residual path
# from input to output, which gives stable gradients and removes the need for
# the warmup-heavy Noam schedule. Xiong et al. 2020 showed Pre-LN trains
# without warmup; LLaMA, GPT-NeoX, Mistral all use Pre-LN.
class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfigV2) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_kv: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # Sublayer 1: pre-norm → attention → residual add
        attn_out, new_kv = self.attn(self.attn_norm(x), cos, sin, past_kv=past_kv)
        x = x + attn_out
        # Sublayer 2: pre-norm → FFN → residual add
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


# ===========================================================================
# 7. The model
# ===========================================================================
class GPTv2(nn.Module):
    def __init__(self, cfg: GPTConfigV2) -> None:
        super().__init__()
        self.cfg = cfg

        # No more PositionalEncoding module — RoPE handles position info inside attention.
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        # Final norm is REQUIRED for Pre-LN — without it the residual stream is
        # un-normalized when it hits lm_head and outputs can blow up.
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tied embeddings — same trick as Phase 1.
        self.lm_head.weight = self.embed.weight

        # Precompute RoPE cos/sin once. Registered as buffer so it moves to GPU.
        cos, sin = precompute_rope_cache(
            d_head=cfg.d_model // cfg.n_heads,
            max_seq_len=cfg.max_len,
            base=cfg.rope_base,
            device=torch.device("cpu"),
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_params()

    # GPT-2 init (Radford et al. 2019, §2.3): N(0, 0.02), with residual-feeding
    # projections additionally scaled by 1/sqrt(2N) to prevent activation growth
    # along the residual stream as depth N increases.
    #
    # The "residual-feeding" projections are the ones that write INTO the
    # residual stream: out_proj in attention, and w2 in SwiGLU. Embedding is
    # initialized at 0.02 too. Norms initialize gamma=1, which RMSNorm already
    # does in its __init__.
    def _init_params(self) -> None:
        std = 0.02
        scaled_std = std / math.sqrt(2 * self.cfg.n_layers)

        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue   # skip biases (none here) and norm weights (init=1, fine)
            # Residual-feeding linears get the scaled init.
            if name.endswith("out_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=scaled_std)
            else:
                nn.init.normal_(p, mean=0.0, std=std)

        # Embedding (and tied lm_head) gets standard 0.02.
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: Tensor,                            # (B, T)
        past_kvs: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """
        Returns (logits, new_past_kvs).
        During training, pass past_kvs=None and ignore the returned cache.
        During generation, pass the previous cache and reuse the new one.
        """
        B, T = input_ids.shape
        # When using KV cache we need RoPE positions starting at past_len.
        past_len = past_kvs[0][0].size(2) if past_kvs is not None else 0
        assert past_len + T <= self.cfg.max_len, "context exceeds max_len"

        # Embedding. We DON'T multiply by sqrt(d_model) here — that scaling was
        # important for the additive sinusoidal PE balance in Vaswani 2017.
        # With RoPE there's no PE to balance against. LLaMA et al. don't scale.
        x = self.embed(input_ids)                                      # (B, T, d_model)

        # Slice RoPE cache for this forward pass.
        cos = self.rope_cos[: past_len + T]
        sin = self.rope_sin[: past_len + T]

        new_past_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, cos, sin, past_kv=past_kv)
            new_past_kvs.append(new_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)                                       # (B, T, vocab_size)
        return logits, new_past_kvs

    # --- generation with KV cache --------------------------------------------
    # KV cache turns generation from O(T^2) per token to O(T):
    #   - First call: process the entire prompt, store K/V for every layer.
    #   - Subsequent calls: feed in only the NEW token (T=1), re-using all
    #     cached K/V. Each layer concatenates new K/V with cached ones.
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """
        Greedy / temperature / top-k sampled generation with KV cache.
        temperature=0 (or top_k=1) → deterministic greedy.
        """
        self.eval()

        # Initial pass: feed the full prompt, build the cache.
        logits, past_kvs = self(input_ids)                  # (B, T_prompt, V)
        next_logits = logits[:, -1, :]                      # (B, V)

        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                logits_t = next_logits / max(temperature, 1e-6)
                if top_k is not None and top_k < logits_t.size(-1):
                    topk_vals, _ = torch.topk(logits_t, top_k, dim=-1)
                    threshold = topk_vals[:, -1:].expand_as(logits_t)
                    logits_t = torch.where(logits_t < threshold,
                                           torch.full_like(logits_t, float("-inf")),
                                           logits_t)
                probs = F.softmax(logits_t, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break

            # Subsequent passes: feed only the new token, reuse cache.
            logits, past_kvs = self(next_id, past_kvs=past_kvs)
            next_logits = logits[:, -1, :]

        return input_ids


# ===========================================================================
# 8. Smoke test
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfigV2(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_len=128,
    )
    print(f"Auto-computed d_ff: {cfg.d_ff}  (= 8/3 * {cfg.d_model} rounded to multiple of 64)")

    model = GPTv2(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")

    # Forward pass
    input_ids = torch.tensor([
        [1, 5, 7, 9, 11, 13, 15, 0, 0, 0],
        [1, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    ])
    logits, _ = model(input_ids)
    print(f"logits shape: {tuple(logits.shape)}  (expected: (2, 10, 1000))")

    # Generation with KV cache
    prompt = torch.tensor([[1, 5, 7]])
    out = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    print(f"greedy generated: {out.tolist()}")

    out = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=20)
    print(f"sampled (top-k=20): {out.tolist()}")