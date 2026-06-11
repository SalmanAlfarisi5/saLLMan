"""
saLLMan Phase 3 — Adds gradient checkpointing to the Phase 2 model.

Same architecture as decoder_only_v2.py (Pre-LN + RMSNorm + RoPE + SwiGLU +
SDPA + KV-cache). The only addition: a `gradient_checkpointing` config flag
that, when enabled, recomputes activations during the backward pass instead
of storing them.

Why gradient checkpointing matters for an 8GB GPU
-------------------------------------------------
Activation memory scales as O(n_layers * batch * seq_len * d_model). At 46.7M
params with d_model=512, n_layers=12, block_size=1024, batch=4, this can
easily blow past 6GB just for activations. Checkpointing trades ~30% extra
compute for a roughly 4-5x reduction in activation memory by storing only
layer inputs and recomputing intermediates during backward.

Reference: Chen et al. 2016, "Training Deep Nets with Sublinear Memory Cost"
https://arxiv.org/abs/1604.06174

Usage
-----
    cfg = GPTConfigV3(vocab_size=..., gradient_checkpointing=True)
    model = GPTv3(cfg)

For inference / generation, gradient_checkpointing has no effect (we're in
no_grad anyway), so it's safe to leave on always.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# Ensure project root is on sys.path so phase2 is importable from anywhere.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse Phase 2 building blocks — only the top-level model class changes.
from model_v2 import (
    GPTConfigV2,
    RMSNorm,
    TransformerBlock,
    precompute_rope_cache,
)


@dataclass
class GPTConfigV3(GPTConfigV2):
    # New in v3:
    gradient_checkpointing: bool = False


class GPTv3(nn.Module):
    """
    Identical to GPTv2 except:
      - cfg type is GPTConfigV3
      - if cfg.gradient_checkpointing is True AND we're in training mode,
        each block is wrapped in torch.utils.checkpoint during forward.
    """

    def __init__(self, cfg: GPTConfigV3) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        cos, sin = precompute_rope_cache(
            d_head=cfg.d_model // cfg.n_heads,
            max_seq_len=cfg.max_len,
            base=cfg.rope_base,
            device=torch.device("cpu"),
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_params()

    def _init_params(self) -> None:
        std = 0.02
        scaled_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue
            if name.endswith("out_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=scaled_std)
            else:
                nn.init.normal_(p, mean=0.0, std=std)
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: Tensor,
        past_kvs: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        B, T = input_ids.shape
        past_len = past_kvs[0][0].size(2) if past_kvs is not None else 0
        assert past_len + T <= self.cfg.max_len

        x = self.embed(input_ids)
        cos = self.rope_cos[: past_len + T]
        sin = self.rope_sin[: past_len + T]

        # Gradient checkpointing only makes sense when:
        #   - we're training (otherwise we're in no_grad and there's nothing to save)
        #   - we're not using KV cache (caching + checkpointing don't compose well —
        #     checkpoint recomputation would re-rotate K/V at wrong positions)
        use_ckpt = (self.cfg.gradient_checkpointing and self.training and past_kvs is None)

        new_past_kvs: list[tuple[Tensor, Tensor]] = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            if use_ckpt:
                # checkpoint requires the wrapped function to take only Tensors
                # as positional args. We use a small closure to thread None past_kv.
                def make_runner(blk):
                    def run(x_, cos_, sin_):
                        out, new_kv_ = blk(x_, cos_, sin_, past_kv=None)
                        # We can't checkpoint a tuple-returning op cleanly, but
                        # during training without cache we don't need new_kv.
                        return out
                    return run
                # use_reentrant=False is the modern recommended setting (PyTorch 2.x)
                x = checkpoint(make_runner(block), x, cos, sin, use_reentrant=False)
                new_past_kvs.append((torch.empty(0), torch.empty(0)))   # placeholder
            else:
                x, new_kv = block(x, cos, sin, past_kv=past_kv)
                new_past_kvs.append(new_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_past_kvs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Identical to GPTv2.generate. Gradient checkpointing is automatically
        skipped because we set self.eval() and the use_ckpt check requires
        self.training=True."""
        self.eval()
        logits, past_kvs = self(input_ids)
        next_logits = logits[:, -1, :]

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

            logits, past_kvs = self(next_id, past_kvs=past_kvs)
            next_logits = logits[:, -1, :]

        return input_ids


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfigV3(
        vocab_size=1000, d_model=64, n_heads=4, n_layers=2, max_len=128,
        gradient_checkpointing=True,
    )
    model = GPTv3(cfg)
    model.train()
    x = torch.randint(0, 1000, (2, 16))
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()                   # verify checkpointed backward works
    print(f"forward+backward OK, logits shape {tuple(logits.shape)}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
