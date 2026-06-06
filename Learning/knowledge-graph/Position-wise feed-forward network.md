# Position-wise feed-forward network

**Cluster:** [[Phase 0 - Vanilla Transformer]]

## Intuition
A small per-token MLP applied identically at every position. Attention mixes *across* positions; the FFN mixes *features* within a position.

## Formula
```
FFN(x) = max(0, xW1 + b1) W2 + b2     # ReLU
```
Inner dimension is typically 4 × d_model.

## Connects to
Replaced in [[Phase 2 - LLaMA-class decoder]] by [[SwiGLU]] (a gated variant with the 8/3 hidden-dim trick).

## Reference
Vaswani et al. 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
