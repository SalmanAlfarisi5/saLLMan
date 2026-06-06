# Sinusoidal positional encoding

**Cluster:** [[Phase 0 - Vanilla Transformer]]

## Intuition
Attention is order-blind on its own, so inject absolute position using fixed sinusoids of geometrically increasing wavelength. The model can then reason about *where* a token sits.

## Formula
```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```
These are *added* to the token embeddings (which is why embeddings get scaled by √d_model — see [[Weight tying and embedding scaling]]).

## Connects to
Superseded in [[Phase 2 - LLaMA-class modernized decoder]] by [[RoPE]], which encodes *relative* position by rotating Q/K instead of adding to the input.

## Reference
Vaswani et al. 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
