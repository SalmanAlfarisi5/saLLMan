# Multi-head attention

**Cluster:** [[Phase 0 - Vanilla Transformer]]

## Intuition
Run several attention "heads" in parallel, each in its own subspace, so the model can attend to different relations at once (e.g. one head tracks syntax, another tracks long-range dependency).

## Mechanism
Project Q, K, V into h subspaces of dimension `d_k = d_model / h`, run [[Scaled dot-product attention]] in each, concatenate the outputs, and project back to d_model.

## Connects to
[[Scaled dot-product attention]] · [[KV-cache]] (caches each head's K/V at inference) · [[FlashAttention]]

## Reference
Vaswani et al. 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

Status: Done