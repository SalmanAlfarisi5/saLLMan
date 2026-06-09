# Scaled dot-product attention

**Cluster:** [[Phase 0 - Vanilla Transformer]]

## Intuition
Each query retrieves a weighted average of values, where the weights come from query–key similarity. "Which past tokens are relevant to me, and how much?"

## Formula
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

## Why the √d_k scaling
If query/key components are independent with mean 0 and variance 1, the dot product has variance d_k. Per Vaswani §3.2.1, for large d_k the dot products grow large in magnitude, pushing softmax into regions with extremely small gradients. Dividing by √d_k counteracts this.

## Connects to
[[Multi-head attention]] · [[FlashAttention]] (the IO-aware exact implementation) · [[Causal and padding masking]] (how future tokens get −∞ before softmax)

## Reference
Vaswani et al. 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

Status: Done