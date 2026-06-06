# Noam scheduler

**Cluster:** [[Phase 0 - Vanilla Transformer]]

## Intuition
Warm the learning rate up linearly for the first few thousand steps, then decay it as the inverse square root of the step. The warmup is required because of [[Post-LN]]'s unstable early gradients.

## Formula (Vaswani §5.3, Eq. 3)
```
lrate = d_model^(−0.5) · min(step^(−0.5), step · warmup^(−1.5))
```
with warmup_steps = 4000.

## Connects to
[[Post-LN]] (the reason warmup is needed) · [[Cosine schedule]] (the modern replacement, enabled by [[Pre-LN]])

## Reference
Vaswani et al. 2017 §5.3 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
