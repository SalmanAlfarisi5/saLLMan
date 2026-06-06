# Teacher forcing

**Cluster:** [[Phase 0 - Vanilla Transformer]]

## Intuition
During training, feed the model the *ground-truth* previous token (not its own prediction) as input. Combined with [[Causal masking]], this lets every position be trained in parallel in a single forward pass.

## Connects to
[[Causal masking]] · [[Next-token prediction]]

## Reference
Standard sequence-to-sequence practice; used throughout Vaswani et al. 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
