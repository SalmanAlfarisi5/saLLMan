# Label smoothing

**Cluster:** [[Phase 0 - Vanilla Transformer]]

## Intuition
Instead of one-hot targets, use a softened distribution: mass (1 − ε) on the true token, ε spread over the rest. This stops the model becoming overconfident. Per Vaswani §5.4 it *hurts* perplexity (the model is deliberately less sure) but *improves* accuracy and BLEU.

## Mechanism
Minimize KL divergence to the smoothed target distribution. Vaswani used ε = 0.1.

## Connects to
[[BLEU perplexity and decoding]] · contrast with [[Phase 3 - Production-scale code pretraining and SFT]], which uses plain cross-entropy (label smoothing is less helpful at scale).

## Reference
Vaswani et al. 2017 §5.4 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
