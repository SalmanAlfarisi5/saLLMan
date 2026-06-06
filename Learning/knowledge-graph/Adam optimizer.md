# Adam optimizer

**Cluster:** [[Phase 0 - Vanilla Transformer]]

## Intuition
Adaptive per-parameter learning rates using running estimates of the first and second gradient moments. Vaswani used Adam with β1 = 0.9, β2 = 0.98, ε = 1e-9.

## Connects to
[[AdamW]] — the Phase-2 successor with *decoupled* weight decay.

## Reference
"Adam: A Method for Stochastic Optimization," Kingma & Ba, 2014 — [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
