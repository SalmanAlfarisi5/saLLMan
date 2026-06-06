# LayerNorm

**Cluster:** [[Phase 0 - Vanilla Transformer]]

## Intuition
Normalise each token's activation vector to zero mean / unit variance, then rescale and shift with learned parameters γ and β. Keeps activations in a stable range across the depth of the network.

## Formula
```
LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β
```

## Connects to
Placement on the [[Residual stream]] defines [[Post-LN]] vs [[Pre-LN]]. The cheaper modern variant is [[RMSNorm]], which drops the mean-centering and the bias.

## Reference
"Layer Normalization," Ba, Kiros & Hinton, 2016 — [arXiv:1607.06450](https://arxiv.org/abs/1607.06450).
