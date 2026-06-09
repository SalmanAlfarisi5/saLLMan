# Post-LN

**Cluster:** [[Phase 0 - Vanilla Transformer]]

## Intuition
The original Transformer applies [[LayerNorm]] *after* the residual add:
```
x = LayerNorm(x + Sublayer(x))
```
i.e. it normalises the [[Residual stream]] itself at every block.

## Why it needs warmup
Xiong et al. showed (mean-field analysis) that at initialization Post-LN has large expected gradients near the output layer, so a large learning rate is unstable. A learning-rate warmup stage (the [[Noam scheduler]]) is therefore essential.

## Connects to
[[Pre-LN]] (the fix) · [[Noam scheduler]] · [[Residual stream]]

## Reference
"On Layer Normalization in the Transformer Architecture," Xiong et al., 2020 — [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
Status: Done