# Pre-LN

**Cluster:** [[Phase 2 - LLaMA-class modernized decoder]]

**Intuition.** Apply [[LayerNorm]] *inside* the residual branch, before each sub-layer:
```
x = x + Sublayer(LayerNorm(x))
``` 
plus one final norm before the output head.

**Why stable without warmup.** Gradients are well-behaved at initialization (they scale down with
depth), so the [[Noam scheduler]] warmup can be removed and a [[Cosine schedule]] used instead.
Pre-LN preserves a clean additive [[Residual stream]] from input to output.

**In saLLMan.** Phase 2's change 2a. Requires a **final [[RMSNorm]] before the lm_head**, else the
un-normalized stream blows up logits.

## Reference
- "On Layer Normalization in the Transformer Architecture," Xiong et al., 2020 - arXiv:2002.04745.

**Connects to:** [[Post-LN]] | [[Noam scheduler]] | [[Cosine schedule]] | [[Residual stream]] | [[RMSNorm]]

Status: Done