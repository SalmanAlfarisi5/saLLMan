# GPT-2 initialization

**Cluster:** [[Phase 2 - LLaMA-class modernized decoder]]

**Intuition.** Initialize weights from `N(0, 0.02)`, and scale residual-writing projections
(attention `out_proj`, SwiGLU `w2`) by `1/sqrt(2N)` (N = number of layers) so the [[Residual stream]]
variance stays bounded as depth grows.

**In saLLMan.** Phase 2 switches to this from [[Xavier initialization]]; embeddings get plain 0.02,
norm gains stay at 1.

## Reference
- GPT-2 "Language Models are Unsupervised Multitask Learners," Radford et al., 2019 (OpenAI tech report).

**Connects to:** [[Residual stream]] | [[Xavier initialization]]
