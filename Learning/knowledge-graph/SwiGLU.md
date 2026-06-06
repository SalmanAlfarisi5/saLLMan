# SwiGLU

**Cluster:** [[Phase 2 - LLaMA-class modernized decoder]]

**Intuition.** Replace the ReLU [[Position-wise feed-forward network]] with a gated linear unit
using a Swish gate. The element-wise gate lets each hidden dimension be independently amplified
or suppressed; empirically the best of the GLU variants.

**Formula.** `SwiGLU(x) = (Swish(x W1) (*) x W3) W2`, where `Swish(x) = x * sigmoid(x)` (= SiLU).

**The 8/3 ratio.** SwiGLU adds a third matrix, so the hidden dim is set to `~8/3 * d_model`
(instead of `4 * d_model`) to keep parameter count constant. LLaMA names: `w1 = gate_proj`,
`w3 = up_proj`, `w2 = down_proj`.

**In saLLMan.** Phase 2's change 2d.

## Reference
- "GLU Variants Improve Transformer," Shazeer, 2020 - arXiv:2002.05202.

**Connects to:** [[Position-wise feed-forward network]] | [[LLaMA architecture]]
