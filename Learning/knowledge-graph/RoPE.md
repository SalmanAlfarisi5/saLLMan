# RoPE (Rotary Position Embedding)

**Cluster:** [[Phase 2 - LLaMA-class modernized decoder]]

**Intuition.** Encode position by *rotating* each query/key 2D sub-vector by an angle proportional
to its position. The dot product then depends only on the **relative** offset `m - n` - exactly
what attention wants ("how far apart", not "what absolute index").

**Mechanism.** Apply block-diagonal 2x2 rotations with frequencies `theta_i = base^(-2i/d_head)`.
Applied to Q and K only (not V - V carries content, not position).

**Why it composes with [[FlashAttention]].** RoPE rotates Q/K *before* the attention matmul and
preserves norms, so it slots in without materializing any bias matrix.

**RoPE scaling / position interpolation.** Rescale positions (or the base frequency) to extend
context beyond the trained length.

**In saLLMan.** Phase 2's change 2c (replaces [[Sinusoidal positional encoding]]). The RoPE cos/sin
tables are precomputed once and sliced per forward pass; during [[KV-cache]] decoding the new
token is rotated at its *absolute* position `past_len`, not 0.

## Reference
- "RoFormer: Enhanced Transformer with Rotary Position Embedding," Su et al., 2021 - arXiv:2104.09864.

**Connects to:** [[Sinusoidal positional encoding]] | [[FlashAttention]] | [[KV-cache]]
