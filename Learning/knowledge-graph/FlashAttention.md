# FlashAttention

**Cluster:** [[Phase 2 - LLaMA-class modernized decoder]]

**Intuition.** An IO-aware, *exact* attention algorithm. It tiles Q/K/V into blocks kept in fast
on-chip SRAM and computes the softmax online, so the full N x N score matrix is never written to
slow HBM.

**Result.** O(N) memory (vs O(N^2)) and a large wall-clock speedup, with **no approximation**.
The single biggest memory win for an 8 GB GPU ([[Memory budget]]).

**In PyTorch.** `F.scaled_dot_product_attention` dispatches to a fused FlashAttention kernel on
Ampere+; passing `is_causal=True` applies the causal mask internally (no mask tensor needed).

## References
- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," Dao et al., 2022 - arXiv:2205.14135.
- "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," Dao, 2023 - arXiv:2307.08691.

**Connects to:** [[Scaled dot-product attention]] | [[RoPE]] | [[KV-cache]] | [[Memory budget]]
