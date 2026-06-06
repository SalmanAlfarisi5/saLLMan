# Transformer architecture

**Cluster:** [[Phase 0 - Vanilla Transformer]]

**Intuition.** Replaces recurrence with attention so all positions process in parallel. An
encoder builds a bidirectional representation of the source; a decoder generates the target
autoregressively while attending to the encoder via [[Encoder vs decoder cross-attention]].

**Mechanism.** Stacked blocks of [[Multi-head attention]] + [[Position-wise feed-forward network]],
each wrapped in a residual add + [[LayerNorm]] (the [[Residual stream]]).

**In saLLMan.** Phase 0 builds this end to end; Phase 1 strips the encoder to get a
[[Decoder-only GPT architecture]].

## Reference
- "Attention Is All You Need," Vaswani et al., 2017 - arXiv:1706.03762 (NeurIPS 2017).

**Connects to:** [[Scaled dot-product attention]] | [[Multi-head attention]] | [[Position-wise feed-forward network]] | [[Encoder vs decoder cross-attention]] | [[Residual stream]]
