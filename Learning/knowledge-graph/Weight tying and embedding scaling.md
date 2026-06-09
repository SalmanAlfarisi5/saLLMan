# Weight tying and embedding scaling

**Cluster:** [[Phase 0 - Vanilla Transformer]]

**Intuition.** **Weight tying:** share one matrix between the input embedding and the output
(pre-softmax) projection - saves `vocab_size * d_model` parameters and slightly improves quality.
**Embedding scaling:** multiply embeddings by `sqrt(d_model)` to balance their magnitude against
the additive [[Sinusoidal positional encoding]].

**In saLLMan.** Both used in Phases 0/1. Phase 2 keeps weight tying but **drops the sqrt(d_model)
scaling** because [[RoPE]] doesn't perturb the embedding stream.

## Reference
- Vaswani et al., 2017, 3.4 - arXiv:1706.03762.

**Connects to:** [[Sinusoidal positional encoding]] | [[RoPE]] | [[Tokenization thread]]

Status: Done