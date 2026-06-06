# Gradient checkpointing

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** Store activations for only a subset of layers in the forward pass and *recompute*
the rest during backprop. Trades ~30% extra compute for a large activation-memory reduction
(~sqrt(L)). Numerically identical results.

**In saLLMan.** Phase 3's only architectural addition (`GPTv3`). Enabled only when training AND not
using a [[KV-cache]] (the two don't compose). Required at 2048 context on 8 GB ([[Memory budget]]).

## Reference
- "Training Deep Nets with Sublinear Memory Cost," Chen et al., 2016 - arXiv:1604.06174.

**Connects to:** [[Mixed precision training]] | [[Gradient accumulation]] | [[Memory budget]] | [[KV-cache]]
