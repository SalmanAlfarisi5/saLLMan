# Encoder vs decoder cross-attention

**Cluster:** [[Phase 0 - Vanilla Transformer]]

**Intuition.** Decoder layers have a *third* sub-layer, encoder-decoder (cross) attention, where
queries come from the decoder and keys/values from the encoder output. This is how a translation
decoder conditions on the source.

**In saLLMan.** Present in Phase 0 only. **Phase 1's single architectural change is deleting this
sub-layer** - that is the entire encoder-decoder -> [[Decoder-only GPT architecture]] transition.

## Reference
- Vaswani et al., 2017 - arXiv:1706.03762.

**Connects to:** [[Decoder-only GPT architecture]] | [[Transformer architecture]] | [[Multi-head attention]]
