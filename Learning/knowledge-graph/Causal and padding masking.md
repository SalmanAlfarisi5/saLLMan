# Causal and padding masking

**Cluster:** [[Phase 0 - Vanilla Transformer]]

**Intuition.** Two masks. The **causal mask** prevents a position from attending to *future*
tokens (sets future logits to -inf before softmax), preserving autoregressive validity. The
**padding mask** hides `<pad>` tokens in batched variable-length inputs.

**Mechanism.** Combine both with logical AND so attention keeps a position only if it's
non-future AND non-pad.

**In saLLMan.** `make_causal_mask` / `make_pad_mask` in Phase 0, reused in Phase 1. Phase 2 hands
causality to [[FlashAttention]] via `is_causal=True` and never materializes the mask tensor.

**Connects to:** [[Teacher forcing]] | [[Next-token prediction]] | [[Dynamic padding]] | [[FlashAttention]]

Status: Done