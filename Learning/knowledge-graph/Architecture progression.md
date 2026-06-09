# Architecture progression

**Cluster:** cross-cutting hub · part of [[Home]]

## Intuition
saLLMan is built as a *progression*, where each phase swaps in components without ever changing the auto regressive [[Next-token prediction]] core:

1. **[[Phase 0 - Vanilla Transformer]]** — the original encoder-decoder Transformer (translation).
2. **[[Phase 1 - Decoder-only GPT]]** — drop the encoder + cross-attention; one causal decoder stack.
3. **[[Phase 2 - LLaMA-class modernized decoder]]** — improve the internals: [[Pre-LN]], [[RMSNorm]], [[RoPE]], [[SwiGLU]], [[FlashAttention]].
4. **[[Phase 3 - Production-scale code pretraining and SFT]]** — scale up on real code data, then [[Supervised fine-tuning]] on reasoning traces.
5. **[[Phase 4 - GRPO reinforcement learning]]** — optimize against a verifiable reward (does the code pass tests?).
6. **[[Phase 5 - Evaluation]]** — measure functional correctness with [[pass@k]].

## Why this framing helps
Each upgrade is *isolated*: you can reason about one change (e.g. swapping [[Sinusoidal positional encoding]] for [[RoPE]]) without re-learning everything else. The constants that never change — the [[Residual stream]], the [[Tokenization thread]], next-token prediction — are the thread you hold onto.

## Connects to
[[Residual stream]] · [[Tokenization thread]] · [[Next-token prediction]]

Status: Done