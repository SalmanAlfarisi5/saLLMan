# Phase 0 - Vanilla Transformer

**Cluster:** Phase 0 overview · part of [[saLLMan MOC]] · step 1 of [[Architecture progression]]

## What this phase is
A from-scratch reimplementation of the original **encoder-decoder Transformer** (Vaswani et al. 2017), trained on DE→EN translation ([[Multi30k dataset]]). It establishes all the primitives the later decoder-only phases reuse.

## The big idea
Replace recurrence with attention so every position is processed in parallel. An **encoder** builds a bidirectional representation of the source; a **decoder** generates the target autoregressively while attending to the encoder via cross-attention. See [[Encoder vs decoder]].

## Concepts introduced here
- [[Scaled dot-product attention]]
- [[Multi-head attention]]
- [[Sinusoidal positional encoding]]
- [[Position-wise feed-forward network]]
- [[LayerNorm]] and [[Post-LN]]
- [[Encoder vs decoder]] / cross-attention
- [[Causal masking]] and padding masking
- [[Weight tying]] and embedding scaling
- [[Teacher forcing]]
- [[Label smoothing]]
- [[Noam scheduler]]
- [[Xavier initialization]]
- [[Adam optimizer]]
- [[BLEU and perplexity]] / decoding

## Reference
"Attention Is All You Need," Vaswani et al., 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
