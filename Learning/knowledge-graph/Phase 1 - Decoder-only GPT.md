# Phase 1 - Decoder-only GPT

Strips the encoder + cross-attention from Phase 0 to get a GPT-style language model. The
sanity-check phase: prove the architectural surgery trains before modernizing in Phase 2.

## Concepts
[[Decoder-only GPT architecture]] | [[Byte-level BPE]] | [[Subword tokenization]] | [[Block packing]] | [[Mixed precision training]] | [[WikiText-2 dataset]]

## In saLLMan
`phase1/decoder_only.py` reuses Phase 0 primitives; `phase1/train_lm.py` trains on WikiText-2.
Result: ~71 val perplexity, the baseline Phase 2 beats by ~10% at equal parameter count.

**Connects to:** [[Phase 0 - Vanilla Transformer]] | [[LLaMA architecture]] | [[Architecture progression]]
