# Phase 2 - LLaMA-class modernized decoder

Applies every post-2020 architectural improvement that LLaMA/Mistral/Qwen share. Same parameter
count and training budget as Phase 1 - what changes is *quality per parameter* (~10% lower
perplexity on WikiText-2).

## Concepts
[[Pre-LN]] | [[RMSNorm]] | [[RoPE]] | [[SwiGLU]] | [[FlashAttention]] | [[KV-cache]] | [[GPT-2 initialization]] | [[AdamW]] | [[Cosine schedule]] | [[LLaMA architecture]] | [[Bias-free linear layers]]

## In saLLMan
`phase2/decoder_only_v2.py` is the only self-contained model file (every primitive changed).
This is the architecture Phase 3 scales up unchanged.

**Connects to:** [[Phase 1 - Decoder-only GPT]] | [[Phase 3 - Production-scale code pretraining and SFT]] | [[Architecture progression]]
