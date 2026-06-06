# KV-cache

**Cluster:** [[Phase 2 - LLaMA-class modernized decoder]]

**Intuition.** During generation, cache the key/value tensors of all previous tokens so each new
token attends against stored K/V (~O(T) per step) instead of recomputing them (O(T^2) per step).
Total decoding drops from ~O(T^3) to ~O(T^2).

**Subtleties (saLLMan).** With a cache, [[RoPE]] must rotate the new token at its *absolute*
position; and `is_causal` must be **False** during single-token decoding (the new query must attend
to *all* cached keys). [[Gradient checkpointing]] is disabled when the cache is active (they don't compose).

**In saLLMan.** Powers `GPTv2.generate` / `GPTv3.generate`, and will power fast Phase 4 rollouts
([[On-policy vs off-policy]]) and Phase 5 sampling ([[Temperature sampling]]).

## Reference
- The memory-bandwidth framing appears in "Fast Transformer Decoding: One Write-Head is All You Need," Shazeer, 2019 - arXiv:1911.02150 (whose actual contribution is multi-query attention). The bare cache is standard practice (e.g. nanoGPT).

**Connects to:** [[Multi-head attention]] | [[RoPE]] | [[On-policy vs off-policy]] | [[Temperature sampling]]
