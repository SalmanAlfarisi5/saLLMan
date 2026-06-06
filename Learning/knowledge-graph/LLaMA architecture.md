# LLaMA architecture

**Cluster:** [[Phase 2 - LLaMA-class modernized decoder]]

**Intuition.** The canonical modern open decoder = [[Pre-LN]] + [[RMSNorm]] + [[RoPE]] + [[SwiGLU]]
+ [[Bias-free linear layers]] + [[AdamW]] + [[Cosine schedule]]. Mistral and Qwen are LLaMA-class
variants (adding e.g. grouped-query attention, sliding-window attention).

**In saLLMan.** This *is* the Phase 2 model, and the architecture Phase 3 scales to ~97M params and
2048 context.

## Reference
- "LLaMA: Open and Efficient Foundation Language Models," Touvron et al., 2023 - arXiv:2302.13971.

**Connects to:** [[Pre-LN]] | [[RMSNorm]] | [[RoPE]] | [[SwiGLU]] | [[Bias-free linear layers]] | [[Decoder-only GPT architecture]]
