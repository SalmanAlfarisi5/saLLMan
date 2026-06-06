# RMSNorm

**Cluster:** [[Phase 2 - LLaMA-class modernized decoder]]

**Intuition.** Drop [[LayerNorm]]'s mean-centering (and bias); normalize only by the root-mean-square.
Cheaper, and the rescaling - not the centering - is what does the work.

**Formula.** `RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)`

**Implementation note.** Compute the variance in fp32 even under bf16 - norms are where precision
loss turns into NaNs.

**In saLLMan.** Phase 2's change 2b; used by LLaMA, T5, Gemma, Qwen, Mistral.

## Reference
- "Root Mean Square Layer Normalization," Zhang & Sennrich, 2019 - arXiv:1910.07467.

**Connects to:** [[LayerNorm]] | [[LLaMA architecture]] | [[Bias-free linear layers]] | [[Pre-LN]]
