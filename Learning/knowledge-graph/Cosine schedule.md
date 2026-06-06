# Cosine schedule (with linear warmup)

**Cluster:** [[Phase 2 - LLaMA-class modernized decoder]]

**Intuition.** Linearly warm up the LR over a few hundred/thousand steps, then decay it along a
cosine curve to a small floor (~10% of max). The GPT-3 / Chinchilla / LLaMA default.

**Formula.** warmup: `lr = max_lr * step/warmup`; then
`lr = min_lr + 0.5*(max_lr - min_lr)*(1 + cos(pi * progress))`.

**In saLLMan.** Replaces the [[Noam scheduler]] because [[Pre-LN]] doesn't need aggressive warmup.
Reused (inlined) in Phase 3 pretrain and fine-tune.

**Connects to:** [[Noam scheduler]] | [[Pre-LN]] | [[AdamW]]
