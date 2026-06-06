# Chinchilla scaling laws

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** For a fixed compute budget, model size `N` and training tokens `D` should scale
*together*; the compute-optimal ratio is roughly **20 tokens per parameter**. Under-training a
large model (or over-epoching a tiny corpus) wastes compute and overfits.

**Formula.** Loss `L(N,D) = E + A/N^alpha + B/D^beta`; compute `C ~= 6 N D`.

**In saLLMan.** The whole Phase 3 v1->v2 story: v1 had ~0.5 tokens/param (overfit, train PPL 1.34
vs val 68); v2 targets ~20 tokens/param with a 2B-token corpus. See [[Overfitting and train-val divergence]].

## Reference
- "Training Compute-Optimal Large Language Models," Hoffmann et al., 2022 - arXiv:2203.15556.

**Connects to:** [[Tokenization thread]] | [[Overfitting and train-val divergence]] | [[The Stack dataset]]
