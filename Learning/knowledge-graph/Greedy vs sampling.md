# Greedy vs sampling

**Cluster:** [[Phase 5 - Evaluation]]

**Intuition.** **Greedy** (argmax / T=0) is deterministic and best when you report a single attempt
([[pass@k|pass@1]]). **Stochastic sampling** is *required* to estimate pass@k for k>1, because greedy
would give `k` identical samples.

**Connects to:** [[Temperature sampling]] | [[pass@k]] | [[BLEU perplexity and decoding]]
