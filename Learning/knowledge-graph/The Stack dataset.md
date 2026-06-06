# The Stack dataset

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** A large permissively-licensed source-code corpus for code-LLM pretraining.
**Deduplication** (exact + near-dup) materially improves quality - duplicates silently inflate
epoch counts and hurt generalization, so `the-stack-dedup` is preferred.

**In saLLMan.** Phase 3 streams `bigcode/the-stack-dedup` (Python) to a ~2B-token corpus.

## References
- "The Stack: 3 TB of permissively licensed source code," Kocetkov et al., 2022 - arXiv:2211.15533.
- "StarCoder 2 and The Stack v2," Lozhkov et al., 2024 - arXiv:2402.19173.

**Connects to:** [[Code LLMs]] | [[Decontamination]] | [[Chinchilla scaling laws]]
