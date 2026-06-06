# DeepSeek-R1 / R1-Zero with GRPO

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** R1-Zero applied [[GRPO]] directly to a base model with *only* rule-based / verifiable
rewards (no SFT, no learned reward model) and reasoning behavior emerged on its own (AIME 2024 pass@1
15.6% -> 71.0%, 86.7% with majority voting). R1 added a cold-start SFT stage before RL to fix
readability and language-mixing.

**In saLLMan.** The blueprint for Phase 4: SFT checkpoint -> GRPO with a [[Code-execution reward]].

## Reference
- "DeepSeek-R1," DeepSeek-AI, 2025 - arXiv:2501.12948.

**Connects to:** [[GRPO]] | [[Reward model vs RLVR]] | [[Chain-of-thought]] | [[DeepSeek-R1]]
