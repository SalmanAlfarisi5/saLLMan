# DeepSeek-R1

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** A reasoning model that emits explicit `<think>...</think>` traces, trained with
[[GRPO]] and *rule-based / verifiable* rewards. **R1-Zero** used pure RL with no SFT and reasoning
emerged spontaneously (AIME 2024 pass@1 15.6% -> 71.0%, 86.7% with majority voting). R1 added
cold-start SFT to fix readability.

**In saLLMan.** The recipe Phase 4 follows; its distilled traces seed [[codeforces-cots]]. The two
rule-based rewards (accuracy + format) are the model for saLLMan's [[Code-execution reward]].

## Reference
- "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," DeepSeek-AI, 2025 - arXiv:2501.12948.

**Connects to:** [[GRPO]] | [[Reward model vs RLVR]] | [[Chain-of-thought]] | [[DeepSeek-R1 with GRPO]]
