# Reward model vs RLVR

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** Classic [[RLHF]] *learns* a scalar reward model from human preferences. **RL with
Verifiable Rewards (RLVR)** replaces it with a *programmatic, rule-based* signal - e.g. does the
generated code pass the unit tests? - eliminating the reward model and its [[Reward hacking]] surface.

**DeepSeek-R1's two rule-based rewards:** **accuracy** (answer correct / code passes tests) and
**format** (correct `<think>` structure).

**In saLLMan.** RLVR is the natural fit: the reward is the [[Code-execution reward]] from running
solutions against `public_tests`/`private_tests`.

**Connects to:** [[Code-execution reward]] | [[GRPO]] | [[DeepSeek-R1]] | [[Reward hacking]]
