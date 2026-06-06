# Phase 4 - GRPO reinforcement learning *(not yet implemented)*

> These notes are **deeper** than earlier phases because this is new material to learn.

Reinforcement-learning stage: generate multiple solution attempts per problem, run them against
test cases, and use pass/fail as a *verifiable* reward to push the SFT policy toward correctness -
the DeepSeek-R1 recipe via [[GRPO]].

## Concepts
[[RLHF]] | [[PPO]] | [[Reward model vs RLVR]] | [[GRPO]] | [[DeepSeek-R1 with GRPO]] | [[Code-execution reward]] | [[KL regularization]] | [[Advantage estimation]] | [[Reward hacking]] | [[On-policy vs off-policy]] | [[GRPO in practice]]

## Why it matters for saLLMan
[[GRPO]] is chosen specifically because it drops PPO's value network - the single biggest VRAM
saving in the RL phase ([[Memory budget]]). Built on top of the Phase 3 SFT checkpoint.

**Connects to:** [[Phase 3 - Production-scale code pretraining and SFT]] | [[Phase 5 - Evaluation]] | [[Architecture progression]]
