# RLHF (the overall pipeline)

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** Align a model to desired behavior in three stages:
1. **[[Supervised fine-tuning]]** to get an initial policy `pi_SFT`.
2. Train a **[[Reward model vs RLVR|reward model]]** on human preference rankings (Bradley-Terry).
3. Optimize the policy against that reward with **[[PPO]]**, under a **[[KL regularization]]** penalty
   to `pi_SFT`.

**The saLLMan twist.** saLLMan skips the *learned* reward model and uses a programmatic
[[Code-execution reward]] instead (RLVR), and uses [[GRPO]] in place of PPO. So the pipeline here is:
pretrain -> SFT -> GRPO-with-verifiable-rewards.

## References
- "Training language models to follow instructions with human feedback" (InstructGPT), Ouyang et al., 2022 - arXiv:2203.02155.
- "Deep reinforcement learning from human preferences," Christiano et al., 2017 - arXiv:1706.03741.

**Connects to:** [[PPO]] | [[Reward model vs RLVR]] | [[Supervised fine-tuning]] | [[KL regularization]] | [[GRPO]]
