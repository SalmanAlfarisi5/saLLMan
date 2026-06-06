# KL regularization

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** The `beta * D_KL(pi_theta || pi_ref)` term penalizes drifting too far from the
reference (SFT) policy, preventing the model from collapsing into degenerate high-reward gibberish.
A core part of both [[PPO]] and [[GRPO]].

**Tuning.** If reward climbs but outputs degrade or eval [[pass@k]] stalls, raise `beta` (and
strengthen tests) - a classic [[Reward hacking]] symptom.

**Connects to:** [[PPO]] | [[GRPO]] | [[Reward hacking]] | [[RLHF]]
