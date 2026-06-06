# Advantage estimation (policy-gradient basics)

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** Policy gradients push *up* the log-probability of actions with positive advantage
(better than baseline) and *down* those with negative advantage. The **advantage** = reward minus a
baseline; the **log-prob ratio** `pi_theta / pi_theta_old` reweights for off-policy data.

**The key contrast.** [[PPO]] estimates the baseline with a learned value network; [[GRPO]] uses the
*group mean* of sampled rewards - no value network, hence the memory saving.

**Connects to:** [[PPO]] | [[GRPO]] | [[On-policy vs off-policy]]
