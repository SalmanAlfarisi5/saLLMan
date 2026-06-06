# On-policy vs off-policy (rollouts)

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** [[GRPO]]/[[PPO]] are on-policy in derivation but applied slightly off-policy in
practice (sample with `pi_theta_old`, update `pi_theta`; the ratio corrects for the gap). A
**rollout** = sampling N completions per prompt - and those N samples are exactly the *group* GRPO
normalizes over.

**In saLLMan.** Fast rollouts rely on the [[KV-cache]] and [[Temperature sampling]] (and a serving
engine like vLLM in [[GRPO in practice]]).

**Connects to:** [[GRPO]] | [[KV-cache]] | [[Temperature sampling]] | [[Advantage estimation]]
