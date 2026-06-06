# GRPO log-prob bookkeeping

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** The [[GRPO]] objective reads as a single expression, but implementing it correctly
means tracking **three** separate sets of per-token log-probs over the *same* sampled tokens. Confuse
them and you silently corrupt either the importance ratio or the KL penalty - the two places the
gradient actually enters.

**Mechanism.** For every sampled completion token you keep log-probs under three policies:
1. **pi_theta** - the *current* policy being optimized. Recomputed at every gradient step; the only
   one carrying gradient.
2. **pi_theta_old** - a *frozen snapshot* taken per rollout batch. It is exactly the distribution you
   sampled the group with, so it forms the importance ratio `rho = pi_theta / pi_theta_old`.
   **Refreshed each rollout** - it tracks the policy as it moves.
3. **pi_ref** - a *frozen reference* policy (the Phase 3 SFT checkpoint), used **only** for the KL
   penalty `D_KL(pi_theta || pi_ref)` (unbiased k3 estimator). **Fixed for the whole run** - it never
   updates.

Both `pi_theta_old` and `pi_ref` are frozen, but on different clocks: `pi_theta_old` is
re-snapshotted every rollout, while `pi_ref` stays put from the first step to the last.

**8 GB consequence.** The two frozen sets cost very differently. `pi_theta_old` log-probs can be
**computed once at rollout/generation time and cached** as plain tensors - you are already running
that forward pass to sample, so it needs **no extra resident model**. `pi_ref` is different: the KL
term needs `pi_ref`'s log-probs of the *current* tokens, so it demands a **second resident forward
pass** (or a cached frozen copy of the SFT weights). That standing reference model is the
**"two model copies"** (policy + reference) of the [[Memory budget]] note made concrete -
`pi_theta_old` is free, `pi_ref` is not.

**Connects to:** [[GRPO]] | [[PPO]] | [[Advantage estimation]] | [[KL regularization]] | [[On-policy vs off-policy]] | [[Memory budget]]
