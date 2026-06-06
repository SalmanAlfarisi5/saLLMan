# PPO (Proximal Policy Optimization)

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** A policy-gradient method that keeps each update inside a "trust region" by
*clipping* the probability ratio, preventing destructive large updates. The original RLHF optimizer.

**Formula (clipped surrogate).**
`J_PPO(theta) = E[ min( r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t ) ]`
where `r_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)` is the probability ratio and `A_t` the
[[Advantage estimation|advantage]].

**The cost that motivates [[GRPO]].** PPO needs a separate **value/critic network** (~same size as
the policy) to estimate advantages. With policy + reference + value, that's **three** model copies in
memory - painful on 8 GB ([[Memory budget]]).

## Reference
- "Proximal Policy Optimization Algorithms," Schulman et al., 2017 - arXiv:1707.06347.

**Connects to:** [[GRPO]] | [[Advantage estimation]] | [[KL regularization]] | [[RLHF]]
