# GRPO (Group Relative Policy Optimization) - central Phase 4 node

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** A [[PPO]] variant that **removes the value/critic network entirely**. Instead of
learning a value baseline, it samples a *group* of `G` completions per prompt, scores each, and uses
the group's own mean as the baseline. The advantage is simply how much better/worse a completion is
than its peers.

**Advantage formula.** For prompt `q` with `G` sampled outputs `{o_1..o_G}` and rewards `{r_1..r_G}`:
`A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)`.
In the standard outcome-supervision case, every token in `o_i` gets the same advantage `A_i`.

**Objective.**
`J_GRPO(theta) = E[ (1/G) sum_i (1/|o_i|) sum_t { min( rho * A_i,  clip(rho, 1-eps, 1+eps) * A_i ) - beta * D_KL(pi_theta || pi_ref) } ]`
with token ratio `rho = pi_theta(o_{i,t}) / pi_theta_old(o_{i,t})`.

**Why memory-efficient (crucial for 8 GB).** PPO keeps **three** model copies (policy + reference +
value); GRPO keeps **two** (policy + reference), since the group-mean baseline replaces the value
network. This is *the* reason saLLMan can attempt RL on a 3060 Ti ([[Memory budget]]).

## Reference
- "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," Shao et al., 2024 - arXiv:2402.03300.

**Connects to:** [[PPO]] | [[Advantage estimation]] | [[KL regularization]] | [[DeepSeek-R1 with GRPO]] | [[On-policy vs off-policy]] | [[Memory budget]]
