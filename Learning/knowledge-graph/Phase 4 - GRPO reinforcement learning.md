# Phase 4 - GRPO reinforcement learning

Reinforcement-learning stage: generate multiple solution attempts per problem, run them against
test cases, and use pass/fail as a *verifiable* reward to push the SFT policy toward correctness -
the DeepSeek-R1 recipe via [[GRPO]]. **Implemented and run** in saLLMan; the headline outcome is a
documented negative result driven by [[Reward hacking]].

## Concepts
[[RLHF]] | [[PPO]] | [[Reward model vs RLVR]] | [[GRPO]] | [[DeepSeek-R1 with GRPO]] | [[Code-execution reward]] | [[KL regularization]] | [[Advantage estimation]] | [[Reward hacking]] | [[On-policy vs off-policy]] | [[GRPO in practice]] | [[GRPO log-prob bookkeeping]]

## Process lessons (this run)
[[Reward hacking]] | [[Verifiable reward has a baseline]] | [[Read the outputs not just the metric]]

## Why GRPO for saLLMan
[[GRPO]] drops PPO's value network - the single biggest VRAM saving in the RL phase
([[Memory budget]]). saLLMan runs **two resident 97M models** (trainable policy + frozen reference
for the KL term), peak ~3 GB on the 8 GB 3060 Ti; three (PPO) would not fit.

## In saLLMan (`phase4/`)
- `code_executor.py` - subprocess+rlimit sandbox, `reward_fraction` (raw pass rate) and the
  anti-hack `reward_advantage = max(0, fraction - constant_baseline)` + constant-output guard.
- `rollouts.py::generate_group` - G completions per problem, each scored (the group GRPO normalises).
- `build_curriculum.py` - scores the pool to find problems with reward *variance* (group_std>0);
  dead-signal (std=0) problems give zero advantage and are excluded.
- `grpo.py` - the loop: clipped surrogate `min(rho·A, clip(rho,1±ε)·A)` minus `β·KL_k3`, masked on
  response tokens, held-out eval, resume-safe checkpoints. The three-log-prob bookkeeping
  ([[GRPO log-prob bookkeeping]]): policy carries grad, old + ref are detached.
- `compare_grpo.py` - pre-vs-post qualitative side-by-side (read the code, not just the number).

## The result (negative, and instructive)
| Run | Reward | Held-out | Reading |
|-----|--------|----------|---------|
| v1 (1500 steps) | raw pass fraction | 0.022 -> 0.238 | **mostly reward hacking** (constant outputs) |
| v2 (200 steps) | advantage (anti-hack) | 0.010 -> 0.010 | hack blocked -> ~no real headroom at 97M |

Re-scoring the curriculum under the honest reward dropped the learnable rate **25.4% -> 1.8%**:
almost all the v1 "learnable spread" was hack-vs-crash variance. The GRPO machinery, reward design,
curriculum, guard, and held-out methodology all work; the limiting factor is **model scale x dataset
difficulty**, not the RL implementation.

**Connects to:** [[Phase 3 - Production-scale code pretraining and SFT]] | [[Phase 5 - Evaluation]] | [[Architecture progression]]
