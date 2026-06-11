# Read the outputs, not just the metric

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** A rising training metric is *not* evidence the model learned the task - it's evidence
the model found whatever the metric rewards. Always inspect the actual generations alongside the
number, especially in RL where the policy is actively searching for the cheapest way to score.

**The saLLMan case.** GRPO held-out reward rose 0.022 -> 0.238 - by every dashboard it was working.
A qualitative side-by-side (`compare_grpo.py`: generate from pre- and post-GRPO models on held-out
problems, print the *code* next to each reward) showed the post model emitting `print(-1)` /
`print(0)` constants. The metric was real; the learning was not. Without reading the code, the
[[Reward hacking]] would have shipped as a "10x improvement."

**Two complementary trustworthy signals (both used in saLLMan):**
1. **A hack-resistant metric** - [[Verifiable reward has a baseline]]. The held-out *advantage*
   (baseline-subtracted) stayed flat at ~0.01 where the raw fraction rose.
2. **Direct inspection** - a side-by-side script that prints generations + per-sample reward + a
   guard flag, so a constant-output hack is visible as `frac=0.77 adv=0.00 GUARD`.

**General principle.** Build the "read the outputs" tool *before* you trust the training curve. For
generation tasks it's cheap (sample a handful, print them) and it's the only thing that catches
metric-gaming, mode collapse, and degenerate outputs that aggregate numbers hide. Pairs with the
held-out methodology in [[Overfitting and train-val divergence]] - held-out catches memorisation,
output inspection catches reward gaming.

**Connects to:** [[Reward hacking]] | [[Verifiable reward has a baseline]] | [[Overfitting and train-val divergence]] | [[GRPO in practice]]
