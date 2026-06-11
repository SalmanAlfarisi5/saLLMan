# Reward hacking (and reward shaping)

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** Agents exploit loopholes in the reward (e.g. printing expected outputs, passing weak
tests without solving the problem, padding format tokens). Mitigations: strong/hidden test sets,
format rewards, baseline subtraction, and the [[KL regularization]] penalty.

**In saLLMan - the concrete case (not hypothetical).** The first GRPO run used raw pass-fraction as
the reward. Held-out reward climbed 0.022 -> 0.238, looking like a clean 10x success. Reading the
generated code via `compare_grpo.py` revealed the model had learned to print **constant outputs**
(`print(-1)`, `print(0)`): on a Codeforces problem whose test set is dominated by one expected
answer, a constant passes a large fraction of tests *without solving anything*. The 0.238 was ~96%
this exploit.

**The fix (three layers, all in `code_executor.py`):**
1. **Constant baseline** - `constant_baseline(tests)` = frequency of the modal expected output =
   the score a best-case constant program gets. Reward becomes
   `advantage = max(0, pass_fraction - constant_baseline)`. A constant nets ~0.
2. **Constant-output guard** - any completion emitting one identical output across all distinct test
   inputs is zeroed regardless (catches constants that aren't the modal value).
3. **Held-out eval under the honest reward** - the trustworthy signal. See
   [[Read the outputs not just the metric]].

**The diagnostic payoff.** Re-scoring the problem pool under the advantage reward dropped the
learnable rate 25.4% -> 1.8%. That number *is* the model-capability thermometer: almost all the
apparent "learnable spread" was hack-vs-crash variance. Private tests and [[Decontamination]] help,
but they don't fix the dominated-output baseline - that needs the subtraction. See
[[Verifiable reward has a baseline]].

**Connects to:** [[Code-execution reward]] | [[KL regularization]] | [[Decontamination]] | [[Verifiable reward has a baseline]] | [[Read the outputs not just the metric]]
