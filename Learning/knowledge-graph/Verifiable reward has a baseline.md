# Verifiable reward has a baseline

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** A "verifiable" reward (run the code, count passing tests) feels objective and
un-gameable - but it still has a **trivial baseline**. If a problem's test set is dominated by one
expected output, a program that ignores the input and prints that constant passes a large fraction.
The reward must be measured *relative to that baseline*, not in absolute pass-rate.

**Formula (saLLMan).**
```
constant_baseline(tests) = (count of the modal expected output) / n_tests
advantage = max(0, pass_fraction - constant_baseline)
```
Plus a guard: zero any completion that emits one identical output across all distinct inputs.
Only beating the constant earns reward. Unit-asserted: a `print(-1)` on a `-1`-dominated set scores
~0; a genuinely-partial program scores >0; a full solution scores highest.

**Why it matters.** Competitive-programming datasets are full of dominated-output problems
(yes/no, -1/answer, divisibility). Under raw pass-fraction these are exactly where a model finds the
cheapest reward - so an un-baselined verifiable reward actively *teaches* constant-output behaviour.
This is the [[Code-execution reward]] failure mode behind saLLMan's [[Reward hacking]] result.

**General principle.** Any reward should be scored against the best trivial policy's reward, not
against zero. This is the same idea as the advantage baseline in [[Advantage estimation]] /
[[GRPO]] - a group-relative or constant baseline removes the "free" reward that any policy could get,
leaving only the part attributable to actually solving the task.

**Connects to:** [[Reward hacking]] | [[Code-execution reward]] | [[Advantage estimation]] | [[GRPO]]
