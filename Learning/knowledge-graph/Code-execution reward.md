# Code-execution reward

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** The reward function compiles/runs each generated solution against test cases and
returns the pass rate. This is the **verifiable** reward signal ([[Reward model vs RLVR]]) - no
learned model needed. In open-r1, the code reward "executes solutions against a set of test cases and
the overall success rate is returned as the final reward" (E2B / Morph sandbox).

**In saLLMan.** Uses the `public_tests` / `private_tests` from [[codeforces-cots]]. The same execution
machinery powers Phase 5's [[Test-case-based evaluation]] and [[pass@k]].

**Watch for.** [[Reward hacking]] - weak tests can be gamed (printing expected outputs, etc.).

**Connects to:** [[Reward model vs RLVR]] | [[pass@k]] | [[Test-case-based evaluation]] | [[codeforces-cots]] | [[Reward hacking]]
