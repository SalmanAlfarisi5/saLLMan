# pass@k - central Phase 5 node

**Cluster:** [[Phase 5 - Evaluation]]

**Intuition.** The probability that *at least one* of `k` sampled solutions is correct. The naive
plug-in estimator `1 - (1 - c/n)^k` is **biased downward** (it *underestimates* pass@k): `1 - (1 - x)^k`
is concave in `x`, so by Jensen's inequality its expectation falls below the true value. Instead sample
`n >= k` solutions, count `c` correct, and use the **combinatorial estimator** below, which is both
**unbiased** and **lower-variance**. Worked example `n=2, k=2, p=0.5`: true `0.750`, plug-in `0.625`
(biased low), combinatorial `0.750` (exact).

**Formula.** `pass@k = E_problems[ 1 - C(n-c, k) / C(n, k) ]`.
Numerically: if `n - c < k` return 1.0; else use the product form to avoid overflow.

**Why n > k.** Drawing more samples than `k` and using the combinatorial estimator drastically reduces
variance versus generating exactly `k`.

**In saLLMan.** The Phase 5 headline metric, computed by [[Test-case-based evaluation]]. Tune
[[Temperature sampling]] to k: low T for pass@1, higher T for large k.

## Reference
- "Evaluating Large Language Models Trained on Code," Chen et al., 2021 - arXiv:2107.03374.

**Connects to:** [[HumanEval]] | [[Temperature sampling]] | [[Code-execution reward]] | [[Functional correctness]] | [[Greedy vs sampling]]
