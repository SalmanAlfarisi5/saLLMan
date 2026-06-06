# Decontamination

**Cluster:** [[Phase 5 - Evaluation]]

**Intuition.** Ensure benchmark/test problems do **not** appear in training data (e.g. via n-gram
overlap filtering), otherwise scores are inflated by memorization rather than reasoning.

**In saLLMan.** The `solutions_py_decontaminated` subset of [[codeforces-cots]] is decontaminated via
8-gram overlap against common benchmarks. Related to the [[Overfitting and train-val divergence]]
concern and to [[Reward hacking]] in Phase 4.

**Connects to:** [[codeforces-cots]] | [[Overfitting and train-val divergence]] | [[Reward hacking]]
