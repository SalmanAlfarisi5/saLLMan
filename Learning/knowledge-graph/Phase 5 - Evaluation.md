# Phase 5 - Evaluation *(not yet implemented)*

> These notes are **deeper** than earlier phases because this is new material to learn.

Measure the model the way it actually matters for code: does it *run and pass tests*? The headline
metric is [[pass@k]] on held-out problems, scored by execution - not by text match.

## Concepts
[[pass@k]] | [[HumanEval]] | [[MBPP]] | [[LeetCode-style evaluation]] | [[Functional correctness]] | [[Test-case-based evaluation]] | [[Decontamination]] | [[Temperature sampling]] | [[Greedy vs sampling]]

## In saLLMan
Held-out LeetCode (`greengerong/leetcode`) + HumanEval, reusing the Phase 4 [[Code-execution reward]]
execution machinery.

**Connects to:** [[Phase 4 - GRPO reinforcement learning]] | [[Architecture progression]]
