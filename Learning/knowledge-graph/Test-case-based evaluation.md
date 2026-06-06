# Test-case-based / execution-based evaluation

**Cluster:** [[Phase 5 - Evaluation]]

**Intuition.** Run generated code in a sandbox against input/output test cases (public + private);
the pass/fail outcome is the ground truth. The same machinery as the Phase 4 [[Code-execution reward]],
just used for scoring instead of training.

**Connects to:** [[pass@k]] | [[Code-execution reward]] | [[Functional correctness]]
