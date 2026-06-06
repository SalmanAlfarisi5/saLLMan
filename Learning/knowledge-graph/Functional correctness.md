# Functional correctness (vs match-based metrics)

**Cluster:** [[Phase 5 - Evaluation]]

**Intuition.** Code is correct iff it *runs and passes tests*, not if its text matches a reference.
BLEU / exact-match correlate poorly with correctness because a correct program can be written many
ways. This is why Phase 5 uses execution, not [[BLEU perplexity and decoding|BLEU]].

**Connects to:** [[BLEU perplexity and decoding]] | [[pass@k]] | [[Test-case-based evaluation]]
