# BLEU, perplexity, and decoding

**Cluster:** [[Phase 0 - Vanilla Transformer]]

**Intuition.**
- **BLEU** - n-gram precision metric for translation quality.
- **Perplexity** - `exp(cross-entropy)`; lower = better LM fit (1 is perfect, vocab-size is random).
- **Greedy decoding** - take the argmax token each step.
- **Beam search** - keep the top-b partial hypotheses.

**In saLLMan.** Perplexity is the headline metric through Phases 0-3. **For code (Phase 5),
BLEU is a poor fit** - see [[Functional correctness]] and [[pass@k]], which measure whether code
*runs and passes tests*.

**Connects to:** [[Functional correctness]] | [[pass@k]] | [[Label smoothing]] | [[Greedy vs sampling]]
