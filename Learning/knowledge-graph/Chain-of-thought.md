# Chain-of-thought (editorial-as-reasoning)

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** Train on step-by-step reasoning traces so the model learns to "think" before
emitting an answer. saLLMan uses concise human-written **editorials** as the reasoning target
rather than raw R1 `<think>` traces (whose ~15k-token median blows the 2048 context).

**In saLLMan.** The SFT target format is `<problem>...<reasoning>{editorial}</reasoning><code>{code}</code>`.
See [[DeepSeek-R1]] (origin of the `<think>` traces) and [[codeforces-cots]] (the source data).

**Connects to:** [[DeepSeek-R1]] | [[codeforces-cots]] | [[Supervised fine-tuning]]
