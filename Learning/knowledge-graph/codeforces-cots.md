# codeforces-cots

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** `open-r1/codeforces-cots` - ~10k competitive-programming problems with ~100k
solutions distilled from DeepSeek-R1, plus a separate human-written `editorial` per problem.
Subsets include `solutions_py` and `_decontaminated` variants. ~84% of the Python traces pass the
public tests.

**In saLLMan.** SFT source (config `solutions_py_decontaminated`). saLLMan uses the **editorial** as
reasoning (not the `<think>` trace) and a trimmed `description + input_format + output_format` as the
problem. The `public_tests`/`private_tests` fields are reserved for the Phase 4 [[Code-execution reward]].

## Reference
- open-r1/codeforces-cots (Penedo et al., Hugging Face, 2025). Decontaminated via 8-gram overlap against AIME/GPQA/MATH-500/LiveCodeBench.

**Connects to:** [[Chain-of-thought]] | [[Decontamination]] | [[Supervised fine-tuning]] | [[Code-execution reward]]
