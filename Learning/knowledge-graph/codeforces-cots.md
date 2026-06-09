# codeforces-cots

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** `open-r1/codeforces-cots` - ~10k competitive-programming problems with ~100k
solutions distilled from DeepSeek-R1, plus a separate human-written `editorial` per problem.
Subsets include `solutions_py` and `_decontaminated` variants. ~84% of the Python traces pass the
public tests.

**In saLLMan.** SFT source (config `solutions_py_decontaminated`). saLLMan uses the **editorial** as
reasoning (not the `<think>` trace, which has ~15k-token median - far past 2048) and a trimmed
`description + input_format + output_format` as the problem (drops the Codeforces flavor narrative
that bloats most rows past 2048). The `public_tests`/`private_tests` fields are reserved for the
Phase 4 [[Code-execution reward]].

**Empirical yield (verified).** 8 133 rows seen -> 4 504 yielded by the extractor -> **3 704 kept**
after the 2048-token length filter (3 519 train / 185 val at val_ratio=0.05). Biggest skip bucket
is `skip_no_editorial = 3 228` (not every row has an editorial). `finish_reason="length"` is
*counted but not filtered* - dropping that filter rescued 362 otherwise-discarded rows whose code
block was intact even though the R1 generation hit max-length.

## Reference
- open-r1/codeforces-cots (Penedo et al., Hugging Face, 2025). Decontaminated via 8-gram overlap against AIME/GPQA/MATH-500/LiveCodeBench.

**Connects to:** [[Chain-of-thought]] | [[Decontamination]] | [[Supervised fine-tuning]] | [[Code-execution reward]] | [[Context length from data]] | [[Schema verification before coding]]
