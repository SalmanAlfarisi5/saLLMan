# Block packing

**Cluster:** [[Phase 1 - Decoder-only GPT]]

**Intuition.** Concatenate the entire tokenized corpus into one long stream and slice it into
fixed-length blocks (the nanoGPT pattern). ~100% of tokens contribute to loss - no padding waste.
Trade-off: a block can span a document boundary, which is fine (the model learns to reset at `<eos>`).

**In saLLMan.** `BlockDataset` in Phase 1. Phase 3 keeps the recipe but reads from disk via
[[Memory-mapped data loading]] so the stream never has to fit in RAM.

**Connects to:** [[Memory-mapped data loading]] | [[Next-token prediction]] | [[Byte-level BPE]]
