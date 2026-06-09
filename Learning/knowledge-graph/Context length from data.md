# Context length from data

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** Pick the model's context length (`max_len`) from the **p90/p95 of the longest
training example token count**, not from intuition or from the model's intrinsic capability.
Context is a downstream constraint, not an upstream choice.

**Why it matters.** Activation memory at 2048 ctx is ~4x what it is at 512 ctx; attention compute
is ~16x. Picking too generously wastes compute and may not fit on the GPU. Picking too tightly
silently truncates examples (or drops them entirely via a length filter), discarding training
signal. The correct picking strategy is "smallest power of 2 that holds p95 of the data."

**Recipe (saLLMan).**
1. Run a length pass over the assembled training examples (tokenize them, get a histogram).
2. Choose `max_len` so the length filter drops <20% of examples. Round up to a multiple of 64
   (matmul tile alignment).
3. Plumb that `max_len` end-to-end: pretrain `block_size`, model `max_len`, RoPE cache size,
   FT prep `--max-len`, FT trainer `max_len`.

**The 512 -> 2048 redesign.** saLLMan's initial pretrain ran at 512 ctx because that fit "code
snippets". When `finetune_data_prep.py` measured the SFT example distribution (trimmed problem
+ editorial + code) on `open-r1/codeforces-cots`, it returned p50=1278, p90=2420. Truncating
to 512 would have dropped >80% of examples; truncating per-example would have cut the editorial
or the code. The decision was to rerun pretrain from scratch at 2048 - costly (5x wall-clock)
but the only way to make the FT data usable. The 512 partial run was discarded.

**Cost of getting it wrong, both directions.**
- *Too short:* training signal drops, batches under-utilise context, FT data becomes unusable.
- *Too long:* activation memory blows past 8 GB GPU; gradient checkpointing is forced; throughput
  drops ~30 %. With FlashAttention's O(n) memory, the attention matrix itself doesn't dominate -
  but residual + FFN activations still scale linearly with seq_len.

**Connects to:** [[Memory budget]] | [[Gradient checkpointing]] | [[FlashAttention]] | [[codeforces-cots]] | [[RoPE]]
