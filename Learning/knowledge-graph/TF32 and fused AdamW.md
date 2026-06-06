# TF32 matmuls and fused AdamW

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** Two free Ampere speedups. **TF32** is a tensor-core matmul mode (10-bit mantissa)
giving large speedups at ~fp32 quality - enable via `torch.set_float32_matmul_precision('high')`.
**Fused AdamW** runs the whole optimizer step in one fused CUDA kernel.

**In saLLMan.** Both on in Phase 3 pretrain/finetune; part of the [[Memory budget]] / throughput kit.

**Connects to:** [[Mixed precision training]] | [[AdamW]] | [[Memory budget]]
