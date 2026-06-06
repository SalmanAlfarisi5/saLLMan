# Mixed precision training

**Cluster:** [[Phase 1 - Decoder-only GPT]]

**Intuition.** Compute in 16-bit to halve memory and speed up matmuls, keeping sensitive ops in
32-bit. **bfloat16** has fp32's exponent range (no loss scaling needed) but fewer mantissa bits;
**fp16** is more precise but narrow-range, requiring a **GradScaler**. `autocast` picks per-op
precision. On the 3060 Ti (Ampere), bf16 is the safe default.

**In saLLMan.** `torch.amp.autocast(dtype=torch.bfloat16)` from Phase 1 onward - a core part of
the [[Memory budget]].

**Connects to:** [[Memory budget]] | [[TF32 and fused AdamW]] | [[Gradient checkpointing]]
