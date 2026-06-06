# Phase 3 - Production-scale code pretraining + SFT

Scales the Phase 2 architecture to ~97M params / 2048 context and pretrains on Python code, then
supervised-fine-tunes on reasoning traces. The architecture file just adds [[Gradient checkpointing]];
everything else is data + training engineering.

## Concepts
[[Gradient checkpointing]] | [[Chinchilla scaling laws]] | [[The Stack dataset]] | [[Memory-mapped data loading]] | [[Gradient accumulation]] | [[Overfitting and train-val divergence]] | [[Supervised fine-tuning]] | [[Masked loss]] | [[Chain-of-thought]] | [[codeforces-cots]] | [[DeepSeek-R1]] | [[Code LLMs]] | [[TF32 and fused AdamW]] | [[Dynamic padding]]

## In saLLMan
`phase3/pretrain.py` (memmap loader, full resume) + `phase3/finetune.py` (masked-loss SFT). Feeds
the Phase 4 [[GRPO]] policy and the Phase 5 [[pass@k]] evaluation.

**Connects to:** [[Phase 2 - LLaMA-class modernized decoder]] | [[GRPO]] | [[Architecture progression]]
