# Phase 3 - Production-scale code pretraining + SFT

Scales the Phase 2 architecture to ~97M params / 2048 context and pretrains on Python code, then
supervised-fine-tunes on reasoning traces. The architecture file just adds [[Gradient checkpointing]];
everything else is data + training engineering.

## Concepts
[[Gradient checkpointing]] | [[Chinchilla scaling laws]] | [[The Stack dataset]] | [[Memory-mapped data loading]] | [[Gradient accumulation]] | [[Overfitting and train-val divergence]] | [[Supervised fine-tuning]] | [[Masked loss]] | [[Chain-of-thought]] | [[codeforces-cots]] | [[DeepSeek-R1]] | [[Code LLMs]] | [[TF32 and fused AdamW]] | [[Dynamic padding]]

## Process lessons (this session)
[[Schema verification before coding]] | [[Context length from data]] | [[Loss-mask invariants]]

## In saLLMan
`phase3/pretrain.py` (memmap loader, full resume) + `phase3/finetune.py` (masked-loss SFT). Feeds
the Phase 4 [[GRPO]] policy and the Phase 5 [[pass@k]] evaluation.

## Final results (Phase 3 v2)

**Pretrain** (97 M params, 2048 ctx, 2.20 B train tokens, 60 k steps, ~3.6 epochs):
- Final train loss **1.13**, val loss **1.15** (train/val tracked within ~0.02 nats throughout - no
  overfitting; the v1 failure mode was avoided by the Chinchilla-class data budget).
- Throughput ~22.5 k tok/s on the RTX 3060 Ti.

**SFT** (3 519 train / 185 val from `open-r1/codeforces-cots:solutions_py_decontaminated`, 2 epochs,
effective batch 32, [[Masked loss]] on response tokens only):
- Final train loss **1.32**, val loss **1.38**. Loss dropped sharply in the first ~40 steps (tag
  scaffolding), then plateaued.
- `test_sft_generation.py` confirms structured `<problem>...<reasoning>...<code>...</code><eos>`
  output on 4 held-out DSA problems; clean EOS stop in 3 of 4.

**Connects to:** [[Phase 2 - LLaMA-class modernized decoder]] | [[GRPO]] | [[Architecture progression]]
