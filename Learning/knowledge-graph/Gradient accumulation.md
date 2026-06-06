# Gradient accumulation

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** Sum gradients over several micro-batches before one optimizer step, simulating a
large *effective* batch on limited VRAM. Scale the loss by `1/accum_steps` so the gradient matches
a single big-batch step.

**In saLLMan.** e.g. micro_batch=4 x accum=16 -> effective batch 64 -> 131,072 tokens/step at 2048
context (4 x 16 x 2048 = 131,072). Gradient clipping is applied to the *full accumulated* gradient,
after the loop.

**Connects to:** [[Gradient checkpointing]] | [[Mixed precision training]] | [[Memory budget]]
