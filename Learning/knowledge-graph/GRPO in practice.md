# GRPO in practice (TRL / verifiers / open-r1)

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** How people actually run [[GRPO]]:
- **Hugging Face TRL** provides a `GRPOTrainer`.
- **open-r1** supplies recipes, a [[Code-execution reward]] function, and a vLLM backend for fast rollouts.
- The "verifiers" pattern wraps reward functions as composable verifiers.

**For 8 GB ([[Memory budget]]).** Expect a small policy (<=1.5B) + LoRA/quantization + gradient
checkpointing + tiny group sizes + colocated/vLLM generation. Full-parameter GRPO on a 7B model
will not fit - even the open-r1 team OOM'd scaling a 32B SFT past 20k context on multi-node H100s.

## References
- huggingface/open-r1 (open reproduction of DeepSeek-R1).
- Hugging Face TRL `GRPOTrainer` docs.

**Connects to:** [[GRPO]] | [[Code-execution reward]] | [[Memory budget]]
