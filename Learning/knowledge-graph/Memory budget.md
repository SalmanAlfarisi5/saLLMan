# Memory budget

**Cluster:** cross-cutting hub (8 GB RTX 3060 Ti) · part of [[saLLMan MOC]]

## Why this note exists
On an 8 GB card the binding constraint is **memory**, not compute. These are the concepts that buy you headroom, gathered in one place so you can see the whole survival kit.

## The survival kit
- **[[Mixed precision training]]** (bf16) — halves activation/parameter memory; bf16 needs no loss scaling.
- **[[Gradient checkpointing]]** — recompute activations in backward instead of storing them; ~30 % more compute for a large memory saving.
- **[[Gradient accumulation]]** — simulate a big effective batch by summing grads over micro-batches.
- **[[FlashAttention]]** — O(N) attention memory instead of O(N²); never materialises the score matrix.
- **[[KV-cache]]** — makes generation/rollouts cheap at inference time.
- **[[GRPO]]** — the single biggest RL-phase saving: it keeps **two** model copies in memory (policy + reference) instead of PPO's **three** (policy + reference + value), because the group-mean baseline replaces the value network.

## Phase-4 reality check
Full-parameter GRPO on a 7B model will *not* fit 8 GB. Plan for a ≤1.5B policy, LoRA / quantization, gradient checkpointing, tiny group sizes, and a fast rollout backend (e.g. vLLM). Even the open-r1 team OOM'd scaling a 32B SFT past 20k context on multi-node H100s — scale matters.

## Connects to
[[Mixed precision training]] · [[Gradient checkpointing]] · [[Gradient accumulation]] · [[FlashAttention]] · [[KV-cache]] · [[GRPO]] · [[Chinchilla scaling laws]]
