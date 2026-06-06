# saLLMan — Knowledge Graph Home

Entry point for studying the saLLMan project: a from-scratch PyTorch transformer
progression (Phase 0 -> Phase 5) ending in **saLLMan**, a decoder-only LLM for DSA
(Data Structures & Algorithms) reasoning, trained on an 8 GB RTX 3060 Ti.

Each note follows the same shape: **Intuition -> Formula/Mechanism -> Paper -> Connects to**.
Follow the links to walk the graph. Start with [[Architecture progression]] for the big picture.

## How to read this vault
- The four **hub notes** are the spine; every phase links back to them.
- Phase notes bundle the concepts introduced in that phase.
- Phases 4 and 5 are **not yet implemented**, so those notes are deeper (new material).

## Hub notes (read first)
- [[Architecture progression]]
- [[Residual stream]]
- [[Tokenization thread]]
- [[Next-token prediction]]
- [[Memory budget]]

## Phase 0 - Vanilla Transformer
[[Transformer architecture]] | [[Scaled dot-product attention]] | [[Multi-head attention]] | [[Sinusoidal positional encoding]] | [[Position-wise feed-forward network]] | [[LayerNorm]] | [[Post-LN]] | [[Encoder vs decoder cross-attention]] | [[Causal and padding masking]] | [[Weight tying and embedding scaling]] | [[Teacher forcing]] | [[Label smoothing]] | [[Noam scheduler]] | [[Xavier initialization]] | [[Adam optimizer]] | [[BLEU perplexity and decoding]] | [[Multi30k dataset]]

## Phase 1 - Decoder-only GPT
[[Decoder-only GPT architecture]] | [[Byte-level BPE]] | [[Subword tokenization]] | [[Block packing]] | [[Mixed precision training]] | [[WikiText-2 dataset]]

## Phase 2 - LLaMA-class modernized decoder
[[Pre-LN]] | [[RMSNorm]] | [[RoPE]] | [[SwiGLU]] | [[FlashAttention]] | [[KV-cache]] | [[GPT-2 initialization]] | [[AdamW]] | [[Cosine schedule]] | [[LLaMA architecture]] | [[Bias-free linear layers]]

## Phase 3 - Production-scale code pretraining + SFT
[[Gradient checkpointing]] | [[Chinchilla scaling laws]] | [[The Stack dataset]] | [[Memory-mapped data loading]] | [[Gradient accumulation]] | [[Overfitting and train-val divergence]] | [[Supervised fine-tuning]] | [[Masked loss]] | [[Chain-of-thought]] | [[codeforces-cots]] | [[DeepSeek-R1]] | [[Code LLMs]] | [[TF32 and fused AdamW]] | [[Dynamic padding]]

## Phase 4 - GRPO reinforcement learning *(not yet implemented - deep notes)*
[[RLHF]] | [[PPO]] | [[Reward model vs RLVR]] | [[GRPO]] | [[DeepSeek-R1 with GRPO]] | [[Code-execution reward]] | [[KL regularization]] | [[Advantage estimation]] | [[Reward hacking]] | [[On-policy vs off-policy]] | [[GRPO in practice]] | [[GRPO log-prob bookkeeping]]

## Phase 5 - Evaluation *(not yet implemented - deep notes)*
[[pass@k]] | [[HumanEval]] | [[MBPP]] | [[LeetCode-style evaluation]] | [[Functional correctness]] | [[Test-case-based evaluation]] | [[Decontamination]] | [[Temperature sampling]] | [[Greedy vs sampling]]
