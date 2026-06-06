# Tokenization thread

**Cluster:** cross-cutting hub · part of [[Home]]

## Intuition
A single [[Byte-level BPE]] vocabulary threads through *every* phase of saLLMan and quietly defines a lot:

- It sets the **embedding table** and, via [[Weight tying and embedding scaling]], the output projection.
- Its **special tokens** matter — `<|endoftext|>` for document boundaries, and `<think>`/`</think>` for [[Chain-of-thought]] reasoning traces in [[Phase 3 - Production-scale code pretraining and SFT]].
- Its tokens are the **units counted** in [[Chinchilla scaling laws]] (tokens-per-parameter).
- Its tokens are what gets **scored** by [[pass@k]] at eval time.

## Why it matters for saLLMan
Because it's code-focused, byte-level BPE is the right choice: it never emits `<unk>` (every byte is representable), which matters for source code full of rare symbols and whitespace. The same vocab must be used consistently from pretraining through eval — a mismatch silently corrupts everything downstream.

## Connects to
[[Byte-level BPE]] · [[Subword tokenization]] · [[Weight tying and embedding scaling]] · [[Chinchilla scaling laws]] · [[pass@k]] · [[Chain-of-thought]]
