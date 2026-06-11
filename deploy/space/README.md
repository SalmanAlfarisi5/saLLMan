---
title: saLLMan Demo
emoji: 🧮
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: A 97M DSA-reasoning LLM built from scratch on an 8GB GPU
---

# saLLMan — Step-aware LLM for Algorithm Navigation

A **97M-parameter decoder-only transformer** built and trained from scratch on a
single 8 GB RTX 3060 Ti: modern architecture (RoPE, SwiGLU, RMSNorm, Pre-LN),
2.2B-token pretraining on The Stack (Python), supervised fine-tuning on
Codeforces chain-of-thought data, and GRPO reinforcement learning with a
verifiable code-execution reward.

Give it a DSA / competitive-programming problem; it generates a `<reasoning>`
trace followed by `<code>`.

**Honest scope:** at 97M params this is a portfolio/research model — it produces
the right *structure* but the code is usually not correct. The value is the
from-scratch pipeline, including a documented reward-hacking finding in the RL
phase.

This Space runs the custom `GPTv3` model (not a 🤗 transformers model) on CPU.
The `generate` endpoint is callable via the Gradio HTTP API.
