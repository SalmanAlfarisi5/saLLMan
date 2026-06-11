"""
saLLMan demo — Gradio app for Hugging Face Spaces.

Loads the from-scratch 97M decoder-only model (GRPO-v2 checkpoint, fp16
weights upcast to fp32 for CPU) and exposes:
  - a chat-style UI for the Space page itself, and
  - an HTTP API (api_name="generate") that a portfolio website can call.

The model is NOT a transformers model — it's the custom GPTv3 class from the
saLLMan project, loaded from model_v3.py + the BPE tokenizer.

Honest scope: this is a 97M-parameter model trained on a single 8 GB GPU.
It produces structurally-correct <reasoning>/<code> output but the code is
usually not correct. The demo showcases the from-scratch pipeline
(architecture → pretraining → SFT → GRPO), not benchmark performance.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
from tokenizers import Tokenizer

from model_v3 import GPTv3, GPTConfigV3

# ---------------------------------------------------------------------------
# Tags + special-token ids — must match the SFT training format
# (phase3/finetune_data_prep.py).
# ---------------------------------------------------------------------------
PROBLEM_OPEN  = "<problem>\n"
PROBLEM_CLOSE = "\n</problem>\n"
REASON_OPEN   = "<reasoning>\n"
BOS_IDX, EOS_IDX = 1, 2

_HERE = Path(__file__).resolve().parent
CKPT_PATH = _HERE / "sallman_grpo_v2_fp16.pt"
TOK_PATH  = _HERE / "bpe_tokenizer.json"


# ---------------------------------------------------------------------------
# Load model + tokenizer once at startup (CPU).
# ---------------------------------------------------------------------------
def _load():
    payload = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = GPTConfigV3(**payload["model_cfg"])
    model = GPTv3(cfg)
    # fp16 weights upcast to fp32 on copy — fp16 on CPU is slow/unsupported.
    model.load_state_dict(payload["model_state"])
    model.eval()
    tok = Tokenizer.from_file(str(TOK_PATH))
    return model, tok, cfg


MODEL, TOKENIZER, CFG = _load()
print(f"[saLLMan] loaded {sum(p.numel() for p in MODEL.parameters())/1e6:.1f}M params, "
      f"max_len={CFG.max_len}, vocab={CFG.vocab_size}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_solution(problem: str,
                      temperature: float = 0.8,
                      top_k: int = 40,
                      max_new_tokens: int = 256) -> str:
    """Build the SFT-format prompt, generate, return the decoded completion.

    The returned string is the model's continuation after the opening
    <reasoning> tag — i.e. reasoning + code + closing tags.
    """
    problem = (problem or "").strip()
    if not problem:
        return "(enter a problem statement)"

    prompt = PROBLEM_OPEN + problem + PROBLEM_CLOSE + REASON_OPEN
    prompt_ids = [BOS_IDX] + TOKENIZER.encode(prompt).ids
    # Leave room for the response within the model's context window.
    max_new = int(min(max_new_tokens, CFG.max_len - len(prompt_ids) - 1))
    if max_new <= 0:
        return "(problem too long for the 2048-token context)"

    x = torch.tensor([prompt_ids], dtype=torch.long)
    out = MODEL.generate(
        x,
        max_new_tokens=max_new,
        eos_id=EOS_IDX,
        temperature=float(temperature),
        top_k=int(top_k) if top_k and top_k > 0 else None,
    )
    completion_ids = out[0, len(prompt_ids):].tolist()
    text = TOKENIZER.decode(completion_ids)
    # Present the full tagged block so the structure is visible.
    return f"<reasoning>\n{text}"


# ---------------------------------------------------------------------------
# Curated example prompts (the model's cleaner-structured cases).
# ---------------------------------------------------------------------------
EXAMPLES = [
    "Given an array of integers, return the indices of the two numbers that add up to a target.",
    "Reverse a singly linked list and return the new head.",
    "Check whether a string is a palindrome, ignoring spaces and case.",
    "Find the maximum sum of any contiguous subarray of an integer array.",
    "Merge two sorted arrays into one sorted array.",
]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
DESCRIPTION = """
# saLLMan — a 97M DSA-reasoning LLM, built from scratch

A decoder-only transformer (**RoPE · SwiGLU · RMSNorm · Pre-LN**) I built and
trained end-to-end on a single **8 GB RTX 3060 Ti**:
**2.2B-token pretraining → supervised fine-tuning → GRPO reinforcement learning**.

Give it a DSA / competitive-programming problem and it generates a
`<reasoning>` trace followed by `<code>`.

> ⚠️ **Honest scope:** at 97M parameters this is a *research/portfolio* model.
> It reliably produces the right *structure* (reasoning → code) but the code
> is usually **not correct**. The point is the from-scratch pipeline, not
> benchmark performance. [Project write-up & reward-hacking findings →](https://github.com/Salmanalfarisi1)
"""


with gr.Blocks(title="saLLMan demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=3):
            problem_in = gr.Textbox(
                label="Problem statement",
                placeholder="Describe a DSA problem…",
                lines=4,
            )
            with gr.Row():
                temp = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
                topk = gr.Slider(0, 100, value=40, step=1, label="Top-k (0 = off)")
                mnt  = gr.Slider(32, 512, value=256, step=16, label="Max new tokens")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=4):
            out = gr.Textbox(label="Model output", lines=18)

    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[problem_in])

    # api_name="generate" makes this callable from a portfolio website via the
    # Gradio HTTP API (POST /gradio_api/call/generate).
    go.click(
        generate_solution,
        inputs=[problem_in, temp, topk, mnt],
        outputs=[out],
        api_name="generate",
    )


if __name__ == "__main__":
    demo.queue(max_size=16).launch()
