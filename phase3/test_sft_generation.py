"""
test_sft_generation.py — smoke-test SFT generation on held-out problems.

Loads `checkpoints_finetune_v2/best.pt` and generates completions for a
handful of DSA problems that are NOT in the
open-r1/codeforces-cots:solutions_py_decontaminated training set.
Pure inference — no training, no gradient updates.

What to eyeball in the output:

  (1) The model produces reasoning text after the opening <reasoning> tag.
  (2) It emits </reasoning>\\n<code>\\n and switches into code.
  (3) It stops cleanly at </code><eos> rather than rambling.

If any of those three break, the SFT didn't learn the tag scaffolding —
re-check that finetune_data_v2/train.jsonl looks right via
verify_finetune_data.py first.

Usage
-----
    cd phase3
    python test_sft_generation.py
    python test_sft_generation.py --checkpoint checkpoints_finetune_v2/last.pt
    python test_sft_generation.py --temperature 0.0    # greedy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

# Project-root sys.path tweak — mirrors finetune.py / pretrain.py so the
# bare-module imports (decoder_only_v3, finetune, finetune_data_prep) work
# whether this script is run from project root or from phase3/.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the checkpoint-loader from finetune.py so any future change to
# the payload schema lives in exactly one place. The SFT checkpoint has
# the same layout as the pretrain checkpoint (model_cfg dict +
# model_state), so this loads either.
from finetune import load_pretrained
from finetune_data_prep import (
    PROBLEM_OPEN, PROBLEM_CLOSE, REASON_OPEN,
    BOS_IDX, EOS_IDX,
)


# Held-out DSA problems. None of these appear (verbatim) in
# open-r1/codeforces-cots:solutions_py_decontaminated — they're classic
# LeetCode-style prompts, not the Codeforces problems used for SFT.
HELD_OUT_PROBLEMS: list[str] = [
    "Reverse a singly linked list. Given the head of a singly linked "
    "list, return the head of the reversed list.",
    "Find the longest palindromic substring of a given string s. If "
    "multiple palindromes have the same length, return any of them.",
    "Merge two sorted arrays into a single sorted array. The result must "
    "also be sorted in non-decreasing order.",
    "Given an array of integers and an integer target, return the "
    "indices of the two numbers in the array that add up to target. "
    "You may assume each input has exactly one solution.",
]


def build_prompt(problem: str) -> str:
    """
    Construct the exact prefix used during SFT training.

    The boundary in training is: prompt ends at the opening <reasoning>
    tag, response begins at the reasoning content. Reproducing that
    boundary at inference puts the model in the same conditioning state
    it saw during gradient updates — it should immediately start emitting
    reasoning content.
    """
    return PROBLEM_OPEN + problem.strip() + PROBLEM_CLOSE + REASON_OPEN


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT generation smoke test.")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints_finetune_v2/best.pt"),
        help="SFT checkpoint to load (best.pt or last.pt).",
    )
    parser.add_argument(
        "--meta", type=Path,
        default=Path("finetune_data_v2/meta.json"),
        help="FT meta.json — used only to locate the BPE tokenizer path.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sampling: temperature={args.temperature}, top_k={args.top_k}, "
          f"max_new_tokens={args.max_new_tokens}")

    # ── Tokenizer ───────────────────────────────────────────────────────────
    if not args.meta.exists():
        raise SystemExit(f"meta.json not found at {args.meta}")
    meta = json.loads(args.meta.read_text())
    tokenizer_path = Path(meta["tokenizer_path"])
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Tokenizer: {tokenizer_path} (vocab={tokenizer.get_vocab_size():,})")

    # ── Checkpoint ──────────────────────────────────────────────────────────
    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found at {args.checkpoint}. "
            "Has finetune.py finished writing best.pt yet?"
        )
    # Force gradient_checkpointing off — this is inference; the flag
    # would be ignored under no_grad anyway, but keep it explicit.
    model, model_cfg = load_pretrained(
        args.checkpoint, gradient_checkpointing=False, device=device,
    )
    model.eval()
    print(f"Generating up to {args.max_new_tokens} new tokens per prompt "
          f"(model max_len={model_cfg.max_len}).\n")

    # ── Generate ────────────────────────────────────────────────────────────
    for i, problem in enumerate(HELD_OUT_PROBLEMS, start=1):
        prompt = build_prompt(problem)
        prompt_ids = [BOS_IDX] + tokenizer.encode(prompt).ids
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        out = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            eos_id=EOS_IDX,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # Skip BOS so the printed text starts at <problem>...
        decoded = tokenizer.decode(out[0, 1:].tolist())
        n_generated = out.size(1) - len(prompt_ids)
        stopped_at_eos = out[0, -1].item() == EOS_IDX
        stop_reason = "<eos>" if stopped_at_eos else "max_new_tokens"

        print("=" * 76)
        print(f"PROBLEM {i}: {problem}")
        print("=" * 76)
        print(decoded)
        print()
        print(f"  [generated {n_generated} new tokens, stopped at {stop_reason}]")
        print()


if __name__ == "__main__":
    main()
