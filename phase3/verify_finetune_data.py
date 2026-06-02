"""
verify_finetune_data.py — sanity-check finetune_data_prep.py output.

No training, no GPU. Loads the JSONL produced by finetune_data_prep.py,
re-loads the BPE tokenizer named in meta.json, and:

  Invariant A: len(input_ids) == len(loss_mask)  for every record
  Invariant B: loss_mask is exactly [0]*p + [1]*r (single 0→1 flip)
               for every record. Index of any violation is printed.

For the first --preview records (default 3), prints:
  - The FULL decoded input_ids (everything the model sees during training).
  - SEPARATELY, the decoded tokens where loss_mask==1.

The masked-only print is the visual confirmation that loss is gated to
exactly the response region (reasoning + code + <eos>) and that none of
the problem text leaks across the boundary. Eyeball the second print:
it should start at the reasoning text (no <problem> tag visible) and end
with </code>.

Usage
-----
    cd phase3
    python verify_finetune_data.py
    python verify_finetune_data.py --data finetune_data_v2 --preview 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def find_mask_flip_violation(mask: list[int]) -> int | None:
    """
    Return the first index that breaks the all-0-then-all-1 invariant,
    or None if the mask is valid.

    Valid shapes (any of):
      []                        — empty (shouldn't happen, but vacuously valid)
      [0, 0, ..., 0]            — all zero (no response tokens; suspicious)
      [1, 1, ..., 1]            — all one  (no prompt; very suspicious)
      [0, ..., 0, 1, ..., 1]    — single 0→1 flip (expected)

    Invalid: any 0 appearing AFTER any 1, or any value outside {0, 1}.
    """
    seen_one = False
    for i, v in enumerate(mask):
        if v not in (0, 1):
            return i
        if v == 1:
            seen_one = True
        elif seen_one:   # v == 0 after a 1 → violation
            return i
    return None


def print_preview(idx: int, input_ids: list[int], mask: list[int],
                  tokenizer: Tokenizer) -> None:
    """Print full decoded input_ids, then the mask==1 region separately."""
    n = len(input_ids)
    boundary = mask.index(1) if 1 in mask else n   # first 1 position
    response_ids = [tid for tid, m in zip(input_ids, mask) if m == 1]

    print(f"  === record {idx}: {n} tokens "
          f"(prompt {boundary}, response {n - boundary}) ===")
    print(f"  --- FULL decoded input_ids ---")
    full = tokenizer.decode(input_ids)
    for line in full.split("\n"):
        print(f"  | {line}")
    print(f"  --- LOSS-MASKED tokens only (mask==1), {len(response_ids)} ids ---")
    masked = tokenizer.decode(response_ids)
    for line in masked.split("\n"):
        print(f"  | {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify finetune_data_prep.py output."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("finetune_data_v2"),
        help="Directory containing train.jsonl and meta.json.",
    )
    parser.add_argument(
        "--preview", type=int, default=3,
        help="Number of records to print decoded previews for.",
    )
    args = parser.parse_args()

    train_path = args.data / "train.jsonl"
    meta_path = args.data / "meta.json"

    if not meta_path.exists():
        raise SystemExit(f"meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text())

    n_train = meta.get("n_train")
    n_val = meta.get("n_val")
    max_len = meta.get("max_len")
    vocab_size = meta.get("vocab_size")
    print(
        f"meta.json: n_train={n_train}  n_val={n_val}  "
        f"max_len={max_len}  vocab_size={vocab_size}"
    )

    tokenizer_path = Path(meta["tokenizer_path"])
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(
        f"Loaded tokenizer: {tokenizer_path} "
        f"(vocab={tokenizer.get_vocab_size():,})\n"
    )

    if not train_path.exists():
        raise SystemExit(f"train.jsonl not found at {train_path}")

    # ── Invariant checks + previews ─────────────────────────────────────────
    n_records = 0
    n_length_mismatch = 0
    n_mask_violation = 0
    n_only_zero = 0      # info: loss_mask has no 1s (record contributes no loss)
    n_only_one = 0       # info: loss_mask has no 0s (no prompt — suspicious)
    preview_left = max(0, args.preview)

    print(f"[verify] reading {train_path}")
    with train_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            n_records += 1
            rec = json.loads(line)
            input_ids = rec["input_ids"]
            mask = rec["loss_mask"]

            # Invariant A — length match
            if len(input_ids) != len(mask):
                n_length_mismatch += 1
                print(
                    f"  ! record {line_idx}: length mismatch "
                    f"(input_ids={len(input_ids)}, loss_mask={len(mask)})"
                )

            # Invariant B — single 0→1 flip
            viol = find_mask_flip_violation(mask)
            if viol is not None:
                n_mask_violation += 1
                context_lo = max(0, viol - 3)
                context_hi = min(len(mask), viol + 4)
                print(
                    f"  ! record {line_idx}: mask flip violation at index "
                    f"{viol} (value={mask[viol]}; "
                    f"mask[{context_lo}:{context_hi}]={mask[context_lo:context_hi]})"
                )

            # Informational edge-case counts (don't fail verification).
            if 1 not in mask:
                n_only_zero += 1
            elif 0 not in mask:
                n_only_one += 1

            if preview_left > 0:
                print_preview(line_idx, input_ids, mask, tokenizer)
                preview_left -= 1

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\nChecked {n_records:,} train records:")
    print(
        f"  invariant A — len(input_ids) == len(loss_mask): "
        f"{'OK' if n_length_mismatch == 0 else f'FAILED ({n_length_mismatch} mismatches)'}"
    )
    print(
        f"  invariant B — mask is all-0-then-all-1 (single flip): "
        f"{'OK' if n_mask_violation == 0 else f'FAILED ({n_mask_violation} violations)'}"
    )
    print(f"  info: records with no response tokens (all-zero mask): {n_only_zero}")
    print(f"  info: records with no prompt tokens   (all-one mask):  {n_only_one}")

    if n_length_mismatch == 0 and n_mask_violation == 0:
        print("\nVerification PASSED.")
    else:
        raise SystemExit("\nVerification FAILED — see violations above.")


if __name__ == "__main__":
    main()
