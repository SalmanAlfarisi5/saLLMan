"""
phase4/audit_tests.py — audit test-case coverage of open-r1/codeforces-cots
:solutions_py_decontaminated for Phase 4 GRPO planning.

Reports, over all rows:
  (1) how many rows have non-empty public_tests, non-empty private_tests,
      either, both.
  (2) the structure of public_tests / private_tests entries: type, keys,
      a printed (input, expected_output) example so we can confirm the
      format (presumably stdin / stdout strings).
  (3) the GRPO problem pool — rows with description + input_format +
      output_format + at least one usable test source. Does NOT require
      'editorial' (unlike SFT), so this pool may be larger than the 3,704
      SFT examples.

No training, no GPU. Read-only over the dataset.

Usage
-----
    cd phase4
    python audit_tests.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm


REPO   = "open-r1/codeforces-cots"
CONFIG = "solutions_py_decontaminated"
SPLIT  = "train"


# ---------------------------------------------------------------------------
# Helpers — defensive predicates on the test-field shape.
#
# Verified expected shape (we'll print actual structure later, but this is
# what `public_tests` and `private_tests` look like per the dataset card):
#
#     {"input":  [str, str, ...],
#      "output": [str, str, ...]}
#
# Both lists are parallel: input[i] is the stdin string for test i,
# output[i] is the expected stdout string. Non-empty means both lists have
# at least one element AND the lengths agree.
# ---------------------------------------------------------------------------
def _is_non_empty_tests(tests) -> bool:
    if not isinstance(tests, dict):
        return False
    inputs  = tests.get("input")  or []
    outputs = tests.get("output") or []
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return False
    return len(inputs) > 0 and len(inputs) == len(outputs)


def _is_non_empty_str(x) -> bool:
    return isinstance(x, str) and bool(x.strip())


# ---------------------------------------------------------------------------
# Structure print — show the user the actual schema of a test entry.
# ---------------------------------------------------------------------------
def _print_test_structure(name: str, samples: list[dict]) -> None:
    print(f"\n=== {name} structure (3 samples) ===")
    for i, sample in enumerate(samples, start=1):
        print(f"--- sample {i} ---")
        print(f"  type: {type(sample).__name__}")
        if isinstance(sample, dict):
            print(f"  keys: {sorted(sample.keys())}")
            for k in sample:
                v = sample[k]
                n = len(v) if hasattr(v, "__len__") else None
                print(f"    {k!r}: type={type(v).__name__}"
                      + (f", len={n}" if n is not None else ""))
            inputs  = sample.get("input")  or []
            outputs = sample.get("output") or []
            if inputs and outputs:
                tin  = inputs[0]
                tout = outputs[0]
                # Truncate display to keep the audit readable.
                tin_disp  = tin[:160]  + ("..." if len(tin)  > 160 else "")
                tout_disp = tout[:160] + ("..." if len(tout) > 160 else "")
                print(f"    first (input, expected_output) pair:")
                print(f"      input  ({len(tin)} chars): {tin_disp!r}")
                print(f"      output ({len(tout)} chars): {tout_disp!r}")


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading {REPO}:{CONFIG} split={SPLIT} ...")
    ds = load_dataset(REPO, CONFIG, split=SPLIT, token=hf_token)
    print(f"Total rows: {len(ds):,}")
    print(f"Columns:    {ds.column_names}")

    # ── Single pass: count coverage + collect structure samples + count
    # ── GRPO problem pool. Streaming would also work but the full dataset
    # ── fits in memory once and we want exact counts.
    n_public  = 0
    n_private = 0
    n_either  = 0
    n_both    = 0
    n_grpo_pool = 0

    # Per-row test counts for distribution stats
    n_pub_tests_per_row  : list[int] = []
    n_priv_tests_per_row : list[int] = []

    public_samples  : list[dict] = []
    private_samples : list[dict] = []

    # GRPO-pool counts also report each contributing filter so we see
    # which constraint loses the most rows.
    n_missing_desc   = 0
    n_missing_inputf = 0
    n_missing_outf   = 0
    n_missing_tests  = 0

    for row in tqdm(ds, desc="auditing"):
        pub  = row.get("public_tests")
        priv = row.get("private_tests")

        has_pub  = _is_non_empty_tests(pub)
        has_priv = _is_non_empty_tests(priv)

        if has_pub:
            n_public += 1
            n_pub_tests_per_row.append(len(pub["input"]))
            if len(public_samples) < 3:
                public_samples.append(pub)
        if has_priv:
            n_private += 1
            n_priv_tests_per_row.append(len(priv["input"]))
            if len(private_samples) < 3:
                private_samples.append(priv)
        if has_pub or has_priv:
            n_either += 1
        if has_pub and has_priv:
            n_both += 1

        # GRPO problem pool
        has_desc   = _is_non_empty_str(row.get("description"))
        has_inputf = _is_non_empty_str(row.get("input_format"))
        has_outf   = _is_non_empty_str(row.get("output_format"))
        has_tests  = has_pub or has_priv

        if has_desc and has_inputf and has_outf and has_tests:
            n_grpo_pool += 1
        else:
            if not has_desc:   n_missing_desc   += 1
            if not has_inputf: n_missing_inputf += 1
            if not has_outf:   n_missing_outf   += 1
            if not has_tests:  n_missing_tests  += 1

    total = len(ds)

    # ── Report 1: coverage counts ──────────────────────────────────────────
    print("\n" + "=" * 64)
    print("TEST-FIELD COVERAGE")
    print("=" * 64)
    print(f"  total rows:                {total:,}")
    print(f"  public_tests  non-empty:   {n_public:,}  ({100*n_public/total:.1f}%)")
    print(f"  private_tests non-empty:   {n_private:,}  ({100*n_private/total:.1f}%)")
    print(f"  either:                    {n_either:,}  ({100*n_either/total:.1f}%)")
    print(f"  both:                      {n_both:,}  ({100*n_both/total:.1f}%)")

    if n_pub_tests_per_row:
        s = sorted(n_pub_tests_per_row)
        print(f"\n  per-row public_tests count: "
              f"min={s[0]}  p50={s[len(s)//2]}  p95={s[int(len(s)*0.95)]}  max={s[-1]}")
    if n_priv_tests_per_row:
        s = sorted(n_priv_tests_per_row)
        print(f"  per-row private_tests count: "
              f"min={s[0]}  p50={s[len(s)//2]}  p95={s[int(len(s)*0.95)]}  max={s[-1]}")

    # ── Report 2: schema sample ────────────────────────────────────────────
    _print_test_structure("public_tests",  public_samples)
    _print_test_structure("private_tests", private_samples)

    # ── Report 3: GRPO problem pool ────────────────────────────────────────
    print("\n" + "=" * 64)
    print("GRPO PROBLEM POOL")
    print("=" * 64)
    print(f"  rows with description + input_format + output_format + tests:")
    print(f"      {n_grpo_pool:,}  ({100*n_grpo_pool/total:.1f}% of total)")
    print(f"\n  Disqualifying-filter counts (a row may fail >1):")
    print(f"    missing description:    {n_missing_desc:,}")
    print(f"    missing input_format:   {n_missing_inputf:,}")
    print(f"    missing output_format:  {n_missing_outf:,}")
    print(f"    missing public+private: {n_missing_tests:,}")
    print(f"\n  Compare to SFT set: 3,704 examples (which additionally required")
    print(f"  a non-empty 'editorial' field and finish under 2,048 tokens).")
    print(f"  GRPO pool does NOT require editorial — expect this number to be")
    print(f"  larger than the SFT count.")


if __name__ == "__main__":
    main()
