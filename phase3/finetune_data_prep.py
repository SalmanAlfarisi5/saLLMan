"""
saLLMan Phase 3 (fine-tune) — data prep for supervised fine-tuning.

Source (verified)
-----------------
  open-r1/codeforces-cots, config 'solutions_py_decontaminated', split 'train'.
  Each row is a Codeforces problem with a DeepSeek-R1 chain-of-thought trace,
  a Python solution, and a separate human-written editorial.

Verified schema (treated as ground truth — no field-name probing):
  Per-problem fields:
    description     str — task statement
    input_format    str — input specification
    output_format   str — output specification
    editorial       str — concise human-written reasoning / approach summary
  Generation fields:
    messages        list[dict], len == 2
      messages[0]   = {"role":"user",      "content": <full problem>}
      messages[1]   = {"role":"assistant", "content": <R1 output>}
    generation      str — equal to messages[1].content (used as fallback)
    finish_reason   str — "stop" | "length" | ...  (informational, NOT a filter)
  Test fields (consumed by Phase 4 GRPO, ignored here):
    public_tests, private_tests, generated_tests

What we use from each row
-------------------------
  problem    = description + "\\n\\n" + input_format + "\\n\\n" + output_format
               (drops the Codeforces "story" wrapper in messages[0].content
                that bloats most rows past 2048 tokens)
  reasoning  = editorial
               (the R1 <think> trace has a ~15k-token median — far past the
                2048 context; editorial is the budgeted human-written
                replacement)
  code       = first "```python ... ```" block AFTER "</think>" in the
               assistant output (or 'generation' as fallback). "</think>"
               is used only as a positional marker; we do NOT extract
               anything between the <think> tags.

finish_reason is NOT a filter
-----------------------------
We don't skip rows with finish_reason="length". Reasoning comes from
editorial, not from the assistant output, so a length-truncated R1
generation is fine as long as it still has a well-formed ```python block.
The fence parser validates that. We do still COUNT the finish_reason
distribution for visibility — surprises in that distribution are worth
seeing.

Output format (tagged sections — no new special tokens needed):
    <problem>
    {trimmed problem text}
    </problem>
    <reasoning>           ← prompt ends here, response begins
    {editorial}
    </reasoning>
    <code>
    {solution}
    </code><eos>

Loss masking
------------
Prompt and response are tokenized separately so the boundary in id space is
bit-exact:
    input_ids = [BOS] + prompt_ids + response_ids + [EOS]
    loss_mask = [0]   + [0]*L_p    + [1]*L_r      + [1]
At training time, positions with loss_mask=0 are replaced with -100 in
labels so F.cross_entropy(ignore_index=-100) skips them.

JSONL output (one example per line):
    {"input_ids": [int, ...], "loss_mask": [0/1, ...], "source": "..."}

Verified length stats with this trim+editorial form:
  p50 = 1278 tokens, p90 = 2420 tokens
  ~3846 of 4660 candidate rows fit at max_len=2048.

LeetCode is intentionally NOT a training source — it has no reasoning
traces. Reserved for Phase 5 evaluation.

Usage
-----
    cd phase3
    python finetune_data_prep.py
    python finetune_data_prep.py --out finetune_data_v2 \\
                                 --tokenizer pretrain_data_v2/bpe_tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tokenizers import Tokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Special token ids (must match pretrain_data_v2/bpe_tokenizer.json)
# ---------------------------------------------------------------------------
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Tag strings — boundaries the model will learn to emit
# ---------------------------------------------------------------------------
PROBLEM_OPEN  = "<problem>\n"
PROBLEM_CLOSE = "\n</problem>\n"
REASON_OPEN   = "<reasoning>\n"          # response starts AFTER this
REASON_CLOSE  = "\n</reasoning>\n"
CODE_OPEN     = "<code>\n"
CODE_CLOSE    = "\n</code>"


# ---------------------------------------------------------------------------
# Extractor stats — caller-owned so the summary remains valid even if
# iteration is interrupted (exception mid-tokenize, future head/limit, etc).
#
# Two independent partitions of n_rows_seen are tracked:
#
#   (1) finish_reason distribution     — informational, every row contributes
#       n_finish_stop + n_finish_length + n_finish_other == n_rows_seen
#
#   (2) outcome partition              — every row ends in exactly one of:
#       n_yielded OR one of the skip_* buckets, summing to n_rows_seen.
#
# Both invariants are asserted at print time.
# ---------------------------------------------------------------------------
@dataclass
class ExtractStats:
    n_rows_seen:                int = 0
    n_yielded:                  int = 0
    n_used_generation_fallback: int = 0

    # Informational — distribution of finish_reason across ALL rows seen,
    # not just yielded ones. NOT a skip filter.
    n_finish_stop:              int = 0
    n_finish_length:            int = 0
    n_finish_other:             int = 0

    # Skip buckets (part of the outcome partition).
    skip_no_editorial:          int = 0
    skip_no_description:        int = 0
    skip_no_input_format:       int = 0
    skip_no_output_format:      int = 0
    skip_malformed_msgs:        int = 0
    skip_no_think_close:        int = 0
    skip_no_py_fence:           int = 0
    skip_unclosed_fence:        int = 0
    skip_empty_code:            int = 0


def print_extract_stats(name: str, s: ExtractStats) -> None:
    """
    Print a fixed-width summary and assert both row-accounting invariants.

    A future refactor that adds a new skip branch without a counter will
    trip the outcome-partition assertion; a future refactor that adds a
    new finish_reason bucket without updating the three-way tally will
    trip the finish-reason invariant.
    """
    total_skipped = (
        s.skip_no_editorial
        + s.skip_no_description
        + s.skip_no_input_format
        + s.skip_no_output_format
        + s.skip_malformed_msgs
        + s.skip_no_think_close
        + s.skip_no_py_fence
        + s.skip_unclosed_fence
        + s.skip_empty_code
    )
    assert s.n_yielded + total_skipped == s.n_rows_seen, (
        f"outcome partition mismatch: yielded={s.n_yielded} + "
        f"skipped={total_skipped} != seen={s.n_rows_seen}"
    )
    finish_total = s.n_finish_stop + s.n_finish_length + s.n_finish_other
    assert finish_total == s.n_rows_seen, (
        f"finish_reason partition mismatch: "
        f"stop+length+other={finish_total} != seen={s.n_rows_seen}"
    )

    print(f"\n  {name} extraction over {s.n_rows_seen:,} rows:")
    print(f"      yielded:                              {s.n_yielded:,}")
    print(f"      used 'generation' fallback:           {s.n_used_generation_fallback:,}")
    print(f"    finish_reason distribution (informational, not a filter):")
    print(f"      stop:                                 {s.n_finish_stop:,}")
    print(f"      length:                               {s.n_finish_length:,}")
    print(f"      other:                                {s.n_finish_other:,}")
    print(f"    skip buckets:")
    print(f"      skip (no editorial):                  {s.skip_no_editorial:,}")
    print(f"      skip (no description):                {s.skip_no_description:,}")
    print(f"      skip (no input_format):               {s.skip_no_input_format:,}")
    print(f"      skip (no output_format):              {s.skip_no_output_format:,}")
    print(f"      skip (malformed messages):            {s.skip_malformed_msgs:,}")
    print(f"      skip (no </think>):                   {s.skip_no_think_close:,}")
    print(f"      skip (no ```python after </think>):   {s.skip_no_py_fence:,}")
    print(f"      skip (unclosed ``` fence):            {s.skip_unclosed_fence:,}")
    print(f"      skip (empty code):                    {s.skip_empty_code:,}")


# ---------------------------------------------------------------------------
# Source loader
# ---------------------------------------------------------------------------
def try_load_open_r1_codeforces_cots(hf_token: str | None) -> Dataset | None:
    """
    Load open-r1/codeforces-cots, config 'solutions_py_decontaminated',
    train split. Returns None on load failure so the caller can report
    cleanly rather than the whole script crashing on an HF outage.
    """
    print("[load] open-r1/codeforces-cots (solutions_py_decontaminated) ...")
    try:
        d = load_dataset(
            "open-r1/codeforces-cots",
            "solutions_py_decontaminated",
            split="train",
            token=hf_token,
        )
    except Exception as e:
        print(f"  ! failed to load open-r1/codeforces-cots: {e}")
        return None
    print(f"      rows: {len(d):,}, columns: {d.column_names}")
    return d


# ---------------------------------------------------------------------------
# Problem-text assembly
# ---------------------------------------------------------------------------
def build_trimmed_problem(description: str, input_format: str, output_format: str) -> str:
    """
    Build the trimmed problem text from three structured Codeforces fields:
      description     — the actual task statement
      input_format    — input specification
      output_format   — output specification
    Joined with blank lines (no headers, no flavor narrative).

    Verified length stats with this trimmed form + the editorial as reasoning:
      p50 = 1278 tokens
      p90 = 2420 tokens
      ~3846 of 4660 candidate rows fit at max_len=2048.

    The full messages[0].content includes a Codeforces "story" wrapper that
    pushes most rows well past 2048; this trimmed form is the budgeted
    replacement.
    """
    return "\n\n".join([description.strip(), input_format.strip(), output_format.strip()])


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
def extract_open_r1_codeforces_cots(
    ds: Dataset,
    stats: ExtractStats,
) -> Iterator[tuple[str, str, str]]:
    """
    Yield (problem, reasoning, code) triples.

    Sources, per the verified schema:
      problem    = build_trimmed_problem(description, input_format, output_format)
      reasoning  = row["editorial"]                  ← concise, human-written
      code       = first ```python block AFTER "</think>" in messages[1].content
                   (with row["generation"] as fallback when content is empty).

    The R1 <think>...</think> trace is intentionally NOT used as reasoning —
    its median length is ~15k tokens, far past a 2048 context. We still
    need "</think>" as a positional marker to locate the first ```python
    fence reliably; we don't extract anything between the tags.

    finish_reason is COUNTED but NOT used as a filter. Editorial gives us a
    self-contained reasoning trace, so a length-truncated R1 generation
    that still has a well-formed code block is usable.

    Skip rules (every row ends in exactly one of these buckets OR n_yielded):
      - editorial empty                         → skip_no_editorial
      - description empty                       → skip_no_description
      - input_format empty                      → skip_no_input_format
      - output_format empty                     → skip_no_output_format
      - messages malformed / output empty       → skip_malformed_msgs
      - "</think>" missing                      → skip_no_think_close
      - "```python" missing after "</think>"    → skip_no_py_fence
      - "```" closer missing                    → skip_unclosed_fence
      - code empty after stripping              → skip_empty_code
    """
    THINK_CLOSE = "</think>"
    PY_FENCE    = "```python"
    FENCE_CLOSE = "```"

    for row in ds:
        stats.n_rows_seen += 1

        # Tally finish_reason for visibility — informational, not a filter.
        fr = row.get("finish_reason")
        if fr == "stop":
            stats.n_finish_stop += 1
        elif fr == "length":
            stats.n_finish_length += 1
        else:
            stats.n_finish_other += 1

        # Cheap field-level filters first.
        editorial = (row.get("editorial") or "").strip()
        if not editorial:
            stats.skip_no_editorial += 1
            continue

        description = (row.get("description") or "").strip()
        if not description:
            stats.skip_no_description += 1
            continue
        input_format = (row.get("input_format") or "").strip()
        if not input_format:
            stats.skip_no_input_format += 1
            continue
        output_format = (row.get("output_format") or "").strip()
        if not output_format:
            stats.skip_no_output_format += 1
            continue

        # Messages → assistant output (code source only).
        msgs = row.get("messages") or []
        if not (
            len(msgs) >= 2
            and isinstance(msgs[0], dict) and msgs[0].get("role") == "user"
            and isinstance(msgs[1], dict) and msgs[1].get("role") == "assistant"
        ):
            stats.skip_malformed_msgs += 1
            continue

        output = (msgs[1].get("content") or "").strip()
        if not output:
            output = (row.get("generation") or "").strip()
            if output:
                stats.n_used_generation_fallback += 1
        if not output:
            stats.skip_malformed_msgs += 1
            continue

        # Locate </think> only as a positional marker for the code search.
        close_idx = output.find(THINK_CLOSE)
        if close_idx == -1:
            stats.skip_no_think_close += 1
            continue
        after_think = output[close_idx + len(THINK_CLOSE):]

        fence_open = after_think.find(PY_FENCE)
        if fence_open == -1:
            stats.skip_no_py_fence += 1
            continue
        code_start = fence_open + len(PY_FENCE)
        # Skip a single newline immediately after the fence marker.
        if code_start < len(after_think) and after_think[code_start] == "\n":
            code_start += 1
        fence_close = after_think.find(FENCE_CLOSE, code_start)
        if fence_close == -1:
            stats.skip_unclosed_fence += 1
            continue
        code = after_think[code_start:fence_close].strip()
        if not code:
            stats.skip_empty_code += 1
            continue

        problem = build_trimmed_problem(description, input_format, output_format)

        stats.n_yielded += 1
        yield problem, editorial, code


# ---------------------------------------------------------------------------
# Formatting + tokenization
# ---------------------------------------------------------------------------
def format_prompt_response(problem: str, reasoning: str, code: str) -> tuple[str, str]:
    """
    Build the (prompt, response) string pair.

    The boundary is set so the model conditions on the problem and the
    OPENING <reasoning> tag, then has to produce everything else.
    Putting the open tag in the prompt makes it easy to:
      - At inference time, build the same prompt and call .generate(...);
      - The model knows to start by emitting reasoning content.
    """
    prompt = (
        PROBLEM_OPEN + problem.strip() + PROBLEM_CLOSE
        + REASON_OPEN
    )
    response = (
        reasoning.strip() + REASON_CLOSE
        + CODE_OPEN + code.strip() + CODE_CLOSE
    )
    return prompt, response


def tokenize_example(
    tokenizer: Tokenizer,
    problem: str,
    reasoning: str,
    code: str,
    max_len: int,
) -> dict | None:
    """
    Encode one (problem, reasoning, code) triple into the JSONL record
    format, or return None if the example doesn't fit in max_len.

    Tokenizing prompt and response separately is what gives us a bit-exact
    boundary in id space; do NOT collapse to a single encode + delimiter
    search (BPE may merge tags differently depending on neighbours).

    Layout:
        input_ids = [BOS] + prompt_ids + response_ids + [EOS]
        loss_mask = [0]   + [0]*L_p    + [1]*L_r      + [1]
                                                        ↑ predict EOS too,
                                                          so generation
                                                          learns to stop.
    """
    prompt_text, response_text = format_prompt_response(problem, reasoning, code)

    prompt_ids = tokenizer.encode(prompt_text).ids
    response_ids = tokenizer.encode(response_text).ids

    input_ids = [BOS_IDX] + prompt_ids + response_ids + [EOS_IDX]
    if len(input_ids) > max_len:
        return None

    loss_mask = [0] + [0] * len(prompt_ids) + [1] * len(response_ids) + [1]
    assert len(input_ids) == len(loss_mask)

    return {"input_ids": input_ids, "loss_mask": loss_mask}


# ---------------------------------------------------------------------------
# Write JSONL with deterministic train/val split
# ---------------------------------------------------------------------------
def write_split(
    records: list[dict],
    train_path: Path,
    val_path: Path,
    val_ratio: float,
    seed: int,
) -> tuple[int, int]:
    """Shuffle then split deterministically. Returns (n_train, n_val)."""
    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * val_ratio))
    val, train = records[:n_val], records[n_val:]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    for path, batch in [(train_path, train), (val_path, val)]:
        with path.open("w", encoding="utf-8") as f:
            for rec in batch:
                f.write(json.dumps(rec) + "\n")
    return len(train), len(val)


# ---------------------------------------------------------------------------
# Per-source post-extraction stats (length-filter only — extractor handles
# all empty/missing-field skips before we ever get here).
# ---------------------------------------------------------------------------
@dataclass
class PrepStats:
    source: str
    n_kept: int = 0
    n_dropped_length: int = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SOURCE_NAME = "open-r1/codeforces-cots:solutions_py_decontaminated"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 fine-tune data prep.")
    parser.add_argument("--out", type=Path, default=Path("finetune_data_v2"))
    parser.add_argument(
        "--tokenizer", type=Path,
        default=Path("pretrain_data_v2/bpe_tokenizer.json"),
        help="Path to the BPE tokenizer trained during pretrain data prep.",
    )
    parser.add_argument(
        "--max-len", type=int, default=2048,
        help="Max sequence length (must match the model's max_len).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.out.resolve()}\n")

    # Tokenizer must already exist — we reuse the BPE from pretraining so
    # vocab + special-token ids match the pretrained model exactly.
    if not args.tokenizer.exists():
        raise SystemExit(
            f"Tokenizer not found at {args.tokenizer}. "
            "Run data_prep.py first to produce pretrain_data_v2/bpe_tokenizer.json."
        )
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    print(f"Tokenizer: vocab={tokenizer.get_vocab_size():,}, "
          f"path={args.tokenizer}\n")

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    # ── Load source ─────────────────────────────────────────────────────────
    cf_ds = try_load_open_r1_codeforces_cots(hf_token)
    if cf_ds is None:
        raise SystemExit(
            "Could not load open-r1/codeforces-cots. Check HF auth and that "
            "you've accepted the dataset terms at "
            "https://huggingface.co/datasets/open-r1/codeforces-cots"
        )

    # ── Extract + tokenize ──────────────────────────────────────────────────
    # extract_stats is OWNED HERE and passed to the extractor. The summary
    # remains valid even if the loop below raises or stops early.
    extract_stats = ExtractStats()
    prep_stats = PrepStats(source=SOURCE_NAME)
    all_records: list[dict] = []

    print(f"\n[tokenize] {SOURCE_NAME}")
    triples = extract_open_r1_codeforces_cots(cf_ds, extract_stats)
    try:
        for problem, reasoning, code in tqdm(triples, desc=SOURCE_NAME):
            rec = tokenize_example(tokenizer, problem, reasoning, code, args.max_len)
            if rec is None:
                prep_stats.n_dropped_length += 1
                continue
            rec["source"] = SOURCE_NAME
            all_records.append(rec)
            prep_stats.n_kept += 1
    finally:
        # Always print the extractor summary — even if tokenize_example
        # raised or the user ^C'd mid-loop.
        print_extract_stats(SOURCE_NAME, extract_stats)
        print(f"  post-tokenize (length filter @ max_len={args.max_len}):")
        print(f"      kept:                                 {prep_stats.n_kept:,}")
        print(f"      dropped (too long):                   {prep_stats.n_dropped_length:,}")

    if not all_records:
        raise SystemExit(
            "No usable examples after filtering. Inspect the skip counts "
            "above to diagnose."
        )

    # Length histogram for visibility.
    lens = sorted(len(r["input_ids"]) for r in all_records)
    print(f"\nLength stats over {len(lens):,} kept examples:")
    print(f"  min={lens[0]}  p50={lens[len(lens)//2]}  "
          f"p95={lens[int(len(lens)*0.95)]}  max={lens[-1]}")

    # ── Split + write ───────────────────────────────────────────────────────
    train_path = args.out / "train.jsonl"
    val_path   = args.out / "val.jsonl"
    n_train, n_val = write_split(
        all_records, train_path, val_path, args.val_ratio, args.seed,
    )
    print(f"\nWrote {n_train:,} train / {n_val:,} val examples")
    print(f"  {train_path}")
    print(f"  {val_path}")

    # ── meta.json ───────────────────────────────────────────────────────────
    meta = {
        "tokenizer_path": str(args.tokenizer),
        "vocab_size": tokenizer.get_vocab_size(),
        "max_len": args.max_len,
        "pad_idx": PAD_IDX, "bos_idx": BOS_IDX,
        "eos_idx": EOS_IDX, "unk_idx": UNK_IDX,
        "tags": {
            "problem_open": PROBLEM_OPEN, "problem_close": PROBLEM_CLOSE,
            "reason_open":  REASON_OPEN,  "reason_close":  REASON_CLOSE,
            "code_open":    CODE_OPEN,    "code_close":    CODE_CLOSE,
        },
        "n_train": n_train,
        "n_val": n_val,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "sources": [
            {
                "name": SOURCE_NAME,
                "extract_stats": asdict(extract_stats),
                "n_kept": prep_stats.n_kept,
                "n_dropped_length": prep_stats.n_dropped_length,
            },
        ],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  {args.out / 'meta.json'}")

    # Sanity preview: decode the first train example and locate the
    # prompt → response boundary (where loss_mask flips 0 → 1).
    print("\nSanity preview of first train example:")
    with train_path.open("r") as f:
        ex = json.loads(f.readline())
    boundary = ex["loss_mask"].index(1) if 1 in ex["loss_mask"] else -1
    full = tokenizer.decode(ex["input_ids"])
    print(f"  length: {len(ex['input_ids'])} tokens, "
          f"prompt: {boundary} tokens, response: {len(ex['input_ids']) - boundary} tokens")
    print("  --- first 400 chars ---")
    print("  " + full[:400].replace("\n", "\n  "))
    print("  ...")

    print("\nDone. Next step:")
    print(f"    python finetune.py --data {args.out} \\")
    print(f"                       --pretrain checkpoints_pretrain_v2/best.pt")


if __name__ == "__main__":
    main()
