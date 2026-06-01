"""
saLLMan Phase 3 (fine-tune) — data prep for supervised fine-tuning.

Builds an instruction-tuning corpus from problem-solving datasets that
already contain chain-of-thought reasoning traces:

  - open-r1/codeforces           Codeforces problems with DeepSeek-R1
                                 reasoning traces.
  - greengerong/leetcode         LeetCode problems with solutions.
                                 Used only if a reasoning/explanation
                                 field is present in the schema —
                                 otherwise skipped (per the "only datasets
                                 with reasoning" preference).

The output format (per the chosen tagged-sections convention):
    <problem>
    {problem text}
    </problem>
    <reasoning>
    {chain-of-thought}
    </reasoning>
    <code>
    {final solution code}
    </code><eos>

Why tagged sections rather than ChatML / new special tokens:
  - Plain text means the pretrained tokenizer doesn't need extending.
  - The model has never seen these tags, but it has seen lots of
    XML-style markup during pretraining (the Stack contains HTML, JSX,
    docstrings) — it will pick up on them as boundary signals quickly.
  - We control where the response starts (right after the first
    "<reasoning>\n") which lets us mask loss on the prompt cleanly.

Loss masking
------------
We tokenize prompt and response *separately* so we know the exact
boundary in id space:
    input_ids  = prompt_ids + response_ids
    loss_mask  = [0] * len(prompt_ids) + [1] * len(response_ids)
At training time, positions with loss_mask=0 are replaced with -100 in
labels so `F.cross_entropy(ignore_index=-100)` skips them.

Output format
-------------
JSONL files, one example per line:
    {"input_ids": [int, ...], "loss_mask": [0/1, ...]}
Plus a `meta.json` with example counts, source breakdown, and the
tokenizer path used.

Datasets are small (~10s of MB after tokenization) so we DON'T use
memmap here. A plain JSON read in finetune.py is fine.

Usage
-----
    cd phase3
    python finetune_data_prep.py
    python finetune_data_prep.py --out finetune_data_v2 \\
                                 --tokenizer pretrain_data_v2/bpe_tokenizer.json \\
                                 --max-len 512
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from datasets import load_dataset, Dataset, DatasetDict
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
# Per-source extractors
# ---------------------------------------------------------------------------
# Each extractor returns an iterator of (problem, reasoning, code) triples.
# If a source's schema doesn't expose reasoning, we either skip the source
# (require_reasoning=True) or yield reasoning="" (False).

def extract_open_r1_codeforces(ds: Dataset) -> Iterator[tuple[str, str, str]]:
    """
    open-r1/codeforces schema (as of 2026): each row contains a Codeforces
    problem statement, plus DeepSeek-R1 generated reasoning traces and a
    final solution. Column names have varied across releases of this
    dataset; we probe several common ones and assemble robustly.

    Expected fields (any of these may appear):
      problem text:  one of [`problem`, `problem_statement`, `description`]
      reasoning:     one of [`reasoning`, `generation`, `thought`, `chain_of_thought`]
      solution code: one of [`solution`, `code`, `submission`, `answer`]
      messages:      sometimes the whole thing is a list of role/content dicts
    """
    cols = set(ds.column_names)

    # Path A: column-style schema
    p_key = next((k for k in ("problem", "problem_statement", "description") if k in cols), None)
    r_key = next((k for k in ("reasoning", "generation", "thought", "chain_of_thought") if k in cols), None)
    c_key = next((k for k in ("solution", "code", "submission", "answer") if k in cols), None)

    if p_key and (r_key or "messages" in cols) and c_key:
        for row in ds:
            problem = (row.get(p_key) or "").strip()
            code = (row.get(c_key) or "").strip()
            if r_key:
                reasoning = (row.get(r_key) or "").strip()
            else:
                # Messages-style: concatenate assistant turns as reasoning.
                msgs = row.get("messages") or []
                reasoning = "\n".join(m.get("content", "").strip()
                                      for m in msgs if m.get("role") == "assistant").strip()
            if problem and reasoning and code:
                yield problem, reasoning, code
        return

    # Path B: messages-only schema (single conversation per row)
    if "messages" in cols:
        for row in ds:
            msgs = row.get("messages") or []
            user_parts = [m["content"] for m in msgs if m.get("role") == "user"]
            asst_parts = [m["content"] for m in msgs if m.get("role") == "assistant"]
            if not user_parts or not asst_parts:
                continue
            problem = "\n\n".join(user_parts).strip()
            # Heuristic: last assistant turn = code; everything before = reasoning.
            reasoning = "\n\n".join(asst_parts[:-1]).strip() or asst_parts[-1].strip()
            code = asst_parts[-1].strip()
            if problem and reasoning and code:
                yield problem, reasoning, code
        return

    raise RuntimeError(
        f"open-r1/codeforces: unknown schema. Columns: {sorted(cols)}. "
        "Inspect the dataset on HF and update extract_open_r1_codeforces."
    )


def extract_greengerong_leetcode(
    ds: Dataset,
    require_reasoning: bool,
) -> Iterator[tuple[str, str, str]]:
    """
    greengerong/leetcode schema:
      slug, title, content, difficulty, solution, hints, ...
    `content` is the problem statement, `solution` typically contains a
    Python implementation. There is sometimes no explicit reasoning field,
    in which case (with require_reasoning=True) we skip the source.

    Heuristic: if `solution` contains both prose and code blocks, we treat
    everything before the first code fence as reasoning. This is what most
    LeetCode editorial-style entries look like.
    """
    cols = set(ds.column_names)
    p_key = next((k for k in ("content", "description", "problem") if k in cols), None)
    s_key = next((k for k in ("solution", "python", "code") if k in cols), None)

    if p_key is None or s_key is None:
        print(f"  ! greengerong/leetcode: no usable columns "
              f"(have: {sorted(cols)}); skipping source.")
        return

    n_with_reason = 0
    n_without_reason = 0

    for row in ds:
        problem = (row.get(p_key) or "").strip()
        full_sol = (row.get(s_key) or "").strip()
        if not problem or not full_sol:
            continue

        # Try to split prose-reasoning from code by a fenced code block.
        reasoning, code = _split_prose_and_code(full_sol)
        if reasoning:
            n_with_reason += 1
            yield problem, reasoning, code
        else:
            n_without_reason += 1
            if require_reasoning:
                continue
            yield problem, "", full_sol

    print(f"  greengerong/leetcode: kept {n_with_reason} with reasoning, "
          f"{'skipped' if require_reasoning else 'kept-as-empty-reasoning'} "
          f"{n_without_reason}")


def _split_prose_and_code(text: str) -> tuple[str, str]:
    """
    Best-effort split of an editorial-style answer into (prose, code).
    If a ```python ... ``` (or generic ```...```) fence is present, prose
    is everything before the first fence and code is the first fenced
    block. If no fence is found, returns ("", text) — caller decides.
    """
    fence_marker_long = "```python"
    if fence_marker_long in text:
        before, _, after = text.partition(fence_marker_long)
        code_block, _, _rest = after.partition("```")
        return before.strip(), code_block.strip()

    if "```" in text:
        before, _, after = text.partition("```")
        # Drop any leading language hint like "python\n"
        first_line, _, rest = after.partition("\n")
        if first_line.strip().isalpha():
            after = rest
        code_block, _, _rest = after.partition("```")
        return before.strip(), code_block.strip()

    return "", text.strip()


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

    We tokenize prompt and response separately so we can build loss_mask
    bit-exactly without searching for a delimiter token afterward.

    Layout:
        input_ids  = [BOS] + prompt_ids + response_ids + [EOS]
        loss_mask  = [0]   + [0]*L_p    + [1]*L_r      + [1]
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
# Source loaders — separated so each dataset's quirks live in one place
# ---------------------------------------------------------------------------
def try_load_open_r1_codeforces(hf_token: str | None) -> Dataset | None:
    """
    Returns a `Dataset` (train split) or None if loading fails. We don't
    abort the whole pipeline on a single source failure — open-r1
    occasionally renames or re-shards the dataset.
    """
    print("[load] open-r1/codeforces ...")
    try:
        d = load_dataset("open-r1/codeforces", split="train", token=hf_token)
    except Exception as e:
        print(f"  ! failed to load open-r1/codeforces: {e}")
        return None
    print(f"      rows: {len(d):,}, columns: {d.column_names}")
    return d


def try_load_greengerong_leetcode(hf_token: str | None) -> Dataset | None:
    print("[load] greengerong/leetcode ...")
    try:
        d = load_dataset("greengerong/leetcode", split="train", token=hf_token)
    except Exception as e:
        print(f"  ! failed to load greengerong/leetcode: {e}")
        return None
    print(f"      rows: {len(d):,}, columns: {d.column_names}")
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@dataclass
class PrepStats:
    source: str
    n_loaded: int
    n_kept: int
    n_dropped_length: int
    n_dropped_empty: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 fine-tune data prep.")
    parser.add_argument("--out", type=Path, default=Path("finetune_data_v2"))
    parser.add_argument(
        "--tokenizer", type=Path,
        default=Path("pretrain_data_v2/bpe_tokenizer.json"),
        help="Path to the BPE tokenizer trained during pretrain data prep.",
    )
    parser.add_argument(
        "--max-len", type=int, default=512,
        help="Max sequence length (must match the model's max_len).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-reasoning", action="store_true", default=True,
        help="Skip examples without a non-empty reasoning section "
             "(default; reflects the 'only datasets with reasoning' choice).",
    )
    parser.add_argument("--no-require-reasoning", dest="require_reasoning",
                        action="store_false")
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

    sources: list[tuple[str, "Iterator[tuple[str, str, str]] | None"]] = []
    cf_ds = try_load_open_r1_codeforces(hf_token)
    if cf_ds is not None:
        sources.append(("open-r1/codeforces", extract_open_r1_codeforces(cf_ds)))
    lc_ds = try_load_greengerong_leetcode(hf_token)
    if lc_ds is not None:
        sources.append((
            "greengerong/leetcode",
            extract_greengerong_leetcode(lc_ds, require_reasoning=args.require_reasoning),
        ))

    if not sources:
        raise SystemExit("No sources loaded — check HF auth and dataset access.")

    # Collect all tokenized examples first, then split. The full corpus is
    # small (~10s of MB after tokenization) so this fits comfortably in RAM.
    all_records: list[dict] = []
    stats: list[PrepStats] = []

    for source_name, triples_iter in sources:
        print(f"\n[tokenize] {source_name}")
        n_loaded = n_kept = n_drop_len = n_drop_empty = 0
        for problem, reasoning, code in tqdm(triples_iter, desc=source_name):
            n_loaded += 1
            if not problem or not code:
                n_drop_empty += 1
                continue
            if args.require_reasoning and not reasoning:
                n_drop_empty += 1
                continue
            rec = tokenize_example(tokenizer, problem, reasoning, code, args.max_len)
            if rec is None:
                n_drop_len += 1
                continue
            rec["source"] = source_name
            all_records.append(rec)
            n_kept += 1
        stats.append(PrepStats(source_name, n_loaded, n_kept, n_drop_len, n_drop_empty))
        print(f"      kept: {n_kept:,} / {n_loaded:,} "
              f"(too long: {n_drop_len:,}, empty/no-reason: {n_drop_empty:,})")

    if not all_records:
        raise SystemExit(
            "No usable examples after filtering. Try --no-require-reasoning, "
            "increase --max-len, or check that the source schemas match."
        )

    # Length histogram for visibility — handy when tuning max_len later.
    lens = [len(r["input_ids"]) for r in all_records]
    print(f"\nLength stats over {len(all_records):,} examples:")
    print(f"  min={min(lens)}  p50={sorted(lens)[len(lens)//2]}  "
          f"p95={sorted(lens)[int(len(lens)*0.95)]}  max={max(lens)}")

    # Split + write.
    train_path = args.out / "train.jsonl"
    val_path = args.out / "val.jsonl"
    n_train, n_val = write_split(all_records, train_path, val_path,
                                  args.val_ratio, args.seed)
    print(f"\nWrote {n_train:,} train / {n_val:,} val examples")
    print(f"  {train_path}")
    print(f"  {val_path}")

    # meta.json
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
        "sources": [asdict(s) for s in stats],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  {args.out / 'meta.json'}")

    # Sanity preview: detokenize the first train example, showing where
    # loss_mask flips from 0 to 1 (the prompt → response boundary).
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
    print(f"    python finetune.py --data {args.out} --pretrain checkpoints_pretrain_v2/best.pt")


if __name__ == "__main__":
    main()
