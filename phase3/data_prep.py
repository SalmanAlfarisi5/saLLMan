"""
saLLMan Phase 3a (v2) — Pretraining data prep using full The Stack Python.

What changed from the smol baseline
-----------------------------------
The previous version of this file pulled bigcode/the-stack-smol Python (10k
files, ~24M tokens). At the 46.7M-param target this gave a token/parameter
ratio of ~0.5, ~40x below Chinchilla's ~20 tokens/param lower bound
(Hoffmann et al. 2022, https://arxiv.org/abs/2203.15556), and training ran
for ~28 epochs of the same corpus. Result: train PPL=1.34 vs val PPL=68 at
step 20k — textbook overfitting.

This v2 fixes the data side:
  - Source: bigcode/the-stack-dedup (data/python/), streamed.
  - Stop condition: configurable --target-tokens budget (default 2B,
    Chinchilla-class for the ~97M model in pretrain.py).
  - On-disk JSONL staging — corpus never has to fit in RAM.
  - BPE trained on a bounded sample of documents so the tokenizers trainer
    doesn't OOM on a multi-GB corpus.

The smol pipeline is preserved in git history; this file replaces it.

Outputs (in <out_dir>/):
  - bpe_tokenizer.json       byte-level BPE, vocab=16000, specials at ids 0..3
  - train.bin                uint16 tokens, doc-separated by <eos>
  - val.bin                  uint16 tokens, ~0.5% of docs
  - meta.json                vocab_size, token counts, special ids, sources
  - docs.jsonl               staging file (kept by default for resumability;
                             pass --keep-staging=false to delete after run)

References:
  - Hoffmann et al. 2022, "Training Compute-Optimal LLMs" (Chinchilla)
    https://arxiv.org/abs/2203.15556
  - Kocetkov et al. 2022, "The Stack: 3 TB of permissively licensed code"
    https://arxiv.org/abs/2211.15533
  - Sennrich et al. 2016, BPE — https://arxiv.org/abs/1508.07909
  - Radford et al. 2019, GPT-2 byte-level BPE
  - Karpathy nanoGPT — memory-mapped uint16 token arrays

Auth
----
bigcode/the-stack-dedup is gated. You must:
  1. Accept the dataset terms at
     https://huggingface.co/datasets/bigcode/the-stack-dedup
  2. Have HF_TOKEN in .env (already required by the smol pipeline).

Usage
-----
    python data_prep.py                           # 2B-token default
    python data_prep.py --target-tokens 500_000_000   # smaller for a smoke test
    python data_prep.py --out pretrain_data_v2 --vocab-size 16000
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Special tokens — order = ids 0..3. pretrain.py depends on these IDs.
# ---------------------------------------------------------------------------
SPECIAL_TOKENS: list[str] = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

# Source identifier (also written into meta.json).
# the-stack-dedup is deduplicated near-exact (file-level MinHash); preferred
# over the-stack for training corpora since duplicates inflate epoch counts
# silently and hurt generalisation.
SOURCE_NAME: str = "bigcode/the-stack-dedup:python"
HF_REPO: str = "bigcode/the-stack-dedup"
HF_DATA_DIR: str = "data/python"

# Document filters (empirically chosen)
MIN_CHARS: int = 64           # drop near-empty stubs
MAX_CHARS: int = 100_000      # drop pathological mega-files (notebooks, generated)

# Rough chars/token ratio for Python BPE@16k. Used only to estimate when to
# stop streaming — we verify the true token count after tokenisation.
APPROX_CHARS_PER_TOKEN: float = 3.8

# Cap for BPE training corpus. BpeTrainer holds its input in memory.
BPE_TRAIN_MAX_DOCS: int = 200_000


# ---------------------------------------------------------------------------
# 1. Stream the source and stage to disk
# ---------------------------------------------------------------------------
def stage_corpus(
    staging_path: Path,
    target_tokens: int,
) -> tuple[int, int]:
    """
    Stream bigcode/the-stack-dedup data/python until we've accumulated roughly
    `target_tokens` worth of text (using a char/token estimate), writing each
    doc as a JSONL line {"content": "..."} to `staging_path`.

    Why streaming + JSONL staging
    -----------------------------
    Pulling the full Python subset to local parquet would cost hundreds of GB
    and most of it we wouldn't use. Streaming pages files on demand from HF
    and we stop as soon as the budget is met. JSONL on disk lets every later
    pass (BPE training, tokenisation, val split) iterate without re-streaming.

    Resumable
    ---------
    If `staging_path` already exists, we trust it and skip the download.
    Delete the file to force a re-pull.

    Returns (n_docs_written, total_chars_written).
    """
    if staging_path.exists():
        # Trust the cache. Count what's there so the next stages know the size.
        print(f"[1/5] Staging file already exists at {staging_path} — reusing.")
        n_docs, n_chars = 0, 0
        with staging_path.open("r", encoding="utf-8") as f:
            for line in f:
                n_docs += 1
                # Avoid full JSON parse just to count chars — we serialized
                # exactly one key, so len(content)+overhead is approximately
                # len(line). But for an exact count, parse.
                try:
                    n_chars += len(json.loads(line)["content"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"      cached: {n_docs:,} docs, {n_chars/1e9:.2f}B chars")
        return n_docs, n_chars

    target_chars = int(target_tokens * APPROX_CHARS_PER_TOKEN)
    print(f"[1/5] Streaming {HF_REPO}:{HF_DATA_DIR}")
    print(f"      target: {target_tokens/1e9:.2f}B tokens "
          f"(~{target_chars/1e9:.2f}B chars at {APPROX_CHARS_PER_TOKEN:.1f} char/tok)")

    load_dotenv()  # picks up HF_TOKEN from .env if present
    token = os.environ.get("HF_TOKEN")

    # streaming=True returns an IterableDataset — yields rows without
    # materialising the corpus locally.
    ds = load_dataset(
        HF_REPO,
        data_dir=HF_DATA_DIR,
        split="train",
        streaming=True,
        token=token,
    )

    n_docs = 0
    n_chars = 0
    n_skipped_short = 0
    n_skipped_long = 0

    # tqdm with a chars-based total so the bar reflects budget progress.
    pbar = tqdm(total=target_chars, unit="char", unit_scale=True, desc="staging")

    staging_path.parent.mkdir(parents=True, exist_ok=True)
    with staging_path.open("w", encoding="utf-8") as f:
        for row in ds:
            content = row.get("content")
            if content is None:
                continue
            L = len(content)
            if L < MIN_CHARS:
                n_skipped_short += 1
                continue
            if L > MAX_CHARS:
                n_skipped_long += 1
                continue

            # Write a minimal JSON object — content only — to keep the file
            # small and the parser simple downstream.
            f.write(json.dumps({"content": content}, ensure_ascii=False) + "\n")
            n_docs += 1
            n_chars += L
            pbar.update(L)

            if n_chars >= target_chars:
                break

    pbar.close()

    print(f"      wrote {n_docs:,} docs, {n_chars/1e9:.2f}B chars to {staging_path}")
    print(f"      skipped (too short <{MIN_CHARS}): {n_skipped_short:,}")
    print(f"      skipped (too long >{MAX_CHARS}): {n_skipped_long:,}")
    return n_docs, n_chars


# ---------------------------------------------------------------------------
# 2. Train byte-level BPE on a sampled subset
# ---------------------------------------------------------------------------
def _iter_jsonl_content(path: Path, limit: int | None = None) -> Iterator[str]:
    """Yield 'content' field from each JSONL line, optionally up to `limit`."""
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)["content"]
            except (json.JSONDecodeError, KeyError):
                continue
            n += 1
            if limit is not None and n >= limit:
                return


def train_bpe(staging_path: Path, vocab_size: int, n_total_docs: int) -> Tokenizer:
    """
    Train byte-level BPE on at most BPE_TRAIN_MAX_DOCS sampled from the
    staged corpus. With 2B-token corpora the BPE trainer would otherwise
    keep gigabytes of strings live; ~200k Python files (~600 MB of code)
    is plenty for a converged 16k vocabulary.

    Why byte-level + add_prefix_space=False: same as the smol pipeline —
    code has structural whitespace (indentation) we want to keep literal,
    and byte-level means no <unk> ever fires for arbitrary Unicode.
    """
    n_train = min(BPE_TRAIN_MAX_DOCS, n_total_docs)
    print(f"[2/5] Training byte-level BPE (vocab_size={vocab_size}) on "
          f"{n_train:,} sampled docs ...")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,         # order preserved → ids 0..3
        initial_alphabet=ByteLevel.alphabet(), # seed all 256 byte chars
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        _iter_jsonl_content(staging_path, limit=n_train),
        trainer=trainer,
        length=n_train,
    )

    # Verify special-token ids are 0..3 — pretrain.py and the EOS doc
    # separator below depend on this.
    for name, expected in zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS))):
        got = tokenizer.token_to_id(name)
        if got != expected:
            raise RuntimeError(
                f"Special token {name!r} got id {got}, expected {expected}. "
                "BpeTrainer special-token ordering may have changed."
            )
    print(f"      vocab size: {tokenizer.get_vocab_size():,}")
    print(f"      special token ids: "
          f"{[(t, tokenizer.token_to_id(t)) for t in SPECIAL_TOKENS]}")
    return tokenizer


# ---------------------------------------------------------------------------
# 3. Tokenise the staged corpus and write train/val bins incrementally
# ---------------------------------------------------------------------------
def tokenise_and_pack(
    tokenizer: Tokenizer,
    staging_path: Path,
    train_bin: Path,
    val_bin: Path,
    n_total_docs: int,
    val_ratio: float,
    seed: int,
    encode_chunk: int = 1024,
) -> tuple[int, int]:
    """
    Stream the staged corpus, encode in chunks of `encode_chunk` docs, and
    append to train.bin or val.bin as uint16 raw bytes. Per-doc split is
    decided up-front by hash-shuffling indices so it's deterministic given
    the seed.

    Why incremental writes
    ----------------------
    At 2B tokens, train.bin is ~4 GB. Building it in memory first wastes
    RAM and would race with the BPE training stage. Writing as we go means
    peak memory is bounded by one chunk's worth of encoded tokens (~10MB).

    Returns (n_train_tokens, n_val_tokens).
    """
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise ValueError(
            f"vocab_size={tokenizer.get_vocab_size()} exceeds uint16 range "
            f"({np.iinfo(np.uint16).max}); switch dtype to uint32 here AND "
            "in pretrain.py's np.memmap call."
        )

    # Decide train/val membership for every doc up-front.
    rng = random.Random(seed)
    indices = list(range(n_total_docs))
    rng.shuffle(indices)
    n_val = max(1, int(n_total_docs * val_ratio))
    val_set = set(indices[:n_val])

    print(f"[3/5] Tokenising + writing bins")
    print(f"      train docs: {n_total_docs - n_val:,}")
    print(f"      val docs:   {n_val:,}  (val_ratio={val_ratio})")

    train_bin.parent.mkdir(parents=True, exist_ok=True)
    n_train_tokens = 0
    n_val_tokens = 0

    # Open both bins in binary-append mode (truncate first to be safe).
    with train_bin.open("wb") as ft, val_bin.open("wb") as fv:
        batch: list[str] = []
        batch_idx: list[int] = []
        doc_idx = 0

        def flush(batch: list[str], batch_idx: list[int]) -> tuple[int, int]:
            """Encode the batch, append <eos>, route each doc to train/val."""
            nonlocal n_train_tokens, n_val_tokens
            if not batch:
                return 0, 0
            encs = tokenizer.encode_batch(batch)
            added_train = 0
            added_val = 0
            for enc, idx in zip(encs, batch_idx):
                ids = enc.ids
                ids.append(EOS_IDX)
                arr = np.asarray(ids, dtype=np.uint16)
                if idx in val_set:
                    arr.tofile(fv)
                    n_val_tokens += arr.size
                    added_val += arr.size
                else:
                    arr.tofile(ft)
                    n_train_tokens += arr.size
                    added_train += arr.size
            return added_train, added_val

        pbar = tqdm(total=n_total_docs, desc="tokenising")
        for content in _iter_jsonl_content(staging_path):
            batch.append(content)
            batch_idx.append(doc_idx)
            doc_idx += 1
            if len(batch) >= encode_chunk:
                flush(batch, batch_idx)
                batch.clear()
                batch_idx.clear()
                pbar.update(encode_chunk)
        # Final partial chunk
        if batch:
            flush(batch, batch_idx)
            pbar.update(len(batch))
        pbar.close()

    print(f"      train tokens: {n_train_tokens:,}")
    print(f"      val tokens:   {n_val_tokens:,}")
    print(f"      train.bin: {train_bin.stat().st_size/1e9:.2f} GB")
    print(f"      val.bin:   {val_bin.stat().st_size/1e6:.1f} MB")
    return n_train_tokens, n_val_tokens


# ---------------------------------------------------------------------------
# 4. meta.json — schema must match what pretrain.py reads
# ---------------------------------------------------------------------------
def write_meta(out_dir: Path, vocab_size: int, n_train: int, n_val: int) -> None:
    meta = {
        "vocab_size":     vocab_size,
        "n_train_tokens": n_train,
        "n_val_tokens":   n_val,
        "pad_idx":        PAD_IDX,
        "bos_idx":        BOS_IDX,
        "eos_idx":        EOS_IDX,
        "unk_idx":        UNK_IDX,
        "datasets":       [SOURCE_NAME],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# 5. Verify outputs
# ---------------------------------------------------------------------------
def verify_outputs(out_dir: Path, tokenizer: Tokenizer, n_preview: int = 100) -> None:
    arr = np.memmap(out_dir / "train.bin", dtype=np.uint16, mode="r")
    print(f"\n[5/5] Verifying train.bin: {len(arr):,} tokens (memory-mapped)")
    head = arr[:n_preview].tolist()
    print(f"      first {n_preview} ids head/tail: {head[:10]} ... {head[-5:]}")
    print("      decoded preview:")
    print("      " + "-" * 60)
    decoded = tokenizer.decode(head)
    for line in decoded.split("\n"):
        print(f"      | {line}")
    print("      " + "-" * 60)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3a (v2) data prep: The Stack Python → train.bin/val.bin."
    )
    parser.add_argument("--out", type=Path, default=Path("pretrain_data_v2"))
    parser.add_argument(
        "--target-tokens", type=int, default=2_000_000_000,
        help="Approximate training token budget (default 2B, Chinchilla-class "
             "for ~97M-param target).",
    )
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument(
        "--val-ratio", type=float, default=0.005,
        help="Fraction of documents held out for validation (default 0.5%%).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-staging", action="store_true", default=True,
        help="Keep docs.jsonl after run (default; allows re-tokenisation "
             "without re-streaming). Pass --no-keep-staging to delete.",
    )
    parser.add_argument("--no-keep-staging", dest="keep_staging",
                        action="store_false")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out.resolve()}\n")

    staging_path = args.out / "docs.jsonl"

    # 1. Stream + stage
    n_docs, _ = stage_corpus(staging_path, args.target_tokens)
    if n_docs == 0:
        raise RuntimeError(
            "No documents staged. Check HF_TOKEN access to "
            f"{HF_REPO} (accept terms on the dataset page)."
        )

    # 2. Train BPE
    tokenizer = train_bpe(staging_path, vocab_size=args.vocab_size,
                          n_total_docs=n_docs)
    tokenizer.save(str(args.out / "bpe_tokenizer.json"))
    print(f"      saved tokenizer to {args.out / 'bpe_tokenizer.json'}")

    # 3. Tokenise + pack
    n_train, n_val = tokenise_and_pack(
        tokenizer,
        staging_path=staging_path,
        train_bin=args.out / "train.bin",
        val_bin=args.out / "val.bin",
        n_total_docs=n_docs,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 4. meta.json
    print("[4/5] Writing meta.json ...")
    write_meta(
        args.out,
        vocab_size=tokenizer.get_vocab_size(),
        n_train=n_train,
        n_val=n_val,
    )
    print(f"      meta.json written")

    # 5. Verify
    verify_outputs(args.out, tokenizer)

    if not args.keep_staging:
        staging_path.unlink(missing_ok=True)
        print(f"\nDeleted staging file {staging_path}")

    print("\nDone. Next step:")
    print(f"    python pretrain.py            # uses {args.out}/ by default")


if __name__ == "__main__":
    main()
