"""
saLLMan Phase 3a — Data preparation for pretraining.

Produces the four files in <out_dir>/ that pretrain.py expects:
  - bpe_tokenizer.json   byte-level BPE, vocab=16000, specials at ids 0..3
  - train.bin            uint16 tokens (concatenated docs, <eos> between docs)
  - val.bin              same, ~0.5 % of documents
  - meta.json            vocab_size, token counts, special token ids, sources

Phase 3a covers ONE source: bigcode/the-stack-smol Python (10k files, ~24M tokens).
Phase 3b will extend with CodeForces problems + submissions in a separate script.

Lessons baked into this implementation (see Phase 3 handoff):
  L1 — datasets >= 4.0 dropped script-based loaders. We bypass entirely by
       downloading a single file with hf_hub_download and loading it as a
       generic JSON dataset, never invoking the repo's loader script.
  L4 — the-stack-smol's data_dir / data_files config keys are fragile. The
       robust pattern is hf_hub_download → load_dataset("json", data_files=…).
  L5 — schema verified live (2026-05-10): rows=10000, columns include
       'content' (the code), 'repository_name', 'path', 'lang', 'licenses', …
  L6 — the-stack-smol layout is data/{lang}/data.json (single JSON), NOT
       data/{lang}/*.parquet shards. The handoff's parquet glob was wrong.

References:
  - Sennrich et al. 2016, "Neural Machine Translation of Rare Words with
    Subword Units" (BPE) — https://arxiv.org/abs/1508.07909
  - Radford et al. 2019, GPT-2 (byte-level BPE) — https://cdn.openai.com/
    better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  - Karpathy nanoGPT (memory-mapped uint16 token arrays) —
    https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

Usage:
    python data_prep.py [--out pretrain_data] [--vocab-size 16000]
                        [--val-ratio 0.005] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Special tokens — order = ids 0..3.  pretrain.py depends on these IDs.
# ---------------------------------------------------------------------------
SPECIAL_TOKENS: list[str] = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

# Source identifier (also written into meta.json)
SOURCE_NAME: str = "bigcode/the-stack-smol:python"
HF_REPO: str = "bigcode/the-stack-smol"
HF_FILE: str = "data/python/data.json"


# ---------------------------------------------------------------------------
# 1. Load source data
# ---------------------------------------------------------------------------
def load_stack_smol_python() -> Dataset:
    """
    Download data/python/data.json from bigcode/the-stack-smol and load it
    as a HF Dataset.

    The hf_hub_download → load_dataset("json", ...) pattern bypasses both
    the deprecated script loader (L1) and the data_dir/data_files config
    indirection that broke in earlier iterations (L4).
    """
    print(f"[1/5] Downloading {HF_REPO}:{HF_FILE} ...")
    local_path = hf_hub_download(HF_REPO, filename=HF_FILE, repo_type="dataset")
    print(f"      cached at: {local_path}")
    print(f"      size: {os.path.getsize(local_path) / 1e6:.1f} MB")

    ds = load_dataset("json", data_files=local_path, split="train")

    # Defensive schema check (L5). If 'content' disappears we want a clear
    # error pointing at the actual columns, not 0 silent results downstream.
    if "content" not in ds.column_names:
        sample_keys = list(ds[0].keys()) if len(ds) > 0 else []
        raise RuntimeError(
            f"Expected 'content' column in {SOURCE_NAME}; got columns "
            f"{ds.column_names}. First-row keys: {sample_keys}. "
            "If the dataset schema changed, update HF_FILE or the column name."
        )

    print(f"      rows: {len(ds):,}")
    print(f"      columns: {ds.column_names}")
    return ds


# ---------------------------------------------------------------------------
# 2. Train byte-level BPE
# ---------------------------------------------------------------------------
def train_bpe(docs: list[str], vocab_size: int) -> Tokenizer:
    """
    Byte-level BPE following GPT-2 / LLaMA conventions.

    Why byte-level
    --------------
    Code contains arbitrary Unicode (string literals, comments) and unusual
    whitespace. Byte-level BPE has a closed alphabet of 256 bytes and never
    produces <unk>, which matters for code where any character is potentially
    semantically meaningful.

    Why add_prefix_space=False
    --------------------------
    GPT-2 uses add_prefix_space=True so 'Hello' and ' Hello' are merged into
    one token type. For code, leading whitespace is structurally meaningful
    (Python indentation!) so we keep prefixes literal.
    """
    print(
        f"[2/5] Training byte-level BPE (vocab_size={vocab_size}) on "
        f"{len(docs):,} docs ..."
    )

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,            # order is preserved → ids 0..3
        initial_alphabet=ByteLevel.alphabet(),    # seed all 256 byte chars
        show_progress=True,
    )

    def iter_docs() -> Iterator[str]:
        for d in docs:
            yield d

    tokenizer.train_from_iterator(iter_docs(), trainer=trainer, length=len(docs))

    # Verify special-token ids are exactly 0..3 (pretrain.py depends on this).
    for name, expected in zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS))):
        got = tokenizer.token_to_id(name)
        if got != expected:
            raise RuntimeError(
                f"Special token {name!r} got id {got}, expected {expected}. "
                "BpeTrainer special-token ordering may have changed."
            )
    print(f"      vocab size: {tokenizer.get_vocab_size():,}")
    print(
        f"      special token ids verified: "
        f"{[(t, tokenizer.token_to_id(t)) for t in SPECIAL_TOKENS]}"
    )
    return tokenizer


# ---------------------------------------------------------------------------
# 3. Tokenize and pack into a uint16 array
# ---------------------------------------------------------------------------
def tokenize_split(
    tokenizer: Tokenizer,
    docs: list[str],
    desc: str,
) -> np.ndarray:
    """
    Encode each document, append <eos>, and concatenate into a 1-D uint16
    array. Returns shape (n_total_tokens,).

    Document boundaries: <eos> after each doc; no <bos> between docs (BOS is
    only used at the start of a generation prompt, per the handoff).

    uint16 is safe iff vocab_size <= 65535 — asserted below.
    """
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise ValueError(
            f"vocab_size={tokenizer.get_vocab_size()} exceeds uint16 range "
            f"({np.iinfo(np.uint16).max}); switch dtype to uint32 here AND "
            "in pretrain.py's np.memmap call."
        )

    # encode_batch is the fast Rust path. We chunk it to bound peak memory
    # and to get a meaningful progress bar.
    chunk_size = 1024
    pieces: list[np.ndarray] = []
    n_tokens = 0

    for i in tqdm(range(0, len(docs), chunk_size), desc=desc):
        batch = docs[i : i + chunk_size]
        encs = tokenizer.encode_batch(batch)
        for enc in encs:
            ids = enc.ids
            ids.append(EOS_IDX)                                 # doc separator
            arr = np.asarray(ids, dtype=np.uint16)
            pieces.append(arr)
            n_tokens += len(arr)

    out = np.concatenate(pieces, dtype=np.uint16)
    assert out.shape == (n_tokens,)
    return out


# ---------------------------------------------------------------------------
# 4. Output writers
# ---------------------------------------------------------------------------
def write_bin(arr: np.ndarray, path: Path) -> None:
    """Write a 1-D uint16 array as raw bytes (memory-mappable, no header)."""
    assert arr.dtype == np.uint16, f"expected uint16, got {arr.dtype}"
    arr.tofile(path)


def write_meta(
    out_dir: Path,
    vocab_size: int,
    n_train: int,
    n_val: int,
) -> None:
    """Schema must match what pretrain.py reads — see handoff doc."""
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
# 5. Verify outputs (handoff requirement #3)
# ---------------------------------------------------------------------------
def verify_outputs(out_dir: Path, tokenizer: Tokenizer, n_preview: int = 100) -> None:
    """Memory-map train.bin and decode the first N tokens for a sanity check."""
    arr = np.memmap(out_dir / "train.bin", dtype=np.uint16, mode="r")
    print(f"\n[5/5] Verifying train.bin: {len(arr):,} tokens (memory-mapped)")
    head = arr[:n_preview].tolist()
    print(f"      first {n_preview} ids head/tail: {head[:10]} ... {head[-5:]}")
    print(f"      decoded preview:")
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
        description="Phase 3a data prep: the-stack-smol Python → train.bin/val.bin."
    )
    parser.add_argument("--out", type=Path, default=Path("pretrain_data"))
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument(
        "--val-ratio", type=float, default=0.005,
        help="Fraction of documents held out for validation (default 0.5%%)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out.resolve()}\n")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    ds = load_stack_smol_python()
    docs: list[str] = list(ds["content"])

    # ── 2. Document-level train/val split ─────────────────────────────────────
    # Splitting at document granularity (not token granularity) avoids leaking
    # the start of a held-out doc into the train tail, which would make val
    # loss artificially low.
    rng = random.Random(args.seed)
    indices = list(range(len(docs)))
    rng.shuffle(indices)
    n_val = max(1, int(len(docs) * args.val_ratio))
    val_idx = set(indices[:n_val])
    train_docs = [docs[i] for i in range(len(docs)) if i not in val_idx]
    val_docs   = [docs[i] for i in range(len(docs)) if i in val_idx]
    print(
        f"      split: {len(train_docs):,} train docs / "
        f"{len(val_docs):,} val docs  (seed={args.seed})"
    )

    # ── 3. Train BPE on TRAIN docs only ──────────────────────────────────────
    # Training the tokenizer on val docs would technically leak val statistics
    # into the model's vocabulary. The cost of excluding 50 docs is negligible.
    tokenizer = train_bpe(train_docs, vocab_size=args.vocab_size)
    tokenizer.save(str(args.out / "bpe_tokenizer.json"))
    print(f"      saved tokenizer to {args.out / 'bpe_tokenizer.json'}")

    # ── 4. Tokenize and write bins ────────────────────────────────────────────
    print("[3/5] Tokenizing splits ...")
    train_arr = tokenize_split(tokenizer, train_docs, desc="train")
    val_arr   = tokenize_split(tokenizer, val_docs,   desc="val  ")
    print(f"      train tokens: {len(train_arr):,}")
    print(f"      val tokens:   {len(val_arr):,}")
    print(
        f"      train/val ratio: {len(train_arr)/(len(val_arr) or 1):.1f}x  "
        f"(target: ~{1/args.val_ratio:.0f}x)"
    )

    print("[4/5] Writing bins + meta.json ...")
    write_bin(train_arr, args.out / "train.bin")
    write_bin(val_arr,   args.out / "val.bin")
    write_meta(
        args.out,
        vocab_size=tokenizer.get_vocab_size(),
        n_train=len(train_arr),
        n_val=len(val_arr),
    )
    print(f"      train.bin: {(args.out / 'train.bin').stat().st_size / 1e6:.1f} MB")
    print(f"      val.bin:   {(args.out / 'val.bin').stat().st_size / 1e6:.1f} MB")
    print(f"      meta.json written")

    # ── 5. Verify ─────────────────────────────────────────────────────────────
    verify_outputs(args.out, tokenizer)

    print("\nDone. Next step:")
    print("    python pretrain.py")


if __name__ == "__main__":
    main()