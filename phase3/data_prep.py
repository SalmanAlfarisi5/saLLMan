"""
saLLMan Phase 3 — Data preparation for code+DSA pretraining.

This script does two things and saves the results to disk:

  1. Trains a fresh byte-level BPE tokenizer (16k vocab) on a sample of the
     pretraining corpus. The WikiText BPE from Phase 1/2 is unsuitable for
     code — it spends vocab on natural-language whitespace patterns and
     fragments common Python tokens (`def`, `self`, `range`) into pieces.

  2. Tokenizes the entire pretraining corpus and writes it to disk as a
     single binary uint16 array per split (train/val). This is the
     "nanoGPT pattern": one giant memory-mapped token blob, then chop into
     blocks at training time. Vastly faster than re-tokenizing every epoch.

Datasets used (all permissively licensed, all on HuggingFace Hub)
-----------------------------------------------------------------
  - bigcode/the-stack-smol (data/python)        ~10k Python files
  - code_search_net (python config)             ~450k function+docstring pairs
  - codeparrot/apps                              10k algorithmic problems

Why this mix:
  - the-stack-smol gives general high-quality Python code (the bulk).
  - CodeSearchNet pairs natural-language docstrings with code — the model
    learns to associate problem descriptions with implementations.
  - APPS gives explicit "natural-language problem → Python solution" pairs,
    which is exactly the saLLMan target distribution.

We deliberately do NOT use:
  - The full Stack — overkill for a 75M model on a 3060 Ti, and the streaming
    loader has known issues (HF datasets #7467).
  - LeetCode/USACO/Codeforces here — those are PHASE 3 fine-tuning data, run
    separately after pretraining.

Output files (in ./pretrain_data/)
----------------------------------
  bpe_tokenizer.json     trained tokenizer
  train.bin              uint16 array of training tokens (memory-mappable)
  val.bin                uint16 array of validation tokens
  meta.json              {vocab_size, n_train_tokens, n_val_tokens, ...}

Dependencies:
    pip install datasets tokenizers tqdm numpy
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tqdm import tqdm


OUT_DIR = Path("pretrain_data")
OUT_DIR.mkdir(exist_ok=True)

# Special tokens — same ids as Phase 1/2 for consistency across the project.
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
VOCAB_SIZE = 16000        # 2x the Phase 1/2 vocab — code has more "words"
VAL_FRACTION = 0.005      # 0.5% — enough for a stable PPL estimate, leave the rest for training
SEED = 42


# ===========================================================================
# 1. Source iterators — yield raw text strings from each dataset.
# ===========================================================================
# Each iterator yields document-level strings. We'll separate documents with
# <eos> when we tokenize and concatenate. A "document" here is one Python file,
# one (docstring, code) pair, or one (problem, solution) pair.

def iter_the_stack_smol() -> Iterator[str]:
    """Python files from bigcode/the-stack-smol. ~10k files."""
    print("Loading the-stack-smol (Python)...")
    ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")
    for ex in ds:
        # Light cleanup: skip extremely short or extremely long files.
        # Extremely short = probably auto-generated stubs.
        # Extremely long = often minified, generated, or test fixtures.
        content = ex["content"]
        if 100 <= len(content) <= 50_000:
            yield content


def iter_code_search_net() -> Iterator[str]:
    """
    Python (docstring, code) pairs from CodeSearchNet. We format each as:
        \"\"\"<docstring>\"\"\"
        <code>
    so the model sees natural-language → code pattern in its training data.
    """
    print("Loading code_search_net (Python)...")
    # CodeSearchNet has train/valid/test splits — concatenate train + valid
    # since we make our own val split below.
    ds = load_dataset("code_search_net", "python", split="train",
                      trust_remote_code=True)
    for ex in ds:
        doc = (ex.get("func_documentation_string") or "").strip()
        code = (ex.get("func_code_string") or "").strip()
        if not code:
            continue
        if doc:
            text = f'"""\n{doc}\n"""\n{code}\n'
        else:
            text = code + "\n"
        if 50 <= len(text) <= 10_000:
            yield text


def iter_apps() -> Iterator[str]:
    """
    APPS problems formatted as:
        # Problem
        <question>
        # Solution
        <solution>
    For problems with multiple solutions, we pick one at random per problem
    so we don't oversample any single problem.
    """
    print("Loading codeparrot/apps...")
    ds = load_dataset("codeparrot/apps", split="train", trust_remote_code=True)
    rng = random.Random(SEED)
    for ex in ds:
        question = ex["question"].strip()
        solutions_raw = ex.get("solutions") or ""
        if not solutions_raw:
            continue
        try:
            solutions = json.loads(solutions_raw)
        except json.JSONDecodeError:
            continue
        if not solutions:
            continue
        sol = rng.choice(solutions)
        text = f"# Problem\n{question}\n\n# Solution\n{sol}\n"
        yield text


def iter_all_documents() -> Iterator[str]:
    """Concatenates all sources. Order doesn't matter — we shuffle blocks later."""
    yield from iter_the_stack_smol()
    yield from iter_code_search_net()
    yield from iter_apps()


# ===========================================================================
# 2. Train BPE tokenizer
# ===========================================================================
def train_tokenizer(sample_size: int = 200_000) -> Tokenizer:
    """
    Train byte-level BPE on a sample of documents. Sampling rather than using
    every document because tokenizer training memory scales with corpus size
    and the marginal benefit beyond ~200k documents is negligible.
    """
    tok_path = OUT_DIR / "bpe_tokenizer.json"
    if tok_path.exists():
        print(f"Loading cached tokenizer from {tok_path}")
        return Tokenizer.from_file(str(tok_path))

    print(f"Sampling up to {sample_size:,} documents for tokenizer training...")
    samples: list[str] = []
    for i, doc in enumerate(iter_all_documents()):
        if i >= sample_size:
            break
        samples.append(doc)
    print(f"Got {len(samples):,} sample documents")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    # add_prefix_space=False: code lines often start without a leading space,
    # and we don't want to artificially insert one.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,    # order = id 0,1,2,3
        # Pre-add common code tokens that BPE might miss as standalone units.
        # These hint to BPE: "consider keeping these whole".
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"Training BPE (target vocab={VOCAB_SIZE})...")
    tokenizer.train_from_iterator(samples, trainer=trainer, length=len(samples))
    tokenizer.save(str(tok_path))
    print(f"Saved tokenizer to {tok_path}")
    return tokenizer


# ===========================================================================
# 3. Tokenize and pack to disk
# ===========================================================================
def tokenize_and_pack(tokenizer: Tokenizer) -> tuple[int, int]:
    """
    Tokenize every document, separated by <eos>, and write the resulting
    token stream to disk as uint16 binary files (train.bin, val.bin).

    We use uint16 because vocab=16000 < 65535 → fits in 2 bytes per token,
    half the disk and memory of int32. nanoGPT uses this trick.

    Splitting into train/val: we sample a small random fraction of DOCUMENTS
    (not tokens) to be validation. Doing it at document granularity prevents
    leakage of context across the split boundary.
    """
    train_path = OUT_DIR / "train.bin"
    val_path   = OUT_DIR / "val.bin"
    if train_path.exists() and val_path.exists():
        n_train = train_path.stat().st_size // 2
        n_val   = val_path.stat().st_size // 2
        print(f"Found existing token files: train={n_train:,}, val={n_val:,}")
        return n_train, n_val

    rng = random.Random(SEED)
    # We accumulate tokens in chunks then write to disk in append mode to
    # keep peak memory bounded. Each chunk is up to ~10M tokens (~20MB each).
    CHUNK_TOKENS = 10_000_000

    train_buf: list[int] = []
    val_buf:   list[int] = []
    n_train_total = 0
    n_val_total = 0

    # Open output files in append-binary mode. We'll write chunks as they fill.
    train_f = open(train_path, "wb")
    val_f   = open(val_path,   "wb")

    def flush(buf: list[int], f) -> int:
        if not buf:
            return 0
        arr = np.asarray(buf, dtype=np.uint16)
        f.write(arr.tobytes())
        f.flush()
        return arr.size

    try:
        for doc in tqdm(iter_all_documents(), desc="Tokenizing"):
            ids = tokenizer.encode(doc).ids
            ids.append(EOS_IDX)   # document boundary marker

            # Route this document's tokens to train or val.
            if rng.random() < VAL_FRACTION:
                val_buf.extend(ids)
                if len(val_buf) >= CHUNK_TOKENS:
                    n_val_total += flush(val_buf, val_f)
                    val_buf.clear()
            else:
                train_buf.extend(ids)
                if len(train_buf) >= CHUNK_TOKENS:
                    n_train_total += flush(train_buf, train_f)
                    train_buf.clear()

        # Final flush of remaining buffers.
        n_train_total += flush(train_buf, train_f)
        n_val_total   += flush(val_buf,   val_f)
    finally:
        train_f.close()
        val_f.close()

    print(f"Wrote train.bin: {n_train_total:,} tokens ({n_train_total * 2 / 1e6:.1f} MB)")
    print(f"Wrote val.bin:   {n_val_total:,} tokens ({n_val_total * 2 / 1e6:.1f} MB)")
    return n_train_total, n_val_total


# ===========================================================================
# 4. Main
# ===========================================================================
def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    tokenizer = train_tokenizer()
    actual_vocab = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {actual_vocab}")

    n_train, n_val = tokenize_and_pack(tokenizer)

    meta = {
        "vocab_size":     actual_vocab,
        "n_train_tokens": n_train,
        "n_val_tokens":   n_val,
        "pad_idx": PAD_IDX, "bos_idx": BOS_IDX, "eos_idx": EOS_IDX, "unk_idx": UNK_IDX,
        "datasets": [
            "bigcode/the-stack-smol (python)",
            "code_search_net (python)",
            "codeparrot/apps (train)",
        ],
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {OUT_DIR / 'meta.json'}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
