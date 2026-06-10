"""
phase4/build_curriculum.py — score every GRPO-pool problem with the SFT
model to find which problems have non-zero reward variance (the learnable
subset for Phase 4 GRPO).

Why
---
The rollouts.py smoke test showed ~2/3 of randomly-sampled problems are
dead-signal (all G completions score 0) for the current 97M SFT model.
GRPO's group-relative advantage is exactly zero on a dead-signal problem,
so optimising against those wastes rollout compute. Before the training
loop, we score the full pool once to find:

  - DEAD problems (group_std == 0) — exclude from GRPO training pool.
  - SOLVABLE problems (group_std > 0) — the curriculum.
  - ALREADY-SOLVED-SOMETIMES problems (max_reward >= 0.99) — still
    contribute gradient if not every completion is perfect, but useful
    to flag.

Output: one JSONL line per scored problem, sorted by group_std descending
(most learnable first), in `phase4/curriculum.jsonl`. The GRPO training
loop will read this file and pick the top-K problems by group_std.

Cost control
------------
Test-execution cost scales with problems × group_size × tests_per_problem.
Some codeforces-cots private_tests have 50+ cases; at 5 s timeout each
that's 50·5·8·5612 = ~3.1 M seconds worst case. We cap reward evaluation
at MAX_TESTS = 15 randomly-sampled tests per problem (per-row seed so the
sample is reproducible). The mean reward over 15 tests is a noisy but
unbiased estimator of the mean over all tests — good enough for sorting.

Crash safety
------------
- The JSONL is written append-only every 100 problems (flush + fsync).
- Resume is automatic: existing row_indices are skipped on re-run.
- KeyboardInterrupt is caught and the partial JSONL is sorted before exit.

Run
---
    cd phase4
    python build_curriculum.py --limit 50        # quick smoke run
    python build_curriculum.py                   # full pool
    python build_curriculum.py --limit 500       # continue 500 more on top
    rm curriculum.jsonl && python build_curriculum.py    # full restart
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tokenizers import Tokenizer
from tqdm import tqdm


# sys.path setup: project root + phase3/ for the cross-phase imports.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_PHASE3 = _ROOT / "phase3"
for _p in (str(_ROOT), str(_PHASE3), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finetune import load_pretrained                          # noqa: E402
from rollouts import (                                        # noqa: E402
    generate_group,
    _pick_tests,
    _row_is_grpo_pool,
)


# ---------------------------------------------------------------------------
# Test subsampling — keep cost bounded while preserving a usable signal
# ---------------------------------------------------------------------------
def _subsample_tests(
    row: dict,
    max_tests: int,
    rng_seed: int,
) -> tuple[dict, int, str]:
    """Return (row_copy, n_tests_used, source).

    row_copy is a shallow copy with its chosen test-source field
    (`private_tests` or `public_tests`) replaced by a max_tests-sized
    random subsample. Per-row RNG seed makes the sample stable across
    re-runs and resumes.

    If the original test count is <= max_tests, the row is returned with
    the full test set.
    """
    tests, source = _pick_tests(row)
    inputs = tests.get("input") or []
    outputs = tests.get("output") or []

    if source == "none" or not inputs:
        return dict(row), 0, source
    if len(inputs) != len(outputs):
        # Defensive — the audit said these are parallel lists.
        return dict(row), 0, source

    n_orig = len(inputs)
    if n_orig <= max_tests:
        return dict(row), n_orig, source

    rng = random.Random(rng_seed)
    chosen = rng.sample(range(n_orig), max_tests)
    sub_in  = [inputs[i]  for i in chosen]
    sub_out = [outputs[i] for i in chosen]

    row_copy = dict(row)
    field = f"{source}_tests"
    row_copy[field] = {"input": sub_in, "output": sub_out}
    return row_copy, max_tests, source


# ---------------------------------------------------------------------------
# Resume + sort helpers
# ---------------------------------------------------------------------------
def _load_already_scored(path: Path) -> set[int]:
    """Read existing JSONL and return the set of row_index values present."""
    if not path.exists():
        return set()
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                seen.add(int(rec["row_index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return seen


def _sort_jsonl_inplace(path: Path) -> int:
    """Read all records, sort by group_std desc, rewrite. Returns count."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda r: r.get("group_std", 0.0), reverse=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, path)
    return len(records)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
@dataclass
class Summary:
    n_total: int = 0
    n_dead: int = 0
    n_solvable: int = 0
    n_already_solved: int = 0
    max_reward_buckets: list[int] | None = None  # [zero, lt25, lt50, lt99, ge99]

    def __post_init__(self):
        if self.max_reward_buckets is None:
            self.max_reward_buckets = [0, 0, 0, 0, 0]

    def add(self, rec: dict) -> None:
        self.n_total += 1
        if rec["group_std"] == 0.0:
            self.n_dead += 1
        else:
            self.n_solvable += 1
        if rec["max_reward"] >= 0.99:
            self.n_already_solved += 1

        m = rec["max_reward"]
        if   m == 0.0: self.max_reward_buckets[0] += 1
        elif m < 0.25: self.max_reward_buckets[1] += 1
        elif m < 0.50: self.max_reward_buckets[2] += 1
        elif m < 0.99: self.max_reward_buckets[3] += 1
        else:          self.max_reward_buckets[4] += 1

    def print(self) -> None:
        N = max(self.n_total, 1)
        print(f"\n{'=' * 64}")
        print(f"CURRICULUM SUMMARY")
        print(f"{'=' * 64}")
        print(f"  total scored:               {self.n_total}")
        print(f"  DEAD  (group_std == 0):     {self.n_dead}  "
              f"({100*self.n_dead/N:.1f}%)")
        print(f"  SOLVABLE (group_std > 0):   {self.n_solvable}  "
              f"({100*self.n_solvable/N:.1f}%)")
        print(f"  ALREADY-SOLVED-SOMETIMES:   {self.n_already_solved}  "
              f"({100*self.n_already_solved/N:.1f}%)  "
              f"(any completion ≥ 0.99; subset of either group)")
        b = self.max_reward_buckets
        print(f"\n  Max-reward histogram:")
        print(f"    == 0.00:                  {b[0]}")
        print(f"    0.00 <  x < 0.25:         {b[1]}")
        print(f"    0.25 <= x < 0.50:         {b[2]}")
        print(f"    0.50 <= x < 0.99:         {b[3]}")
        print(f"    >= 0.99:                  {b[4]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score GRPO-pool problems for curriculum building.",
    )
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("../phase3/checkpoints_finetune_v2/best.pt"))
    parser.add_argument("--meta", type=Path,
                        default=Path("../phase3/finetune_data_v2/meta.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("curriculum.jsonl"),
                        help="JSONL output (append + resume-aware).")
    parser.add_argument("--reward-mode", type=str, default="fraction",
                        choices=["fraction", "advantage"],
                        help="Scoring reward. 'advantage' = beats-constant-"
                             "baseline reward (anti-hacking); 'fraction' = raw "
                             "pass rate (legacy).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score at most this many NEW (unscored) problems. "
                             "Already-scored row_indices are skipped first.")
    parser.add_argument("--solvable-target", type=int, default=None,
                        help="Stop once the TOTAL number of SOLVABLE problems "
                             "in the JSONL (including any from previous runs) "
                             "reaches this value. Acts as an early exit; "
                             "--limit still bounds the worst case.")
    parser.add_argument("--max-tests", type=int, default=15,
                        help="Cap reward evaluation at this many randomly "
                             "sampled tests per problem.")
    parser.add_argument("--G", type=int, default=8,
                        help="Group size for rollouts.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--test-timeout-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed. Per-row test-subsample seed is "
                             "(seed + row_index) so re-runs are reproducible.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sampling: G={args.G}  temperature={args.temperature}  "
          f"top_k={args.top_k}  max_new_tokens={args.max_new_tokens}")
    print(f"Reward eval cap: MAX_TESTS={args.max_tests} per problem")
    print(f"Output: {args.out.resolve()}")

    torch.manual_seed(args.seed)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    meta_path = args.meta.resolve()
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text())
    tokenizer_path = Path(meta["tokenizer_path"])
    if not tokenizer_path.is_absolute():
        tokenizer_path = (meta_path.parent.parent / tokenizer_path).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Tokenizer: {tokenizer_path}")

    # ── SFT checkpoint ──────────────────────────────────────────────────────
    ckpt = args.checkpoint.resolve()
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found at {ckpt}")
    model, model_cfg = load_pretrained(
        ckpt, gradient_checkpointing=False, device=device,
    )
    model.eval()

    # ── Dataset ─────────────────────────────────────────────────────────────
    load_dotenv(_ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN")
    print(f"\nLoading open-r1/codeforces-cots ...")
    ds = load_dataset(
        "open-r1/codeforces-cots",
        "solutions_py_decontaminated",
        split="train",
        token=hf_token,
    )
    print(f"Total rows: {len(ds):,}")

    # ── Build the work list: GRPO pool minus already-scored ─────────────────
    already_scored = _load_already_scored(args.out)
    if already_scored:
        print(f"Resuming: skipping {len(already_scored)} already-scored rows.")

    # Count SOLVABLE already in the JSONL so --solvable-target knows our
    # starting point.
    n_solvable_existing = 0
    if args.out.exists():
        with args.out.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("group_std", 0.0) > 0.0:
                        n_solvable_existing += 1
                except (json.JSONDecodeError, KeyError):
                    continue
    if args.solvable_target is not None:
        print(f"SOLVABLE target: {args.solvable_target} "
              f"(currently {n_solvable_existing} in JSONL; "
              f"need {max(0, args.solvable_target - n_solvable_existing)} more).")

    print("Scanning GRPO pool ...")
    pool_indices: list[int] = []
    pool_scan_pbar = tqdm(range(len(ds)), desc="pool scan", leave=False)
    for idx in pool_scan_pbar:
        if idx in already_scored:
            continue
        if _row_is_grpo_pool(ds[idx]):
            pool_indices.append(idx)
            if args.limit is not None and len(pool_indices) >= args.limit:
                break
    pool_scan_pbar.close()

    if not pool_indices:
        print("No new problems to score.")
        if args.out.exists():
            n = _sort_jsonl_inplace(args.out)
            print(f"Existing JSONL has {n} records; re-sorted by group_std desc.")
        sys.exit(0)

    print(f"To score: {len(pool_indices)} problems "
          + (f"(limited to first {args.limit})" if args.limit else "(full pool)"))

    # ── Score ───────────────────────────────────────────────────────────────
    summary = Summary()
    n_done = 0
    n_solvable_running = 0
    t_start = time.time()

    pbar = tqdm(pool_indices, desc="curriculum", unit="prob")
    f = args.out.open("a", encoding="utf-8")
    try:
        for idx in pbar:
            row = ds[int(idx)]
            row_sub, n_tests_used, source = _subsample_tests(
                row, args.max_tests, args.seed + int(idx),
            )

            try:
                group = generate_group(
                    model, tokenizer, row_sub,
                    G=args.G, temperature=args.temperature, top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens, device=device,
                    test_timeout_s=args.test_timeout_s,
                    reward_mode=args.reward_mode,
                )
            except Exception as e:
                # Don't let one bad problem crash a multi-hour pass.
                tqdm.write(f"  ! row {idx} failed: {type(e).__name__}: {e}")
                continue

            rewards = [g["reward"] for g in group]
            n_with_code = sum(1 for g in group if g["code"])
            mean = statistics.fmean(rewards) if rewards else 0.0
            std  = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            mx   = max(rewards) if rewards else 0.0

            rec = {
                "row_index":      int(idx),
                "problem_title":  (row.get("title") or row.get("name")
                                   or f"<row {idx}>"),
                "group_mean":     mean,
                "group_std":      std,
                "max_reward":     mx,
                "parseable_rate": n_with_code / max(args.G, 1),
                "n_tests_used":   n_tests_used,
                "test_source":    source,
            }
            f.write(json.dumps(rec) + "\n")
            summary.add(rec)

            n_done += 1
            if std > 0.0:
                n_solvable_running += 1
            pbar.set_postfix(
                solvable=n_solvable_running,
                rate=f"{100*n_solvable_running/max(n_done,1):.0f}%",
            )

            if n_done % 100 == 0:
                f.flush()
                os.fsync(f.fileno())
                tqdm.write(
                    f"  [checkpoint @ {n_done}: {n_solvable_running} new solvable "
                    f"({100*n_solvable_running/n_done:.1f}%); "
                    f"total in JSONL: {n_solvable_existing + n_solvable_running}]"
                )

            # --solvable-target: early exit once total SOLVABLE meets target.
            if (args.solvable_target is not None
                    and n_solvable_existing + n_solvable_running
                        >= args.solvable_target):
                tqdm.write(
                    f"  >> solvable-target {args.solvable_target} reached "
                    f"({n_solvable_existing + n_solvable_running} total). "
                    f"Stopping."
                )
                break
    except KeyboardInterrupt:
        tqdm.write("\nInterrupted — flushing and sorting partial JSONL ...")
    finally:
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
        f.close()

    elapsed = time.time() - t_start

    # ── Sort + summary ──────────────────────────────────────────────────────
    print(f"\nScored {n_done} problems in {elapsed:.1f}s "
          f"({elapsed/max(n_done,1):.2f}s/problem).")
    print("Sorting JSONL by group_std descending ...")
    total_in_file = _sort_jsonl_inplace(args.out)
    print(f"JSONL now has {total_in_file} records total.")

    summary.print()

    # Final hint for next step.
    if summary.n_solvable > 0:
        print(f"\n  Next: read {args.out.name} and pick the top-K by "
              f"group_std as the GRPO training pool.")
    else:
        print("\n  No SOLVABLE problems found — see rollouts.py recovery hints.")


if __name__ == "__main__":
    main()
