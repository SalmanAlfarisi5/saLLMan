"""
phase4/rollouts.py — GRPO rollout generator (group of completions per problem).

Public function:

    generate_group(model, tokenizer, problem_row, G=8, temperature=0.9,
                   top_k=40, max_new_tokens=512, device) -> list[dict]

For one problem, samples G independent completions from the SFT-trained
model, extracts the <code>...</code> block from each, and scores it via
phase4/code_executor.reward_fraction against the problem's tests
(private_tests if non-empty, else public_tests).

Returns G dicts:
    {"completion_ids": list[int],
     "completion_text": str,
     "code": str,
     "reward": float}

The __main__ runs this on 2-3 problems sampled from the GRPO pool and
prints per-problem reward distributions (mean / std / list of G rewards).

Why this exists (not the training loop yet)
-------------------------------------------
GRPO normalises rewards across a *group* of completions for the same
prompt. If every completion in a group earns identical reward, the
group-relative advantage is zero and there's no gradient signal from
that problem. The minimum viable input to GRPO is therefore: "from
this base model, can I roll out G completions and get a reward
distribution with non-zero variance?"

Run this on the SFT checkpoint with a representative sample of GRPO-pool
problems. The output answers:

  - Are completions parseable? (i.e., is the <code> tag still emitted
    reliably enough off the SFT distribution at temperature=0.9?)
  - Is there reward spread? std > 0 across the G group means GRPO can
    actually learn from this problem.
  - What fraction of problems give std == 0? Those are dead-signal
    problems we'd want to filter from the GRPO pool, or rescue with
    higher temperature / larger G / easier problem subset.

Usage
-----
    cd phase4
    python rollouts.py                                # default: 3 problems, G=8
    python rollouts.py --num-problems 5 --G 16 --temperature 1.0
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tokenizers import Tokenizer


# sys.path: project root for any phase3 qualified imports, plus phase3/
# for the bare imports phase3 modules use internally (finetune.py does
# `from decoder_only_v3 import ...` which needs phase3/ on sys.path).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_PHASE3 = _ROOT / "phase3"
for _p in (str(_ROOT), str(_PHASE3), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finetune import load_pretrained                          # noqa: E402
from finetune_data_prep import (                              # noqa: E402
    build_trimmed_problem,
    PROBLEM_OPEN, PROBLEM_CLOSE, REASON_OPEN,
    BOS_IDX, EOS_IDX,
)
from code_executor import reward_fraction                     # noqa: E402


# ---------------------------------------------------------------------------
# Code extraction from a completion
# ---------------------------------------------------------------------------
# The SFT-trained response shape (see phase3/finetune_data_prep.py) is:
#     {editorial}</reasoning>\n<code>\n{code}\n</code>
#
# At inference we feed the prompt up to "<reasoning>\n" and let the model
# generate the rest. We then need to recover the {code} portion to score.
# Strict matching: require both <code> and </code> in the completion. If
# either is missing, we treat the completion as un-scoreable and assign
# reward 0.0 (GRPO will then pressure the model toward emitting the
# scaffolding, which is what we want).
CODE_OPEN_TAG  = "<code>"
CODE_CLOSE_TAG = "</code>"


def _extract_code_block(completion_text: str) -> str:
    """Return the code between the FIRST <code> and the next </code>.

    Returns "" if either tag is missing — the caller treats "" as a
    no-code completion and scores it 0.0.
    """
    open_idx = completion_text.find(CODE_OPEN_TAG)
    if open_idx == -1:
        return ""
    code_start = open_idx + len(CODE_OPEN_TAG)
    # The training format puts \n right after <code>; tolerate either.
    if code_start < len(completion_text) and completion_text[code_start] == "\n":
        code_start += 1
    close_idx = completion_text.find(CODE_CLOSE_TAG, code_start)
    if close_idx == -1:
        return ""
    return completion_text[code_start:close_idx].rstrip()


# ---------------------------------------------------------------------------
# Test picker — private over public
# ---------------------------------------------------------------------------
def _pick_tests(problem_row: dict) -> tuple[dict, str]:
    """Choose the test set to score against.

    private_tests are typically richer than public_tests (more cases,
    designed to catch wrong-but-passing-on-samples solutions). They're
    also harder for the model to game by overfitting on the few public
    examples it might see in its training input. We prefer them.

    Returns (tests_dict, "private" | "public" | "none").
    """
    priv = problem_row.get("private_tests") or {}
    if (priv.get("input") and priv.get("output")
            and len(priv["input"]) == len(priv["output"])):
        return priv, "private"
    pub = problem_row.get("public_tests") or {}
    if (pub.get("input") and pub.get("output")
            and len(pub["input"]) == len(pub["output"])):
        return pub, "public"
    return {}, "none"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _build_prompt_text(problem_row: dict) -> str:
    """Same prefix shape as SFT training — see phase3/finetune_data_prep.py.

    Putting REASON_OPEN at the end of the prompt means the model's first
    generated token continues the response (i.e. reasoning content).
    """
    problem_text = build_trimmed_problem(
        problem_row.get("description")   or "",
        problem_row.get("input_format")  or "",
        problem_row.get("output_format") or "",
    )
    return PROBLEM_OPEN + problem_text + PROBLEM_CLOSE + REASON_OPEN


# ---------------------------------------------------------------------------
# Core: generate one group of G completions for one problem
# ---------------------------------------------------------------------------
def generate_group(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    problem_row: dict,
    G: int = 8,
    temperature: float = 0.9,
    top_k: int = 40,
    max_new_tokens: int = 512,
    device: torch.device | None = None,
    test_timeout_s: float = 5.0,
) -> list[dict]:
    """Sample G independent completions for one problem; score each.

    Sequential — G separate calls to `model.generate`. Each call uses
    KV-cache internally (see GPTv3.generate). This is the right shape for
    the smoke test; the eventual GRPO loop will want a batched-generate
    path for efficiency.
    """
    if device is None:
        device = next(model.parameters()).device

    tests, source = _pick_tests(problem_row)
    if source == "none":
        # No tests → no reward signal. The caller should have filtered
        # the pool, but we degrade gracefully.
        tests = {"input": [], "output": []}

    prompt_text = _build_prompt_text(problem_row)
    prompt_ids  = [BOS_IDX] + tokenizer.encode(prompt_text).ids
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    results: list[dict] = []
    for g in range(G):
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            eos_id=EOS_IDX,
            temperature=temperature,
            top_k=top_k,
        )
        completion_ids  = out[0, len(prompt_ids):].tolist()
        completion_text = tokenizer.decode(completion_ids)
        code = _extract_code_block(completion_text)

        if code and tests["input"]:
            reward = reward_fraction(code, tests, timeout_s=test_timeout_s)
        else:
            reward = 0.0

        results.append({
            "completion_ids":  completion_ids,
            "completion_text": completion_text,
            "code":            code,
            "reward":          reward,
        })

    return results


# ---------------------------------------------------------------------------
# Helpers for the smoke test main
# ---------------------------------------------------------------------------
def _row_is_grpo_pool(row: dict) -> bool:
    """Same predicate as audit_tests.py's GRPO pool definition: needs the
    three problem fields and at least one usable test source."""
    def nz(x): return isinstance(x, str) and bool(x.strip())
    if not (nz(row.get("description"))
            and nz(row.get("input_format"))
            and nz(row.get("output_format"))):
        return False
    tests, source = _pick_tests(row)
    return source != "none"


def _sample_problems(ds, n: int, seed: int) -> list[dict]:
    """Pick `n` GRPO-pool rows, deterministic given seed."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    picked: list[dict] = []
    for idx in indices:
        row = ds[idx]
        if _row_is_grpo_pool(row):
            picked.append(row)
            if len(picked) >= n:
                break
    return picked


def _print_problem_summary(idx: int, row: dict, source: str,
                            n_tests: int) -> None:
    """One-line problem description for the per-problem header."""
    title = (row.get("title") or row.get("name") or "<no title>")
    desc = (row.get("description") or "").strip().splitlines()
    desc_first = desc[0] if desc else ""
    if len(desc_first) > 100:
        desc_first = desc_first[:97] + "..."
    print(f"\n=== PROBLEM {idx} — {title!r} ({source} tests: {n_tests}) ===")
    print(f"    description (line 1): {desc_first}")


def _print_group_summary(rewards: list[float]) -> None:
    """Mean / std / dead-signal flag for a group's rewards."""
    mean = statistics.fmean(rewards)
    # pstdev (population) is what GRPO uses for advantage normalisation,
    # so report that — not sample stdev.
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    n_zero    = sum(1 for r in rewards if r == 0.0)
    n_perfect = sum(1 for r in rewards if r >= 0.99)
    n_partial = sum(1 for r in rewards if 0.0 < r < 0.99)

    formatted = "[" + ", ".join(f"{r:.2f}" for r in rewards) + "]"
    print(f"    rewards (G={len(rewards)}): {formatted}")
    print(f"    mean={mean:.3f}  std={std:.3f}  "
          f"zero={n_zero}  partial={n_partial}  perfect={n_perfect}")

    if std == 0.0:
        if mean == 0.0:
            print(f"    !! DEAD SIGNAL (all rewards = 0): GRPO advantage = 0 "
                  f"for this problem. Too hard for current model.")
        elif mean >= 0.99:
            print(f"    !! DEAD SIGNAL (all rewards = 1): GRPO advantage = 0 "
                  f"for this problem. Already solved — no learning here.")
        else:
            print(f"    !! DEAD SIGNAL (zero variance): advantage = 0.")
    else:
        print(f"    OK — non-zero variance, GRPO can learn from this problem.")


# ---------------------------------------------------------------------------
# Main — load SFT model, run generate_group on a few problems
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test GRPO rollouts on the SFT-trained model.",
    )
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("../phase3/checkpoints_finetune_v2/best.pt"),
                        help="SFT checkpoint to roll out from.")
    parser.add_argument("--meta", type=Path,
                        default=Path("../phase3/finetune_data_v2/meta.json"),
                        help="FT meta.json (used to locate the BPE tokenizer).")
    parser.add_argument("--num-problems", type=int, default=3,
                        help="How many GRPO-pool problems to sample.")
    parser.add_argument("--G", type=int, default=8,
                        help="Group size (completions per problem).")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--test-timeout-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for problem selection AND sampling RNG.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sampling: G={args.G}  temperature={args.temperature}  "
          f"top_k={args.top_k}  max_new_tokens={args.max_new_tokens}")

    # Determinism (per the user-supplied seed). We seed once; G generates
    # in sequence will then produce G different samples because the RNG
    # state advances across multinomial draws.
    torch.manual_seed(args.seed)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    import json
    meta_path = args.meta.resolve()
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text())
    tokenizer_path = Path(meta["tokenizer_path"])
    if not tokenizer_path.is_absolute():
        tokenizer_path = (meta_path.parent.parent / tokenizer_path).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Tokenizer: {tokenizer_path} (vocab={tokenizer.get_vocab_size():,})")

    # ── SFT checkpoint ──────────────────────────────────────────────────────
    ckpt = args.checkpoint.resolve()
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found at {ckpt}")
    model, model_cfg = load_pretrained(
        ckpt, gradient_checkpointing=False, device=device,
    )
    model.eval()

    # ── Dataset + problem sample ────────────────────────────────────────────
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
    problems = _sample_problems(ds, args.num_problems, args.seed)
    print(f"Sampled {len(problems)} GRPO-pool problems (seed={args.seed}).")

    # ── Rollouts ────────────────────────────────────────────────────────────
    all_rewards_means: list[float] = []
    n_dead = 0

    for i, row in enumerate(problems, start=1):
        tests, source = _pick_tests(row)
        n_tests = len(tests.get("input", []))
        _print_problem_summary(i, row, source, n_tests)

        t0 = time.time()
        group = generate_group(
            model, tokenizer, row,
            G=args.G, temperature=args.temperature, top_k=args.top_k,
            max_new_tokens=args.max_new_tokens, device=device,
            test_timeout_s=args.test_timeout_s,
        )
        elapsed = time.time() - t0

        rewards = [g["reward"] for g in group]
        n_with_code = sum(1 for g in group if g["code"])
        print(f"    completions with parseable <code>: {n_with_code}/{len(group)}")
        _print_group_summary(rewards)
        print(f"    elapsed: {elapsed:.1f}s "
              f"({elapsed/args.G:.1f}s per completion incl. test exec)")

        all_rewards_means.append(statistics.fmean(rewards))
        if (max(rewards) - min(rewards)) == 0.0:
            n_dead += 1

    # ── Aggregate verdict ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("ACROSS-PROBLEM SUMMARY")
    print("=" * 64)
    print(f"  problems sampled:        {len(problems)}")
    print(f"  dead-signal problems:    {n_dead}/{len(problems)} "
          f"(std == 0 → GRPO can't learn from these)")
    print(f"  per-problem mean reward: "
          f"{['%.3f' % m for m in all_rewards_means]}")
    print(f"  overall mean:            {statistics.fmean(all_rewards_means):.3f}")

    if n_dead == len(problems):
        print("\n  >> All problems gave zero variance. The SFT model can't")
        print("  >> usefully roll out for GRPO at these settings. Try:")
        print("  >>   - higher temperature (--temperature 1.1)")
        print("  >>   - larger group (--G 16)")
        print("  >>   - easier problem subset (filter by problem difficulty)")
    elif n_dead > 0:
        print(f"\n  >> {n_dead} of {len(problems)} problems are dead-signal — "
              f"expect to filter")
        print(f"  >> these out of the GRPO pool, or accept that ~"
              f"{100*n_dead/len(problems):.0f}% of update steps")
        print(f"  >> will be wasted on them at current sampling.")
    else:
        print(f"\n  >> All sampled problems have non-zero reward spread. "
              f"SFT model is")
        print(f"  >> a workable GRPO starting point at these settings.")


if __name__ == "__main__":
    main()
