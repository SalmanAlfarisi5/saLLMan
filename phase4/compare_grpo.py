"""
phase4/compare_grpo.py — qualitative side-by-side of pre-GRPO vs post-GRPO
completions on held-out problems.

Goal: confirm the held-out reward gain (0.022 -> 0.238) shows up as visibly
better code a human can read, not just a scoring artifact.

Reuses existing machinery (no reinvention):
  - load_pretrained          (phase3/finetune.py) for both checkpoints
  - generate_group           (rollouts.py) for generation + scoring
  - _filter_and_split_curriculum (grpo.py) for the EXACT same held-out split

Fairness controls:
  - Same 5 held-out problems (seed=42 split), never trained on by either model.
  - Same decoding settings for both (temp=0.9, top_k=40, max_new_tokens=512).
  - Same torch RNG seed per problem before each model generates, so PRE and
    POST start sampling from the same state — differences come from weights,
    not sampling luck.
  - G=4 completions per model per problem to show consistency, not one
    cherry-picked sample.
  - reward_fraction against ALL available tests (offline — cost doesn't matter).

Usage
-----
    cd phase4
    python compare_grpo.py
    python compare_grpo.py --n-problems 5 --G 4
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tokenizers import Tokenizer

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_PHASE3 = _ROOT / "phase3"
for _p in (str(_ROOT), str(_PHASE3), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finetune import load_pretrained                          # noqa: E402
from rollouts import generate_group, _pick_tests              # noqa: E402
from grpo import _load_curriculum, _filter_and_split_curriculum  # noqa: E402


PRE_CKPT  = "../phase3/checkpoints_finetune_v2/best.pt"
POST_CKPT = "checkpoints_grpo_v1/best.pt"
META      = "../phase3/finetune_data_v2/meta.json"
CURRICULUM = "curriculum.jsonl"

# Must match the calibration/extension run so the held-out split is identical.
POOL_STD_THRESHOLD = 0.04
HOLDOUT_SIZE       = 15
SPLIT_SEED         = 42


def _gen_and_score(model, tokenizer, problem_row, G, device,
                   temperature, top_k, max_new_tokens, per_problem_seed):
    """Seed the RNG, then generate+score G completions via generate_group.

    Seeding immediately before the call makes PRE and POST share the same
    sampling trajectory for this problem.
    """
    torch.manual_seed(per_problem_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(per_problem_seed)
    return generate_group(
        model, tokenizer, problem_row,
        G=G, temperature=temperature, top_k=top_k,
        max_new_tokens=max_new_tokens, device=device,
        test_timeout_s=5.0,
    )


def _print_block(label: str, group: list[dict]) -> tuple[float, float]:
    """Print the G completions in a group; return (mean_reward, best_reward)."""
    rewards = [c["reward"] for c in group]
    print(f"\n  ┌─ {label}  (mean={statistics.fmean(rewards):.3f}, "
          f"best={max(rewards):.3f}) " + "─" * 20)
    for j, c in enumerate(group, start=1):
        code = c["code"]
        has_code = bool(code)
        print(f"  │")
        print(f"  │ [{label} completion {j}/{len(group)}]  "
              f"reward={c['reward']:.3f}  "
              f"{'(no <code> block parsed)' if not has_code else f'({len(code.splitlines())} lines)'}")
        if has_code:
            for line in code.splitlines():
                print(f"  │    {line}")
        else:
            # Show the raw completion head so we can see WHAT it produced
            # instead of code (repetition? prose? truncation?).
            raw = c["completion_text"][:300].replace("\n", "\n  │    ")
            print(f"  │    <raw completion head>: {raw}")
    print(f"  └" + "─" * 50)
    return statistics.fmean(rewards), max(rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre vs post GRPO comparison.")
    parser.add_argument("--n-problems", type=int, default=5)
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ───────────────────────────────────────────────────────────
    meta_path = (_HERE / META).resolve()
    meta = json.loads(meta_path.read_text())
    tok_path = Path(meta["tokenizer_path"])
    if not tok_path.is_absolute():
        tok_path = (meta_path.parent.parent / tok_path).resolve()
    tokenizer = Tokenizer.from_file(str(tok_path))

    # ── Held-out split (identical to training) ──────────────────────────────
    curriculum = _load_curriculum((_HERE / CURRICULUM).resolve(), solvable_only=True)
    _training, holdout = _filter_and_split_curriculum(
        curriculum, POOL_STD_THRESHOLD, HOLDOUT_SIZE, SPLIT_SEED,
    )
    chosen = holdout[:args.n_problems]
    print(f"Held-out problems selected (never trained on): "
          f"{[int(c['row_index']) for c in chosen]}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    load_dotenv(_ROOT / ".env")
    import os
    hf_token = os.environ.get("HF_TOKEN")
    ds = load_dataset("open-r1/codeforces-cots", "solutions_py_decontaminated",
                      split="train", token=hf_token)

    # ── Load BOTH models (two resident — same as GRPO's two-model plan) ─────
    print(f"\nLoading PRE-GRPO  (SFT):  {PRE_CKPT}")
    pre_model, _  = load_pretrained((_HERE / PRE_CKPT).resolve(),
                                    gradient_checkpointing=False, device=device)
    pre_model.eval()
    print(f"Loading POST-GRPO (step 1150): {POST_CKPT}")
    post_model, _ = load_pretrained((_HERE / POST_CKPT).resolve(),
                                    gradient_checkpointing=False, device=device)
    post_model.eval()

    # ── Per-problem side-by-side ────────────────────────────────────────────
    table_rows = []
    for i, crow in enumerate(chosen):
        row_idx = int(crow["row_index"])
        problem_row = ds[row_idx]
        title = crow.get("problem_title", f"<row {row_idx}>")
        tests, source = _pick_tests(problem_row)
        n_tests = len(tests.get("input", []))

        print("\n" + "=" * 72)
        print(f"PROBLEM {i+1}/{len(chosen)} — row {row_idx} — {title!r}")
        print(f"  ({source} tests: {n_tests})")
        # Show the trimmed problem statement (first ~6 lines of description).
        desc = (problem_row.get("description") or "").strip().splitlines()
        for line in desc[:6]:
            print(f"  | {line}")
        if len(desc) > 6:
            print(f"  | ... ({len(desc)-6} more lines)")
        print("=" * 72)

        pseed = args.seed + i * 101  # distinct per problem, shared by both models

        pre_group  = _gen_and_score(pre_model, tokenizer, problem_row, args.G,
                                    device, args.temperature, args.top_k,
                                    args.max_new_tokens, pseed)
        pre_mean, pre_best = _print_block("PRE ", pre_group)

        post_group = _gen_and_score(post_model, tokenizer, problem_row, args.G,
                                    device, args.temperature, args.top_k,
                                    args.max_new_tokens, pseed)
        post_mean, post_best = _print_block("POST", post_group)

        table_rows.append({
            "row": row_idx, "title": title,
            "pre_mean": pre_mean, "post_mean": post_mean,
            "pre_best": pre_best, "post_best": post_best,
        })

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    hdr = f"{'problem':<34} {'pre_mean':>8} {'post_mean':>9} {'pre_best':>8} {'post_best':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in table_rows:
        name = (r["title"][:32]) if r["title"] else f"row {r['row']}"
        print(f"{name:<34} {r['pre_mean']:>8.3f} {r['post_mean']:>9.3f} "
              f"{r['pre_best']:>8.3f} {r['post_best']:>9.3f}")
    print("-" * len(hdr))
    overall = {
        "pre_mean":  statistics.fmean(r["pre_mean"]  for r in table_rows),
        "post_mean": statistics.fmean(r["post_mean"] for r in table_rows),
        "pre_best":  statistics.fmean(r["pre_best"]  for r in table_rows),
        "post_best": statistics.fmean(r["post_best"] for r in table_rows),
    }
    print(f"{'OVERALL':<34} {overall['pre_mean']:>8.3f} {overall['post_mean']:>9.3f} "
          f"{overall['pre_best']:>8.3f} {overall['post_best']:>9.3f}")


if __name__ == "__main__":
    main()
