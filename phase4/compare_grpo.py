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
POST_CKPT = "checkpoints_grpo_v2/best.pt"          # v2 = anti-hack reward
META      = "../phase3/finetune_data_v2/meta.json"
CURRICULUM = "curriculum_v2.jsonl"                  # v2 curriculum

# Must match the v2 calibration so the held-out split is identical:
#   grpo.py --calibration --holdout-size 5 --pool-std-threshold 0.0
POOL_STD_THRESHOLD = 0.0
HOLDOUT_SIZE       = 5
SPLIT_SEED         = 42


def _gen_and_score(model, tokenizer, problem_row, G, device,
                   temperature, top_k, max_new_tokens, per_problem_seed):
    """Seed the RNG, then generate+score G completions via generate_group
    in ADVANTAGE mode — so each completion carries raw pass fraction,
    advantage (anti-hack reward), and the constant-output guard flag.

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
        reward_mode="advantage",
    )


def _print_block(label: str, group: list[dict]) -> dict:
    """Print the G completions; return summary {adv_mean, adv_best,
    frac_mean, frac_best, guard_count}.

    Each completion shows raw pass fraction, advantage, and whether the
    constant-output guard fired — so a constant hack is visible as
    'frac=0.40 adv=0.00 GUARD'.
    """
    advs   = [c["reward"] for c in group]              # advantage (reward_mode=advantage)
    fracs  = [c.get("reward_fraction", 0.0) for c in group]
    guards = [c.get("guard_fired", False) for c in group]
    print(f"\n  ┌─ {label}  (adv mean={statistics.fmean(advs):.3f} best={max(advs):.3f} | "
          f"raw_frac mean={statistics.fmean(fracs):.3f} best={max(fracs):.3f} | "
          f"guard fired {sum(guards)}/{len(group)}) " + "─" * 6)
    for j, c in enumerate(group, start=1):
        code = c["code"]
        has_code = bool(code)
        flag = "  ⚠ GUARD(constant-output)" if c.get("guard_fired") else ""
        print(f"  │")
        print(f"  │ [{label} completion {j}/{len(group)}]  "
              f"frac={c.get('reward_fraction', 0.0):.3f} adv={c['reward']:.3f}{flag}  "
              f"{'(no <code> block parsed)' if not has_code else f'({len(code.splitlines())} lines)'}")
        if has_code:
            for line in code.splitlines():
                print(f"  │    {line}")
        else:
            raw = c["completion_text"][:300].replace("\n", "\n  │    ")
            print(f"  │    <raw completion head>: {raw}")
    print(f"  └" + "─" * 50)
    return {
        "adv_mean":    statistics.fmean(advs),
        "adv_best":    max(advs),
        "frac_mean":   statistics.fmean(fracs),
        "frac_best":   max(fracs),
        "guard_count": sum(guards),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre vs post GRPO comparison.")
    parser.add_argument("--n-problems", type=int, default=5)
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pre",  type=str, default=PRE_CKPT,
                        help="Pre-GRPO checkpoint (default: SFT best.pt).")
    parser.add_argument("--post", type=str, default=POST_CKPT,
                        help="Post-GRPO checkpoint (default: grpo_v2/best.pt).")
    parser.add_argument("--curriculum", type=str, default=CURRICULUM,
                        help="Curriculum JSONL for the held-out split.")
    parser.add_argument("--holdout-size", type=int, default=HOLDOUT_SIZE)
    parser.add_argument("--pool-std-threshold", type=float, default=POOL_STD_THRESHOLD)
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
    curriculum = _load_curriculum((_HERE / args.curriculum).resolve(), solvable_only=True)
    _training, holdout = _filter_and_split_curriculum(
        curriculum, args.pool_std_threshold, args.holdout_size, SPLIT_SEED,
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
    print(f"\nLoading PRE-GRPO  (SFT):  {args.pre}")
    pre_model, _  = load_pretrained((_HERE / args.pre).resolve(),
                                    gradient_checkpointing=False, device=device)
    pre_model.eval()
    print(f"Loading POST-GRPO:        {args.post}")
    post_model, _ = load_pretrained((_HERE / args.post).resolve(),
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
        pre_s = _print_block("PRE ", pre_group)

        post_group = _gen_and_score(post_model, tokenizer, problem_row, args.G,
                                    device, args.temperature, args.top_k,
                                    args.max_new_tokens, pseed)
        post_s = _print_block("POST", post_group)

        table_rows.append({"row": row_idx, "title": title,
                           "pre": pre_s, "post": post_s})

    # ── Summary table ───────────────────────────────────────────────────────
    # Two metrics side by side:
    #   adv  = anti-hack reward (max(0, frac - constant_baseline), guard-zeroed)
    #   frac = raw pass fraction (what the OLD reward used — inflated by constants)
    #   guard = # of completions (out of G) that were constant-output hacks
    print("\n" + "=" * 78)
    print("SUMMARY  (adv = anti-hack reward, frac = raw pass rate, gG = guard fires/run)")
    print("=" * 78)
    hdr = (f"{'problem':<26} {'pre_adv':>7} {'post_adv':>8} "
           f"{'pre_frac':>8} {'post_frac':>9} {'pre_gG':>6} {'post_gG':>7}")
    print(hdr)
    print("-" * len(hdr))
    G = args.G
    for r in table_rows:
        name = (r["title"][:24]) if r["title"] else f"row {r['row']}"
        print(f"{name:<26} {r['pre']['adv_mean']:>7.3f} {r['post']['adv_mean']:>8.3f} "
              f"{r['pre']['frac_mean']:>8.3f} {r['post']['frac_mean']:>9.3f} "
              f"{r['pre']['guard_count']:>4}/{G} {r['post']['guard_count']:>5}/{G}")
    print("-" * len(hdr))
    o = lambda side, key: statistics.fmean(r[side][key] for r in table_rows)
    tot_pre_g  = sum(r["pre"]["guard_count"]  for r in table_rows)
    tot_post_g = sum(r["post"]["guard_count"] for r in table_rows)
    denom = G * len(table_rows)
    print(f"{'OVERALL':<26} {o('pre','adv_mean'):>7.3f} {o('post','adv_mean'):>8.3f} "
          f"{o('pre','frac_mean'):>8.3f} {o('post','frac_mean'):>9.3f} "
          f"{tot_pre_g:>3}/{denom} {tot_post_g:>4}/{denom}")


if __name__ == "__main__":
    main()
