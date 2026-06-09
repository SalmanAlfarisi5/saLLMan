"""
phase4/grpo.py — GRPO training loop for the SFT-trained 97 M policy.

Two-model memory plan (GRPO's win vs PPO's three)
-------------------------------------------------
  - policy:    GPTv3 from checkpoints_finetune_v2/best.pt, TRAINABLE.
  - reference: a SECOND frozen GPTv3 from the same checkpoint, eval mode,
               requires_grad=False, used ONLY for the KL term.

No value/critic network — GRPO replaces the per-token value baseline with
a group-relative mean. Two resident models instead of three.

Per-step recipe
---------------
  1. Pick a SOLVABLE problem from curriculum.jsonl (group_std > 0 only —
     dead-signal problems contribute zero advantage and waste a step).
  2. Roll out G completions with the policy at the configured temperature.
  3. Reward each via reward_fraction over a subsampled test set
     (max_tests cap + per-completion wall-clock budget so one infinite
     loop can't stall the step).
  4. Advantage = (r_i - mean(r)) / (std(r) + 1e-4). Same scalar advantage
     gates every response token in completion i — outcome supervision.
  5. Loss math (every term per response token, then masked-mean):
         rho       = exp(logp_policy - logp_old)
         surrogate = min(rho · A_i, clip(rho, 1-eps, 1+eps) · A_i)
         kl_k3     = exp(logp_ref - logp_policy) - (logp_ref - logp_policy) - 1
         obj       = surrogate - kl_beta · kl_k3
         loss      = -mean_over_response(obj)  averaged over G completions
     logp_old (the policy snapshot at sample time) and logp_ref are
     DETACHED — only logp_policy carries gradient.

Smoke test
----------
    python grpo.py --smoke
runs 20 steps on the SOLVABLE-only subset of curriculum.jsonl. Reports
per-step loss / kl / reward / advantage / GPU peak and a 20-step trace.

The "smoke" run is to prove mechanics: loss finite, KL starts ~0 and
grows, mean reward doesn't collapse, memory under 8 GB. Quality of the
resulting policy is not the point at 20 steps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from tokenizers import Tokenizer


# sys.path: project root + phase3/ + phase4/. Mirrors rollouts.py /
# build_curriculum.py so the cross-phase imports resolve identically.
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
from code_executor import run_solution                        # noqa: E402
from rollouts import _pick_tests, _extract_code_block         # noqa: E402
from build_curriculum import _subsample_tests                 # noqa: E402


# ===========================================================================
# 1. Configuration
# ===========================================================================
@dataclass
class GRPOConfig:
    # ── Paths ────────────────────────────────────────────────────────────────
    sft_ckpt: str   = "../phase3/checkpoints_finetune_v2/best.pt"
    meta:     str   = "../phase3/finetune_data_v2/meta.json"
    curriculum: str = "curriculum.jsonl"
    out_dir:    str = "checkpoints_grpo_v1"

    # ── GRPO core ────────────────────────────────────────────────────────────
    G: int                  = 8         # group size
    clip_eps: float         = 0.2       # PPO-style importance-ratio clip
    kl_beta:  float         = 0.04      # KL penalty to reference
    problems_per_step: int  = 1         # gradient accumulation across problems

    # ── Sampling (rollout) ──────────────────────────────────────────────────
    temperature: float      = 0.9
    top_k:       int        = 40
    max_new_tokens: int     = 512       # match rollouts.py/build_curriculum so parse_rate doesn't tank

    # ── Reward execution (per-completion) ───────────────────────────────────
    # MAX_TESTS=15 for calibration runs — reward quality > speed.
    # Per-completion budget kills runaway loops without relying on the
    # per-test timeout firing once per test.
    max_tests:          int   = 15
    test_timeout_s:     float = 3.0
    completion_budget_s: float = 30.0

    # ── Curriculum filtering + held-out eval (calibration mode) ─────────────
    # In calibration mode the curriculum is filtered by std, split into a
    # training pool + held-out eval set, and "best.pt" tracks held-out
    # reward (not training reward — which can rise via overfit).
    pool_std_threshold:        float = 0.04   # drop barely-learnable tail
    holdout_size:              int   = 15     # fixed eval set
    eval_interval:             int   = 25     # held-out eval every N steps
    eval_g:                    int   = 8      # group size for eval rollouts
    eval_completion_budget_s:  float = 30.0   # per-completion budget for eval
    # Per-problem test cap for held-out eval. 999999 ≈ uncapped. Setting
    # this to e.g. 10 trades a small amount of trend precision for a big
    # eval-cost reduction once the model produces working code; the
    # held-out's job is to show direction, not be a final benchmark.
    # Subsampling is deterministic per (seed, row_index), so the same
    # problem uses the same tests across every eval pass — the trend
    # remains an honest comparison.
    eval_max_tests:            int   = 999_999
    # Weighted sampling: weight ∝ min(n_tests_used, weight_test_cap).
    # Pulls samples toward 12-15-test problems (robust signal) over
    # 1-test "spike" problems (overfit-prone). Set False for uniform.
    weighted_sampling: bool = True
    weight_test_cap:   int  = 15

    # ── Optimizer (RL LR is even lower than SFT 2e-5) ───────────────────────
    lr:           float = 1e-6
    weight_decay: float = 0.0   # GRPO commonly uses no WD to avoid fighting the policy gradient
    beta1:        float = 0.9
    beta2:        float = 0.95
    grad_clip:    float = 1.0

    # ── Misc ─────────────────────────────────────────────────────────────────
    total_steps:         int  = 100
    log_interval:        int  = 1
    checkpoint_interval: int  = 25
    reward_ma_window:    int  = 10    # moving-average window for "best" tracking
    seed:                int  = 42
    use_amp:             bool = True
    gradient_checkpointing: bool = False


# ===========================================================================
# 2. Helpers
# ===========================================================================
def _build_prompt_ids(problem_row: dict, tokenizer: Tokenizer) -> list[int]:
    """Same prefix as SFT training (see finetune_data_prep.format_prompt_response)."""
    problem_text = build_trimmed_problem(
        problem_row.get("description")   or "",
        problem_row.get("input_format")  or "",
        problem_row.get("output_format") or "",
    )
    text = PROBLEM_OPEN + problem_text + PROBLEM_CLOSE + REASON_OPEN
    return [BOS_IDX] + tokenizer.encode(text).ids


def _per_token_logp(
    model: nn.Module,
    input_ids: torch.Tensor,           # 1-D LongTensor on device
    response_start: int,               # index where the response begins
    use_amp: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute log p_θ(input_ids[t] | input_ids[<t]) for every t in [1, L).

    Returns (logp, mask), both shape (L-1,):
        logp[i] = log p_θ(input_ids[i+1] | input_ids[:i+1])
        mask[i] = True iff position i+1 is in the response.

    Forward runs under bf16 autocast for the matmuls; the log_softmax is
    upcast to fp32 because importance ratios and KL are sensitive to
    cumulative log-prob precision.
    """
    L = input_ids.size(0)
    if L < 2:
        return None, None

    device_type = "cuda" if input_ids.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=use_amp,
                            dtype=torch.bfloat16):
        logits, _ = model(input_ids.unsqueeze(0))
    logits = logits[0].float()              # (L, V) fp32
    pred_logits = logits[:-1]                # (L-1, V) — pred[t] is for input_ids[t+1]
    target_ids  = input_ids[1:]              # (L-1,)

    logp_dist = F.log_softmax(pred_logits, dim=-1)
    logp_chosen = logp_dist.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    mask = torch.zeros(L - 1, dtype=torch.bool, device=input_ids.device)
    # Prediction for position p (1 <= p <= L-1) is at index p-1 in logp.
    # Response covers positions [response_start, L-1], so mask covers
    # indices [response_start-1, L-2].
    lo = max(0, response_start - 1)
    if lo < L - 1:
        mask[lo:] = True
    return logp_chosen, mask


def _reward_with_budget(
    code: str, tests: dict,
    timeout_s: float, budget_s: float,
) -> tuple[float, int]:
    """Like reward_fraction but with a per-completion wall-clock cap.

    Returns (reward, n_tests_run). Reward = n_pass / n_total — NOT
    n_pass / n_run — so a slow completion can't game the metric by
    running only the easy tests before the budget elapses.
    """
    if not code:
        return 0.0, 0
    inputs  = tests.get("input")  or []
    outputs = tests.get("output") or []
    if not inputs or len(inputs) != len(outputs):
        return 0.0, 0
    n_total = len(inputs)
    n_pass  = 0
    n_run   = 0
    start = time.time()
    for tin, texp in zip(inputs, outputs):
        if time.time() - start >= budget_s:
            break
        if run_solution(code, tin, texp, timeout_s=timeout_s):
            n_pass += 1
        n_run += 1
    return n_pass / n_total, n_run


def _load_curriculum(path: Path, solvable_only: bool = True) -> list[dict]:
    """Read curriculum.jsonl and optionally filter to SOLVABLE-only rows."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if solvable_only and r.get("group_std", 0.0) <= 0.0:
                continue
            rows.append(r)
    return rows


# ===========================================================================
# 3. The GRPO step
# ===========================================================================
def grpo_step(
    policy: nn.Module,
    reference: nn.Module,
    tokenizer: Tokenizer,
    problem_row: dict,
    cfg: GRPOConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    step_idx: int,
    do_optimizer_step: bool = True,
) -> dict:
    """One GRPO update on one problem.

    If the group's reward std is 0, returns {"skipped": True, ...} without
    calling backward / optimizer.step — there's no gradient signal to
    learn from. The caller can pull another problem.

    `do_optimizer_step=False` lets the outer loop accumulate gradients
    across multiple problems before stepping the optimizer.
    """
    G = cfg.G

    # ── 1. Subsample tests (deterministic per step+problem) ─────────────────
    test_seed = cfg.seed + step_idx * 7919 + problem_row.get("row_index_hint", 0)
    row_sub, n_tests_used, source = _subsample_tests(
        problem_row, cfg.max_tests, test_seed,
    )
    tests_field = f"{source}_tests"
    tests = row_sub.get(tests_field) or {"input": [], "output": []}

    # ── 2. Build the prompt once ────────────────────────────────────────────
    prompt_ids = _build_prompt_ids(problem_row, tokenizer)
    L_prompt = len(prompt_ids)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    # ── 3. Roll out G completions with the policy (no grad) ────────────────
    # We use eval mode for generation so the KV-cache + RoPE paths align
    # with how GPTv3.generate is exercised at inference time. Dropout is
    # already 0 in the SFT config so train/eval are numerically identical
    # — eval here is for the cache.
    policy.eval()
    completions: list[dict] = []
    for g in range(G):
        x = prompt_tensor.unsqueeze(0)
        with torch.no_grad():
            out = policy.generate(
                x,
                max_new_tokens=cfg.max_new_tokens,
                eos_id=EOS_IDX,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
            )
        full_ids = out[0].clone()
        completion_ids = full_ids[L_prompt:]
        completion_text = tokenizer.decode(completion_ids.tolist())
        code = _extract_code_block(completion_text)
        completions.append({
            "full_ids":        full_ids,
            "completion_ids":  completion_ids,
            "completion_text": completion_text,
            "code":            code,
        })

    # ── 4. Score each completion (CPU subprocesses) ─────────────────────────
    for c in completions:
        c["reward"], c["n_tests_run"] = _reward_with_budget(
            c["code"], tests,
            timeout_s=cfg.test_timeout_s,
            budget_s=cfg.completion_budget_s,
        )

    rewards_list = [c["reward"] for c in completions]
    r_tensor = torch.tensor(rewards_list, dtype=torch.float, device=device)
    r_mean = r_tensor.mean().item()
    # Population std for advantage normalisation (matches DeepSeek-R1 / TRL).
    r_std  = r_tensor.std(unbiased=False).item()

    # ── 5. Dead-signal early exit ───────────────────────────────────────────
    if r_std == 0.0:
        return {
            "skipped":         True,
            "mean_reward":     r_mean,
            "std_reward":      r_std,
            "parse_rate":      sum(1 for c in completions if c["code"]) / G,
            "pos_reward_rate": sum(1 for r in rewards_list if r > 0) / G,
            "n_tests_used":    n_tests_used,
            "test_source":     source,
        }

    advantages = (r_tensor - r_mean) / (r_std + 1e-4)

    # ── 6. Snapshot logp_old + logp_ref per completion (both detached) ──────
    # logp_old is the policy's per-token log-prob AT SAMPLE TIME. Since
    # we do one gradient update per rollout, sample time == "right now,
    # before optimizer.step()". Detach so it's frozen for the loss.
    for c in completions:
        with torch.no_grad():
            logp_old, mask = _per_token_logp(
                policy, c["full_ids"], L_prompt, cfg.use_amp,
            )
            logp_ref, _ = _per_token_logp(
                reference, c["full_ids"], L_prompt, cfg.use_amp,
            )
        if logp_old is None or mask is None or not bool(mask.any()):
            c["loss_eligible"] = False
            continue
        c["logp_old"] = logp_old.detach()
        c["logp_ref"] = logp_ref.detach()
        c["mask"]     = mask
        c["loss_eligible"] = True

    # ── 7. Forward-with-grad + loss + backward, sequential over G ───────────
    # Per-completion forward keeps activation memory bounded to one
    # sequence at a time. Gradients accumulate in policy.grad across
    # completions, then we either step or hand off to the caller.
    policy.train()
    if do_optimizer_step:
        optimizer.zero_grad(set_to_none=True)

    total_loss_logged   = 0.0
    total_kl_logged     = 0.0
    total_surrogate     = 0.0
    total_response_toks = 0
    n_contributed       = 0
    rho_min, rho_max    = float("inf"), float("-inf")

    for i, c in enumerate(completions):
        if not c.get("loss_eligible", False):
            continue
        A = advantages[i].item()    # scalar — same gate for every response token

        # Recompute logp_policy WITH GRAD on the same full_ids.
        logp_policy, mask2 = _per_token_logp(
            policy, c["full_ids"], L_prompt, cfg.use_amp,
        )
        # Importance ratio. At step 0 (first update), logp_policy == logp_old
        # numerically so rho == 1; gradient still flows through logp_policy.
        diff = logp_policy - c["logp_old"]
        rho  = torch.exp(diff)

        # Clipped surrogate. The clip caps the ratio's magnitude so a
        # huge ratio from a low-prob action can't dominate the update.
        eps = cfg.clip_eps
        surr_unclipped = rho * A
        surr_clipped   = torch.clamp(rho, 1 - eps, 1 + eps) * A
        surrogate = torch.minimum(surr_unclipped, surr_clipped)

        # KL(policy || reference), k3 unbiased estimator (Schulman 2020 blog).
        # Estimator: r - 1 - log(r), where r = pi_ref / pi_policy.
        # We use log r = logp_ref - logp_policy → kl = exp(log_r) - log_r - 1
        log_r = c["logp_ref"] - logp_policy
        kl_k3 = torch.exp(log_r) - log_r - 1.0

        obj = surrogate - cfg.kl_beta * kl_k3   # per-token objective

        # Mean over response tokens, average over G completions (grad accum).
        mask_f = c["mask"].float()
        n_resp = mask_f.sum()
        if n_resp.item() == 0:
            continue
        loss_i = -(obj * mask_f).sum() / n_resp / G
        loss_i.backward()

        # Logging stats (detached scalars, do NOT pull into the graph)
        with torch.no_grad():
            total_loss_logged += loss_i.item() * G
            kl_pt = (kl_k3.detach() * mask_f).sum().item() / n_resp.item()
            su_pt = (surrogate.detach() * mask_f).sum().item() / n_resp.item()
            rho_resp = rho.detach()[c["mask"]]
            rho_min = min(rho_min, rho_resp.min().item())
            rho_max = max(rho_max, rho_resp.max().item())
            total_kl_logged += kl_pt
            total_surrogate += su_pt
            total_response_toks += int(n_resp.item())
            n_contributed += 1

    # ── 8. Optimizer step (unless deferring for grad accum) ─────────────────
    grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
    if do_optimizer_step:
        optimizer.step()

    return {
        "skipped":           False,
        "mean_reward":       r_mean,
        "std_reward":        r_std,
        "advantage_mag":     advantages.abs().mean().item(),
        "loss":              total_loss_logged / max(n_contributed, 1),
        "kl":                total_kl_logged / max(n_contributed, 1),
        "surrogate":         total_surrogate / max(n_contributed, 1),
        "rho_min":           rho_min if rho_min != float("inf") else 1.0,
        "rho_max":           rho_max if rho_max != float("-inf") else 1.0,
        "grad_norm":         float(grad_norm.item()),
        "parse_rate":        sum(1 for c in completions if c["code"]) / G,
        "pos_reward_rate":   sum(1 for r in rewards_list if r > 0) / G,
        "n_response_tokens": total_response_toks,
        "n_contributed":     n_contributed,
        "n_tests_used":      n_tests_used,
        "test_source":       source,
    }


# ===========================================================================
# 4. Driver
# ===========================================================================
def _make_optimizer(policy: nn.Module, cfg: GRPOConfig) -> torch.optim.Optimizer:
    """Same selective-decay recipe as SFT (norms/biases/embeddings exempt)."""
    decay, no_decay = [], []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    fused_ok = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    extra = {"fused": True} if fused_ok and torch.cuda.is_available() else {}
    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8, **extra,
    )


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def _filter_and_split_curriculum(
    rows: list[dict],
    std_threshold: float,
    holdout_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Filter by group_std >= threshold, then deterministically split.

    The split is RNG-determined by `seed`; resume re-derives the same
    split as long as curriculum.jsonl is unchanged. That gives us a
    fixed eval set across the whole run without persisting indices.
    """
    filtered = [r for r in rows if r.get("group_std", 0.0) >= std_threshold]
    if len(filtered) < holdout_size + 10:
        raise SystemExit(
            f"After filtering by std >= {std_threshold} only "
            f"{len(filtered)} problems remain; need at least "
            f"{holdout_size + 10} to have a meaningful training pool."
        )
    rng = random.Random(seed)
    rng.shuffle(filtered)
    holdout  = filtered[:holdout_size]
    training = filtered[holdout_size:]
    return training, holdout


def _make_sampling_weights(pool: list[dict], cap: int) -> list[int]:
    """weight = min(n_tests_used, cap) — robust-signal problems sampled
    more often than 1-test spike problems. Falls back to 1 for any
    record missing n_tests_used."""
    return [max(1, min(int(c.get("n_tests_used", 1)), cap)) for c in pool]


@torch.no_grad()
def _evaluate_holdout(
    policy: nn.Module,
    tokenizer: Tokenizer,
    holdout_rows: list[dict],
    ds,
    cfg: GRPOConfig,
    device: torch.device,
) -> tuple[float, list[dict]]:
    """Score the current policy on the held-out set.

    For each held-out problem:
      - generate eval_g completions at the same temperature/top_k as training
      - score against ALL available tests (no subsampling) with per-completion
        wall-clock budget eval_completion_budget_s
      - reward = (n_pass / n_total_tests)

    Returns (mean across problems, per-problem stats list).
    """
    policy.eval()
    per_problem_means: list[float] = []
    per_problem_stats: list[dict] = []

    for crow in holdout_rows:
        row_idx = int(crow["row_index"])
        problem_row = ds[row_idx]
        tests, source = _pick_tests(problem_row)
        if source == "none" or not tests.get("input"):
            continue

        # Cap tests per held-out problem. Sampling is deterministic per
        # (seed, row_index) so the same problem uses the same tests at
        # every eval pass — the trend across evals is an apples-to-apples
        # comparison even with subsampling.
        inputs  = tests["input"]
        outputs = tests["output"]
        n_total_tests = len(inputs)
        if n_total_tests > cfg.eval_max_tests:
            sub_rng = random.Random(cfg.seed * 1_000_003 + row_idx)
            chosen = sub_rng.sample(range(n_total_tests), cfg.eval_max_tests)
            tests = {"input":  [inputs[i]  for i in chosen],
                     "output": [outputs[i] for i in chosen]}

        prompt_ids = _build_prompt_ids(problem_row, tokenizer)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

        rewards: list[float] = []
        for g in range(cfg.eval_g):
            out = policy.generate(
                prompt_tensor.unsqueeze(0),
                max_new_tokens=cfg.max_new_tokens,
                eos_id=EOS_IDX,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
            )
            completion_ids = out[0][len(prompt_ids):]
            completion_text = tokenizer.decode(completion_ids.tolist())
            code = _extract_code_block(completion_text)
            reward, _ = _reward_with_budget(
                code, tests,
                timeout_s=cfg.test_timeout_s,
                budget_s=cfg.eval_completion_budget_s,
            )
            rewards.append(reward)

        if not rewards:
            continue
        problem_mean = statistics.fmean(rewards)
        per_problem_means.append(problem_mean)
        per_problem_stats.append({
            "row_index":   row_idx,
            "title":       crow.get("problem_title", "<no title>"),
            "n_tests":     len(tests.get("input", [])),
            "mean_reward": problem_mean,
            "max_reward":  max(rewards),
            "n_perfect":   sum(1 for r in rewards if r >= 0.99),
        })

    if not per_problem_means:
        return 0.0, []
    return statistics.fmean(per_problem_means), per_problem_stats


# ---------------------------------------------------------------------------
# Full-state checkpoint for calibration resume
# ---------------------------------------------------------------------------
def _save_full_checkpoint(
    path: Path,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_holdout: float,
    last_holdout: float,
    reward_history: list[float],
    kl_history: list[float],
    loss_history: list[float],
    holdout_history: list[tuple],
    model_cfg,
    rng: random.Random,
    device: torch.device,
) -> None:
    """Self-contained checkpoint — model + optimizer + step + RNG + history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state":            policy.state_dict(),
        "optim_state":            optimizer.state_dict(),
        "step":                   step,
        "best_holdout_reward":    best_holdout,
        "last_holdout_reward":    last_holdout,
        "reward_history":         reward_history,
        "kl_history":             kl_history,
        "loss_history":           loss_history,
        "holdout_history":        holdout_history,
        "model_cfg":              asdict(model_cfg) if hasattr(model_cfg, "__dataclass_fields__") else dict(vars(model_cfg)),
        "torch_rng_state":        torch.get_rng_state(),
        "torch_cuda_rng_state":   torch.cuda.get_rng_state() if device.type == "cuda" else None,
        "python_rng_state":       rng.getstate(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_full_checkpoint(
    path: Path,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    device: torch.device,
) -> dict:
    """Restore everything saved by _save_full_checkpoint. Returns the payload
    so the caller can read scalar fields (step, best_holdout, etc)."""
    print(f"Resuming from {path} ...")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["model_state"])
    # Move policy back to device (load_state_dict respects parameter device,
    # but new state_dict tensors are on cpu).
    policy.to(device)
    optimizer.load_state_dict(payload["optim_state"])
    rng.setstate(tuple(payload["python_rng_state"]) if isinstance(payload["python_rng_state"], list) else payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if device.type == "cuda" and payload.get("torch_cuda_rng_state") is not None:
        try:
            torch.cuda.set_rng_state(payload["torch_cuda_rng_state"])
        except Exception as e:
            print(f"  ! couldn't restore CUDA RNG: {e}")
    print(f"  resumed at step {payload['step']}, "
          f"best_holdout={payload['best_holdout_reward']:.3f}")
    return payload


# ---------------------------------------------------------------------------
# (smoke-mode legacy save — used by non-calibration runs)
# ---------------------------------------------------------------------------
def _save_checkpoint(path: Path, policy: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     step: int, reward_ma: float, model_cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": policy.state_dict(),
        "optim_state": optimizer.state_dict(),
        "model_cfg":   asdict(model_cfg) if hasattr(model_cfg, "__dataclass_fields__") else dict(vars(model_cfg)),
        "step":        step,
        "reward_ma":   reward_ma,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(cfg: GRPOConfig, smoke: bool = False,
          calibration: bool = False, resume: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(cfg.seed)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    meta_path = Path(cfg.meta).resolve()
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text())
    tok_path = Path(meta["tokenizer_path"])
    if not tok_path.is_absolute():
        tok_path = (meta_path.parent.parent / tok_path).resolve()
    tokenizer = Tokenizer.from_file(str(tok_path))

    # ── Policy + reference ──────────────────────────────────────────────────
    ckpt = Path(cfg.sft_ckpt).resolve()
    if not ckpt.exists():
        raise SystemExit(f"SFT checkpoint not found at {ckpt}")
    print(f"Loading policy (trainable) from {ckpt}")
    policy, model_cfg = load_pretrained(
        ckpt, gradient_checkpointing=cfg.gradient_checkpointing, device=device,
    )
    print(f"Loading reference (frozen) from {ckpt}")
    reference, _ = load_pretrained(
        ckpt, gradient_checkpointing=False, device=device,
    )
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    n_policy_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  policy trainable params:    {n_policy_params/1e6:.1f}M")
    print(f"  reference frozen params:    "
          f"{sum(p.numel() for p in reference.parameters())/1e6:.1f}M")

    optimizer = _make_optimizer(policy, cfg)

    # ── Curriculum ─────────────────────────────────────────────────────────
    curriculum_path = Path(cfg.curriculum).resolve()
    curriculum = _load_curriculum(curriculum_path, solvable_only=True)
    print(f"\nCurriculum: {len(curriculum)} SOLVABLE problems "
          f"(loaded from {curriculum_path.name})")
    if not curriculum:
        raise SystemExit("No SOLVABLE problems in curriculum. Run build_curriculum.py first.")

    # Calibration: filter by std + split into training / held-out.
    holdout_pool: list[dict] = []
    sampling_weights: list[int] | None = None
    if calibration:
        training_pool, holdout_pool = _filter_and_split_curriculum(
            curriculum, cfg.pool_std_threshold, cfg.holdout_size, cfg.seed,
        )
        if cfg.weighted_sampling:
            sampling_weights = _make_sampling_weights(training_pool, cfg.weight_test_cap)
        print(f"  filtered by group_std >= {cfg.pool_std_threshold}: "
              f"{len(training_pool) + len(holdout_pool)} kept")
        print(f"  training pool: {len(training_pool)} problems  "
              f"({'weighted' if sampling_weights else 'uniform'} sampling)")
        print(f"  held-out set: {len(holdout_pool)} problems "
              f"(rows: {[int(r['row_index']) for r in holdout_pool[:5]]}...)")
        # Distribution of n_tests_used in training pool — useful sanity check
        tnt = sorted(int(c.get("n_tests_used", 0)) for c in training_pool)
        print(f"  training pool n_tests distribution: "
              f"min={tnt[0]} p50={tnt[len(tnt)//2]} max={tnt[-1]}")
    else:
        training_pool = curriculum

    # ── Dataset (for fetching problem rows by row_index) ────────────────────
    load_dotenv(_ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN")
    print("Loading open-r1/codeforces-cots ...")
    ds = load_dataset(
        "open-r1/codeforces-cots",
        "solutions_py_decontaminated",
        split="train", token=hf_token,
    )
    print(f"  rows: {len(ds):,}")

    # ── Output dir ─────────────────────────────────────────────────────────
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    print(f"\nGRPO config: G={cfg.G}, lr={cfg.lr:g}, kl_beta={cfg.kl_beta:g}, "
          f"clip_eps={cfg.clip_eps}, total_steps={cfg.total_steps}, "
          f"max_tests={cfg.max_tests}")
    print(f"Output dir: {out_dir}/")
    if smoke:
        print("Mode: SMOKE TEST (mechanics validation)")
    print()

    # ── Training loop state ────────────────────────────────────────────────
    rng = random.Random(cfg.seed)
    step = 0
    skipped = 0
    reward_history:        list[float]                = []
    kl_history:            list[float]                = []
    loss_history:          list[float]                = []
    holdout_history:       list[tuple[int, float]]    = []
    rho_extremes:          list[tuple[float, float]]  = []
    peak_gb_overall        = 0.0
    best_holdout_reward    = -float("inf")
    last_holdout_reward    = float("nan")
    best_reward_ma         = -float("inf")     # legacy path for non-calibration

    # ── Resume from last.pt if requested ──────────────────────────────────
    last_ckpt = out_dir / "last.pt"
    if resume and last_ckpt.exists():
        payload = _load_full_checkpoint(last_ckpt, policy, optimizer, rng, device)
        step                 = int(payload.get("step", 0))
        best_holdout_reward  = float(payload.get("best_holdout_reward", -float("inf")))
        last_holdout_reward  = float(payload.get("last_holdout_reward", float("nan")))
        reward_history       = list(payload.get("reward_history", []))
        kl_history           = list(payload.get("kl_history", []))
        loss_history         = list(payload.get("loss_history", []))
        # holdout_history items are saved as lists from JSON; coerce back to tuples
        holdout_history = [
            (int(s), float(r)) for s, r in payload.get("holdout_history", [])
        ]
    elif resume:
        print(f"--resume requested but {last_ckpt} not found; starting fresh.")

    # ── Loop ──────────────────────────────────────────────────────────────
    while step < cfg.total_steps:
        # Sample a SOLVABLE problem from the training pool.
        if sampling_weights is not None:
            crow = rng.choices(training_pool, weights=sampling_weights, k=1)[0]
        else:
            crow = rng.choice(training_pool)

        problem_row = ds[int(crow["row_index"])]
        # Hint goes into the test-subsample seed so the same problem at
        # different steps uses *different* test subsets (more signal).
        problem_row_with_hint = dict(problem_row)
        problem_row_with_hint["row_index_hint"] = int(crow["row_index"])

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        metrics = grpo_step(
            policy, reference, tokenizer, problem_row_with_hint,
            cfg, device, optimizer, step + 1,
            do_optimizer_step=True,
        )
        elapsed = time.time() - t0

        peak_gb = 0.0
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_gb_overall = max(peak_gb_overall, peak_gb)

        # If dead-signal, log and try another (don't burn a step).
        if metrics["skipped"]:
            skipped += 1
            print(f"[skip {step+1+skipped:3d}] row={int(crow['row_index']):5d} "
                  f"r̄={metrics['mean_reward']:.3f} "
                  f"parse={metrics['parse_rate']:.2f}  "
                  f"GPU={peak_gb:.2f}G  ({elapsed:.1f}s)  -- std=0, no advantage")
            continue

        step += 1
        reward_history.append(metrics["mean_reward"])
        kl_history.append(metrics["kl"])
        loss_history.append(metrics["loss"])
        rho_extremes.append((metrics["rho_min"], metrics["rho_max"]))

        if step % cfg.log_interval == 0:
            print(
                f"[step {step:3d}] row={int(crow['row_index']):5d} "
                f"r̄={metrics['mean_reward']:.3f} σr={metrics['std_reward']:.3f} "
                f"|A|={metrics['advantage_mag']:.2f}  "
                f"L={metrics['loss']:+.4f} KL={metrics['kl']:.4f} "
                f"ρ∈[{metrics['rho_min']:.3f},{metrics['rho_max']:.3f}] "
                f"|g|={metrics['grad_norm']:.2f}  "
                f"parse={metrics['parse_rate']:.2f} pos={metrics['pos_reward_rate']:.2f}  "
                f"GPU={peak_gb:.2f}G ({elapsed:.1f}s)"
            )
            with log_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps({
                    "step": step, "row_index": int(crow["row_index"]),
                    "elapsed_s": elapsed, "peak_gb": peak_gb, **metrics,
                }) + "\n")

        # ── Calibration path: eval + best-by-holdout + full checkpoint ────
        if calibration and step % cfg.eval_interval == 0:
            print(f"\n  >> step {step}: running held-out eval over "
                  f"{len(holdout_pool)} problems (G={cfg.eval_g}, all tests) ...")
            t_eval = time.time()
            holdout_mean, per_problem = _evaluate_holdout(
                policy, tokenizer, holdout_pool, ds, cfg, device,
            )
            eval_elapsed = time.time() - t_eval
            policy.train()   # _evaluate_holdout sets eval mode

            last_holdout_reward = holdout_mean
            holdout_history.append((step, holdout_mean))

            # Compact running summary line
            recent_n = min(cfg.eval_interval, len(reward_history))
            recent_train_r = statistics.fmean(reward_history[-recent_n:])
            recent_train_kl = statistics.fmean(kl_history[-recent_n:])
            print(f"  >> step {step}: train r̄(last {recent_n})={recent_train_r:.3f} "
                  f"KL(last {recent_n})={recent_train_kl:.4f}  "
                  f"HELD-OUT r̄={holdout_mean:.3f}  "
                  f"({eval_elapsed:.0f}s, "
                  f"{sum(p['n_perfect'] for p in per_problem)}/{cfg.eval_g*len(per_problem)} perfect rollouts)")

            with log_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps({
                    "step": step, "kind": "holdout_eval",
                    "holdout_mean": holdout_mean,
                    "eval_elapsed_s": eval_elapsed,
                    "recent_train_mean": recent_train_r,
                    "per_problem": per_problem,
                }) + "\n")

            # Always save last.pt at eval boundaries (= checkpoint_interval too)
            _save_full_checkpoint(
                out_dir / "last.pt", policy, optimizer, step,
                best_holdout_reward, last_holdout_reward,
                reward_history, kl_history, loss_history, holdout_history,
                model_cfg, rng, device,
            )

            if holdout_mean > best_holdout_reward:
                best_holdout_reward = holdout_mean
                _save_full_checkpoint(
                    out_dir / "best.pt", policy, optimizer, step,
                    best_holdout_reward, last_holdout_reward,
                    reward_history, kl_history, loss_history, holdout_history,
                    model_cfg, rng, device,
                )
                print(f"  >> NEW BEST held-out r̄={holdout_mean:.3f}  → saved best.pt")
            else:
                print(f"  >> no improvement (best so far {best_holdout_reward:.3f})")
            print()

        # ── Non-calibration path (e.g. smoke): legacy reward-MA best ──────
        elif not calibration:
            win = min(cfg.reward_ma_window, len(reward_history))
            reward_ma = statistics.fmean(reward_history[-win:])
            if step % cfg.checkpoint_interval == 0:
                _save_checkpoint(out_dir / "last.pt", policy, optimizer,
                                  step, reward_ma, model_cfg)
                print(f"  >> saved last.pt at step {step}")
            if reward_ma > best_reward_ma:
                best_reward_ma = reward_ma
                _save_checkpoint(out_dir / "best.pt", policy, optimizer,
                                  step, reward_ma, model_cfg)

    # ── End-of-run summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"DONE: {step} optimizer steps, {skipped} dead-signal skips")
    print(f"{'=' * 70}")

    if reward_history:
        # Trend: first quarter vs last quarter
        q = max(1, len(reward_history) // 4)
        r_init  = statistics.fmean(reward_history[:q])
        r_final = statistics.fmean(reward_history[-q:])
        k_init  = statistics.fmean(kl_history[:q])
        k_final = statistics.fmean(kl_history[-q:])
        l_init  = statistics.fmean(loss_history[:q])
        l_final = statistics.fmean(loss_history[-q:])

        # Did any loss go NaN/Inf?
        any_nan = any(math.isnan(x) or math.isinf(x) for x in loss_history)

        print(f"\n  reward trend:  {r_init:.3f}  →  {r_final:.3f}    "
              f"({'↑' if r_final > r_init else ('↓' if r_final < r_init else '→')})")
        print(f"  KL trend:      {k_init:.4f}  →  {k_final:.4f}   "
              f"({'↑' if k_final > k_init else ('↓' if k_final < k_init else '→')})")
        print(f"  loss trend:    {l_init:+.4f} →  {l_final:+.4f}")
        print(f"  loss finite:   {'✓' if not any_nan else '✗ FAILED (NaN/Inf detected)'}")
        print(f"  GPU peak:      {peak_gb_overall:.2f} GB  "
              f"({'✓ under 8 GB' if peak_gb_overall < 8.0 else '✗ OVER 8 GB'})")
        print(f"  reward range:  [{min(reward_history):.3f}, {max(reward_history):.3f}]")
        print(f"  KL range:      [{min(kl_history):.4f}, {max(kl_history):.4f}]")

        if smoke:
            print(f"\n  smoke verdict:")
            print(f"    loss finite:                 "
                  f"{'PASS' if not any_nan else 'FAIL'}")
            print(f"    KL starts ~0 (< 0.01):       "
                  f"{'PASS' if kl_history[0] < 0.01 else 'CHECK (' + f'{kl_history[0]:.4f}' + ')'}")
            print(f"    KL grows over run:           "
                  f"{'PASS' if k_final > k_init else 'CHECK (did not grow)'}")
            print(f"    reward not collapsed to 0:   "
                  f"{'PASS' if r_final > 0 else 'CHECK (= 0 at end)'}")
            print(f"    GPU under 8 GB:              "
                  f"{'PASS' if peak_gb_overall < 8.0 else 'FAIL'}")

        if calibration and holdout_history:
            print(f"\n  HELD-OUT TRAJECTORY (every {cfg.eval_interval} steps):")
            print(f"    step | holdout r̄ | Δ from prev")
            prev = None
            for s, r in holdout_history:
                delta = f"{r - prev:+.3f}" if prev is not None else "  --  "
                print(f"    {s:>4} | {r:>9.3f} | {delta}")
                prev = r
            init_h = holdout_history[0][1]
            final_h = holdout_history[-1][1]
            print(f"    initial → final: {init_h:.3f} → {final_h:.3f}  "
                  f"({'↑' if final_h > init_h else ('↓' if final_h < init_h else '→')})  "
                  f"best: {best_holdout_reward:.3f}")


# ===========================================================================
# 5. Entry point
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training / smoke test.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 20 steps on SOLVABLE-only curriculum to "
                             "validate mechanics (loss finite, KL grows, etc.).")
    parser.add_argument("--calibration", action="store_true",
                        help="Calibration run: filter curriculum by std, "
                             "weighted sampling, held-out eval every N steps, "
                             "best.pt tracked by held-out reward.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints_grpo_v1/last.pt "
                             "(requires --calibration).")
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--holdout-size", type=int, default=None)
    parser.add_argument("--pool-std-threshold", type=float, default=None)
    parser.add_argument("--eval-completion-budget-s", type=float, default=None,
                        help="Per-completion wall-clock budget for held-out eval. "
                             "Lower = faster eval, less generous to long-running code.")
    parser.add_argument("--eval-max-tests", type=int, default=None,
                        help="Cap tests-per-problem during held-out eval. "
                             "Subsampling is deterministic per row_index — the "
                             "trend stays comparable across eval passes.")
    parser.add_argument("--curriculum", type=Path, default=None)
    parser.add_argument("--sft-ckpt",   type=Path, default=None)
    parser.add_argument("--meta",       type=Path, default=None)
    parser.add_argument("--out-dir",    type=Path, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--G", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--kl-beta", type=float, default=None)
    parser.add_argument("--clip-eps", type=float, default=None)
    parser.add_argument("--max-tests", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = GRPOConfig()

    # Smoke preset: 20 steps, frequent checkpointing off, every step logged.
    if args.smoke:
        cfg.total_steps         = 20
        cfg.log_interval        = 1
        cfg.checkpoint_interval = 10_000   # don't bother during smoke

    # Calibration preset: 400 steps, eval every 25, held-out best-tracking.
    # These are the user-spec defaults for the first real calibration run.
    if args.calibration:
        cfg.total_steps         = 400
        cfg.eval_interval       = 25
        cfg.checkpoint_interval = 25       # paired with eval
        cfg.log_interval        = 1
        cfg.pool_std_threshold  = 0.04
        cfg.holdout_size        = 15
        cfg.weighted_sampling   = True
        cfg.weight_test_cap     = 15
        cfg.max_tests           = 15       # quality > speed for calibration
        cfg.eval_g              = 8
        # All other GRPO knobs (lr, kl_beta, clip_eps, G, temperature) stay
        # at smoke-test defaults so we change one thing at a time.

    # CLI overrides (after smoke preset so user can still override).
    if args.curriculum is not None: cfg.curriculum = str(args.curriculum)
    if args.sft_ckpt is not None:   cfg.sft_ckpt = str(args.sft_ckpt)
    if args.meta is not None:       cfg.meta = str(args.meta)
    if args.out_dir is not None:    cfg.out_dir = str(args.out_dir)
    if args.total_steps is not None: cfg.total_steps = args.total_steps
    if args.G is not None:          cfg.G = args.G
    if args.lr is not None:         cfg.lr = args.lr
    if args.kl_beta is not None:    cfg.kl_beta = args.kl_beta
    if args.clip_eps is not None:   cfg.clip_eps = args.clip_eps
    if args.max_tests is not None:  cfg.max_tests = args.max_tests
    if args.max_new_tokens is not None: cfg.max_new_tokens = args.max_new_tokens
    if args.temperature is not None: cfg.temperature = args.temperature
    if args.gradient_checkpointing: cfg.gradient_checkpointing = True
    if args.seed is not None:       cfg.seed = args.seed
    if args.eval_interval is not None:      cfg.eval_interval = args.eval_interval
    if args.holdout_size is not None:       cfg.holdout_size = args.holdout_size
    if args.pool_std_threshold is not None: cfg.pool_std_threshold = args.pool_std_threshold
    if args.eval_completion_budget_s is not None: cfg.eval_completion_budget_s = args.eval_completion_budget_s
    if args.eval_max_tests is not None:     cfg.eval_max_tests = args.eval_max_tests

    if args.resume and not args.calibration:
        print("--resume only meaningful with --calibration; ignoring.")

    train(cfg, smoke=args.smoke, calibration=args.calibration,
          resume=args.resume and args.calibration)


if __name__ == "__main__":
    main()
