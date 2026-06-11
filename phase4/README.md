# Phase 4 — GRPO Reinforcement Learning

Takes the SFT checkpoint from [Phase 3](../phase3/) and applies **GRPO**
(Group Relative Policy Optimization) with a **verifiable, code-execution
reward**: generate multiple solution attempts per problem, run them against
test cases, reward by how many tests pass, and push the policy toward
correctness.

References:
- Shao et al. 2024 — [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI 2025 — [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- Schulman 2020 — [Approximating KL divergence](http://joschu.net/blog/kl-approx.html) (the k3 estimator)
- Lozhkov et al. 2025 — [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots)

> **The headline result of this phase is a negative one, and it's the most
> instructive finding in the whole project: the first GRPO run's apparent
> 10× reward improvement was almost entirely _reward hacking_ — the model
> learned to print constant outputs (`-1`, `0`) on problems whose test set
> is dominated by one answer. Building a hack-resistant reward revealed
> that a 97M model has essentially no genuine RL headroom on this data.
> See [§The reward-hacking story](#the-reward-hacking-story).**

---

## Files

| File | Role |
|------|------|
| [audit_tests.py](audit_tests.py) | Audit test-case coverage of the dataset; size the GRPO problem pool. |
| [code_executor.py](code_executor.py) | Sandbox runner (subprocess + rlimit + timeout). `run_solution`, `reward_fraction`, and the anti-hack `constant_baseline` + `reward_advantage`. |
| [rollouts.py](rollouts.py) | `generate_group` — sample G completions per problem and score each. The unit GRPO normalises over. |
| [build_curriculum.py](build_curriculum.py) | Score every pool problem with the SFT model to find the learnable subset (non-zero reward variance). Writes `curriculum.jsonl` / `curriculum_v2.jsonl`. |
| [grpo.py](grpo.py) | The GRPO training loop: two-model setup, clipped surrogate + KL loss, held-out eval, resume-safe checkpoints. |
| [compare_grpo.py](compare_grpo.py) | Qualitative pre-vs-post side-by-side on held-out problems — read the actual code, not just the numbers. |
| `curriculum.jsonl` | Pool scored under the raw-fraction reward (200 SOLVABLE @ 25.4%). |
| `curriculum_v2.jsonl` | Pool re-scored under the anti-hack reward (24 SOLVABLE @ 1.8%). |
| `checkpoints_grpo_v1/` | First GRPO run (raw-fraction reward). **Has the hack baked in** — kept as a cautionary artifact. |
| `checkpoints_grpo_v2/` | Re-calibration under the anti-hack reward. |

---

## Why GRPO (not PPO)

PPO needs **three** resident networks: policy, reference, and a value/critic
network that estimates per-token baselines. GRPO drops the critic entirely
and replaces the value baseline with a **group-relative mean**: sample G
completions for the same prompt, and each completion's advantage is its
reward minus the group mean, normalised by the group std.

```
A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + ε)
```

For an 8 GB GPU this is the decisive win — **two 97M models fit (peak ~3 GB),
three would not.** The cost is that GRPO needs a group of G rollouts per
update (more generation), but generation is cheap relative to a third
resident network's memory.

---

## The pipeline, in order

```
audit_tests.py        → how many problems have usable tests? (the pool)
        │
code_executor.py      → run candidate code, score pass-fraction (the reward)
        │
rollouts.py           → G completions per problem, each scored (the group)
        │
build_curriculum.py   → which problems have reward VARIANCE? (the learnable set)
        │
grpo.py               → train the policy on the learnable set (the RL loop)
        │
compare_grpo.py       → did the code actually get better? (the eyeball)
```

---

## [audit_tests.py](audit_tests.py) — sizing the problem pool

GRPO needs problems with **executable test cases**. This scans all 8,133
rows of `open-r1/codeforces-cots:solutions_py_decontaminated` and reports:

- Coverage: how many rows have non-empty `public_tests` / `private_tests`.
- The exact shape of a test entry — verified to be
  `{"input": [stdin_str, ...], "output": [stdout_str, ...]}` (parallel lists).
- The **GRPO problem pool**: rows with `description + input_format +
  output_format + at least one test source`. This does NOT require an
  `editorial` (unlike the SFT prep), so it's a different, larger set than
  the 3,704 SFT examples.

The pool is the universe `build_curriculum.py` then filters down to the
*learnable* subset.

---

## [code_executor.py](code_executor.py) — the reward

### Sandboxing

Runs model-generated code as a subprocess with layered containment
(documented in the module docstring): isolated interpreter (`python -I -S`),
wall-clock timeout, and `resource.setrlimit` caps on CPU time, address
space, file size, and process count. **This is NOT a true sandbox** — the
code can still touch the filesystem and network. It's the pragmatic local
version chosen because each GRPO rollout has to be cheap (no container
startup per test case). For a hardened setup: container / nsjail / E2B.

### Output comparison (CodeForces-style)

`_to_canonical_lines` normalises CRLF→LF, **rstrips every line** (CodeForces
test files have trailing whitespace before newlines — observed `"3 3 3 \n"`),
and drops trailing empty lines. 10 unit asserts in `__main__` pin this down.

### The two rewards

`reward_fraction(code, tests)` — fraction of tests that pass. **This was the
original reward, and it's hackable** (see below).

`reward_advantage(code, tests)` — the anti-hack reward:
```
advantage = max(0, pass_fraction - constant_baseline)
```
where `constant_baseline(tests)` is the fraction a best-case constant-output
program would pass = (count of the modal expected output) / n_tests.
Plus a **secondary guard**: any completion that produces one identical
output across all distinct test inputs is zeroed regardless. 6 unit asserts
verify: a constant scores ~0, a genuinely-partial program scores >0, a full
solution scores highest.

---

## [rollouts.py](rollouts.py) — the group

`generate_group(model, tokenizer, problem_row, G=8, ...)`:

1. Build the prompt with the **same prefix as SFT training** (problem +
   opening `<reasoning>` tag) — imported from `phase3/finetune_data_prep`
   so train-time and rollout-time prompts can't drift.
2. Sample G independent completions at temperature 0.9.
3. Extract the `<code>...</code>` block from each (no block → reward 0).
4. Score each via `reward_fraction` or `reward_advantage` (`reward_mode`).

Returns G dicts `{completion_ids, completion_text, code, reward,
reward_fraction, guard_fired}` — everything the GRPO loss and the logging
need.

The smoke `__main__` answers the prerequisite question: *from this model,
can we roll out G completions and get reward **variance**?* If every
completion scores identically, the group-relative advantage is zero and
GRPO can't learn from that problem.

---

## [build_curriculum.py](build_curriculum.py) — finding the learnable subset

The rollout diagnostic showed **~2/3 of random problems are dead-signal**
(all G completions score 0) for the 97M model. Optimising those wastes
compute — the advantage is exactly zero. So before training, we score the
whole pool once and keep only problems with reward **variance** (`group_std
> 0`).

For each problem: `generate_group` (G=8, temp 0.9), reward each completion
(capped at MAX_TESTS=15 for cost), record group mean / std / max / parseable
rate. Classify DEAD (std=0) vs SOLVABLE (std>0). Output sorted by group_std
descending (most learnable first).

Engineering for a multi-hour pass: `--limit` for testing, tqdm ETA,
checkpoint the JSONL every 100 problems (resume by skipping scored
row_indices), `--solvable-target` early-exit.

**Two curriculum files exist, and the difference between them is the whole
reward-hacking story:**

| | `curriculum.jsonl` | `curriculum_v2.jsonl` |
|---|---|---|
| Reward | raw `fraction` | `advantage` (anti-hack) |
| Scored | 787 | 1313 |
| SOLVABLE | 200 (**25.4%**) | 24 (**1.8%**) |

The hit rate collapsed 14× under the honest reward. **Almost all the
"learnable spread" under the raw reward was hack-vs-crash variance, not
genuine partial-solving.**

---

## [grpo.py](grpo.py) — the training loop

### Two-model setup

- **policy**: `GPTv3` from `checkpoints_finetune_v2/best.pt`, trainable.
- **reference**: a second frozen `GPTv3` from the same checkpoint, eval
  mode, `requires_grad=False`, used only for the KL term.

### Per step

1. Sample a SOLVABLE problem (weighted toward more-test problems so the
   robust signal dominates the 1-test "spike" problems).
2. Roll out G completions with the policy.
3. Reward each (advantage reward, per-completion wall-clock budget).
4. Group-relative advantage `A_i = (r_i - mean) / (std + 1e-4)`. If
   `std == 0`, **skip** — no gradient signal.
5. Loss (per response token, then masked-mean over the group):

```
rho       = exp(logp_policy - logp_old)              # importance ratio
surrogate = min(rho·A_i, clip(rho, 1-ε, 1+ε)·A_i)    # PPO-style clip, ε=0.2
kl_k3     = exp(logp_ref - logp_policy) - (logp_ref - logp_policy) - 1
obj       = surrogate - β·kl_k3                       # β=0.04
loss      = -mean_over_response(obj),  averaged over G
```

The **three-log-prob bookkeeping** is the part to get exactly right:
`logp_policy` carries gradient; `logp_old` (the policy's log-prob at sample
time) and `logp_ref` (the frozen model) are both **detached**.

### Loss masking

Only **response tokens** contribute — the prompt is masked with the same
shift-by-one logic as the SFT loss. The model is graded on what it
*generated*, not on the problem it was *given*.

### Why the displayed loss is ~0

On-policy with one update per rollout, `logp_old == logp_policy`
numerically at sample time, so `rho ≈ 1` and `surrogate ≈ A_i`. Advantages
are zero-mean by construction, so the aggregate loss is ~0. **The signal is
in the per-completion gradient direction, not the scalar loss.** Non-zero
grad norms and growing KL confirm real updates.

### Held-out eval (the honest signal)

15 problems (seed=42 split) are held out and never trained on. Every N
steps the policy is scored on them with **both** `reward_advantage` (the
anti-hack metric) and raw `reward_fraction` (alongside, for visibility).
`best.pt` tracks held-out **advantage** — because training reward can rise
via overfitting/hacking, but held-out advantage rising is real.

### Resume

`--resume` restores model + optimizer + step + all three RNG states +
full metric history from `last.pt` (written every eval). Survives
interruption bit-exactly.

### Modes
- `--smoke` — 20 steps, mechanics validation (loss finite, KL grows, GPU < 8 GB).
- `--calibration` — held-out eval + best-by-holdout + full-state checkpoints.

---

## The reward-hacking story

This is the arc worth studying. It happened in five acts.

### Act 1 — GRPO "works" (v1)

Trained 1500 steps on `curriculum.jsonl` (raw-fraction reward), held-out
reward climbed **0.022 → 0.238** (peak), a clean ~10× improvement. Every
mechanism green: loss finite, KL controlled (0.002→0.058), GPU 3 GB. Looked
like a textbook GRPO success.

### Act 2 — read the actual code (compare_grpo.py)

A pre-vs-post side-by-side on held-out problems showed the POST model
producing things like:

```python
n, m = map(int, input().split())
if n == m:
    print(-1)
else:
    print(-1)
```

It wasn't solving anything. It learned that on problems whose tests are
dominated by one answer (e.g. mostly `-1`), **printing that constant passes
a large fraction of tests.** The 0.238 was mostly this exploit.

### Act 3 — build a hack-resistant reward

`reward_advantage = max(0, pass_fraction − constant_baseline)`, plus a guard
zeroing any completion that emits one identical output across all distinct
inputs. A constant-output program now nets ~0 by construction. Six unit
asserts lock the behaviour.

### Act 4 — re-score the pool, watch the hit rate collapse

Re-scoring under the advantage reward, the SOLVABLE rate fell **25.4% →
1.8%** (24 of 1313). The honest reward revealed that ~24 of every 25
"learnable" problems were only learnable *via the hack*.

### Act 5 — re-calibrate cleanly, get the real answer

200 GRPO steps on the 24-problem clean curriculum. Held-out **advantage
stayed flat at ~0.010** — essentially zero. But the diagnostics show the
mechanism working: the `guard` fires constantly (0.12–0.50 per step), and
where raw fraction spikes (a completion scoring `frac=0.769`), the advantage
is correctly **zeroed**. The model still *reaches* for the constant trick;
the reward refuses to pay for it.

A final v2 side-by-side confirmed it: constant-output completions now carry
`adv=0.00 ⚠GUARD` instead of being rewarded.

### The conclusion

The v1 "10× improvement" was ~96% reward hacking. Under an honest reward, a
**97M model has essentially no genuine RL headroom on this dataset** — only
~1.8% of problems give real learnable signal, and the model can't convert
it into held-out gains. This is a legitimate, well-documented **negative
result**, and arguably more valuable than the inflated v1 number because it
correctly diagnoses *why* the v1 number was misleading.

The GRPO machinery, reward design, curriculum construction, anti-hack guard,
and held-out methodology all work correctly. The limiting factor is **model
scale × dataset difficulty**, not the RL implementation.

---

## Results summary

| Run | Reward | Held-out | Reading |
|-----|--------|----------|---------|
| v1 (1500 steps) | raw fraction | 0.022 → 0.238 | mostly reward hacking |
| v2 (200 steps) | advantage (anti-hack) | 0.010 → 0.010 | hack blocked → ~no real headroom at 97M |

Checkpoints:
- `checkpoints_grpo_v1/best.pt` — step 1150, held-out frac 0.238 (hack baked in)
- `checkpoints_grpo_v2/best.pt` — step 125, held-out adv 0.030 (clean reward)

---

## Running

```bash
cd phase4

# 1. Audit the pool (read-only)
python audit_tests.py

# 2. Sanity-check the executor + reward asserts
python code_executor.py

# 3. Build the learnable curriculum (long; resume-aware)
python build_curriculum.py --reward-mode advantage --out curriculum_v2.jsonl \
    --solvable-target 150 --limit 1500

# 4. Smoke-test the GRPO loop (20 steps, mechanics)
python grpo.py --smoke

# 5. Calibration run (anti-hack reward, held-out eval, fresh from SFT)
python grpo.py --calibration --total-steps 200 --holdout-size 5 \
    --pool-std-threshold 0.0
python grpo.py --calibration --resume ...   # continue from last.pt

# 6. Qualitative pre-vs-post eyeball
python compare_grpo.py            # SFT vs grpo_v2/best.pt on held-out problems
python compare_grpo.py --post checkpoints_grpo_v1/best.pt   # vs the hacked v1
```

All scripts run from `phase4/`. GRPO needs the GPU free; the curriculum
scoring and audit are CPU-bound on the test execution.

---

## Lessons

- **A rising training reward is not evidence of learning** — it can rise
  entirely via reward hacking. You must either read the outputs or use a
  hack-resistant metric (or both). See [[Reward hacking]].
- **Held-out eval under an honest reward is the only trustworthy signal.**
  Train reward and raw-fraction held-out both rose in v1; held-out
  *advantage* did not.
- **Verifiable rewards still have a baseline.** On a test set dominated by
  one answer, a constant scores high. Subtract the constant baseline.
- **Curriculum hit-rate under the clean reward is a model-capability
  thermometer.** 1.8% told us the ceiling before we wasted a long run.

---

## What comes next (Phase 5)

Formal evaluation — pass@1 / pass@k on held-out LeetCode (`greengerong/leetcode`)
+ HumanEval — was not built. The GRPO held-out eval is *a* form of evaluation,
but the standard pass@k benchmark on SFT vs GRPO checkpoints remains open.
