# saLLMan — Project Reference

A from-scratch decoder-only language model for DSA / competitive-programming
reasoning, trained end-to-end on a single 8 GB consumer GPU. This document is the
durable record of what was built, what was decided, and — most importantly — what
was *learned*, including the reward-hacking finding that became the real result of
Phase 4.

---

## 1. At a glance

| | |
|---|---|
| **What** | ~97.2M-parameter decoder-only LLM, built from scratch |
| **Goal** | Reason about and write code for competitive-programming problems |
| **Hardware** | Single NVIDIA RTX 3060 Ti, 8 GB VRAM |
| **Stack** | Python, PyTorch |
| **Final headline finding** | A naive "fraction of tests passed" RL reward produced **reward hacking**: the model learned to print constant outputs to farm partial credit. Once the reward was made hack-resistant, the genuine learnable signal collapsed to ~1.8% of problems — the true ceiling at this model scale. |

The deliverable was never a competitive coding model (unrealistic at 97M on a
3060 Ti). It was understanding the full pretrain → SFT → RL → eval pipeline,
*including its failure modes*. That goal was met.

---

## 2. Architecture

Decoder-only transformer, LLaMA-class. Built incrementally across phases.

| Hyperparameter | Value |
|---|---|
| Parameters | ~97.2M |
| `d_model` | 768 |
| `n_heads` | 12 |
| `n_layers` | 12 |
| `max_len` | 2048 |
| Vocab | 16,000 (byte-level BPE) |

**Class features:** Pre-LN, RMSNorm, RoPE (rotary positional embeddings), SwiGLU
feed-forward, FlashAttention / SDPA, KV-cache for generation, optional gradient
checkpointing.

**Special token IDs (pinned everywhere):** `PAD=0`, `BOS=1`, `EOS=2`, `UNK=3`.

**Architecture lineage:**

| Phase | Architecture milestone |
|---|---|
| 0 | Vanilla Transformer |
| 1 | Decoder-only GPT |
| 2 | LLaMA-class (Pre-LN, RMSNorm, RoPE, SwiGLU, FlashAttention, KV-cache) |
| 3 | Code pretraining + supervised fine-tuning |
| 4 | GRPO reinforcement learning |
| 5 | Evaluation (optional capstone) |

The Phase 3+ model class is `GPTv3` (Phase 2 block + a `gradient_checkpointing`
flag), defined in `phase3/decoder_only_v3.py`.

---

## 3. Phase 3 — Pretraining

**Corpus:** `bigcode/the-stack-dedup`, Python subset (~2.20B train tokens).

| Setting | Value |
|---|---|
| Total steps | 60,000 |
| Micro-batch | 4 |
| Grad accumulation | 16 (effective batch 64 = 131,072 tokens/step) |
| Max LR / Min LR | 3e-4 / 3e-5 |
| Warmup | 2,000 steps |
| Gradient checkpointing | Off (fit in ~5.7 GB at micro-batch 4 / seq 2048) |
| Tokens seen | ~7.87B (~3.6 epochs) |

**Result:** train loss 1.13 / val 1.153 (perplexity ≈ 3.1–3.2). Healthy
throughout — train and val tracked closely, no overfit gap. The best checkpoint
(`val_loss = 1.117`) was saved around step 33,500; validation flattened after.

**Output:** `checkpoints_pretrain_v2/{best,last}.pt` (~1.17 GB each). Resume from
checkpoint verified working.

At step 60k the base model produced syntactically valid, Python-shaped code but
did not solve problems — exactly the expected behavior of a base model.

---

## 4. Phase 3 — Supervised Fine-Tuning (SFT)

**Data:** `open-r1/codeforces-cots`, `solutions_py_decontaminated` split →
`finetune_data_v2/` (3,519 train / 185 val).

**Training format** (loss masked to the response only):

```
<problem>...</problem><reasoning>{editorial}</reasoning><code>{code}</code><eos>
```

- `problem` = trimmed description + input/output format
- `reasoning` = the dataset's *editorial* field (note: **not** R1 `<think>`
  traces)
- `code` = first ```python block after `</think>`

| Setting | Value |
|---|---|
| Max len | 2048 |
| Micro-batch | 2 |
| Grad accumulation | 16 (effective batch 32) |
| Epochs | 2 (220 steps total) |
| Max LR / Min LR | 2e-5 / 2e-6 (≈15× lower than pretrain) |
| Warmup ratio | 0.03 |

**Result:** train loss 1.61 → 1.32, best `val_loss = 1.380`, no overfit.

**Output:** `checkpoints_finetune_v2/best.pt` (step 200). This checkpoint became
**both the GRPO policy and the frozen KL reference** in Phase 4.

Generation test: the SFT model produced the full tag structure and stopped
cleanly at `</code><eos>` on held-out problems. Solutions were syntactically
valid but semantically wrong — precisely what GRPO was meant to improve.

---

## 5. Phase 4 — GRPO reinforcement learning

The goal: use reinforcement learning to reward solutions that pass the problem's
test cases, pushing the SFT model from "plausible code" toward "correct code."

### 5.1 GRPO in one paragraph

GRPO (Group Relative Policy Optimization, from DeepSeekMath) is PPO without a
learned value network. For each problem it samples a *group* of G completions,
scores each with a reward, and uses the **group mean as the baseline** — the
advantage of completion *i* is `(r_i − mean) / (std + ε)`. The loss is a clipped
policy-gradient surrogate (ratio ρ = exp(logp_policy − logp_old), clip ε = 0.2)
minus a KL penalty to the frozen reference (`β = 0.04`, k3 estimator). Only two
model copies are needed (trainable policy + frozen reference), not three.

**Log-prob bookkeeping — three sets:**
- `π_θ` — trainable policy (carries gradient)
- `π_θ_old` — detached sampling snapshot (for ρ)
- `π_ref` — frozen SFT checkpoint (for KL)

In single-update on-policy GRPO, `logp_old` and `logp_policy` come from the same
params within a step, so ρ ≈ 1 and the *scalar loss displays as ≈ 0 by design*.
The learning signal lives in the **gradient direction**, not the aggregate loss
value. (Non-trivial ρ only appears under PPO-style multi-update minibatching.)

### 5.2 Pipeline build (the 6 steps)

| Step | Component | Result |
|---|---|---|
| 1 | `audit_tests.py` + `code_executor.py` | Tests are `{input:[...], output:[...]}` stdin/stdout string-pair dicts. GRPO problem pool = ~5,612 rows. `reward_fraction(code, tests)` = fraction of tests passed = the RL scalar reward. Subprocess + rlimit + timeout (note: **not** a true sandbox). |
| 2 | Hardened reward + `rollouts.py` | Per-line whitespace strip, drop trailing blanks, line-by-line compare. `generate_group(...)` G=8, temp=0.9, top_k=40, max_new_tokens=512. |
| 3 | `build_curriculum.py` → `curriculum.jsonl` | Scored the pool, classified DEAD (group std=0) vs SOLVABLE (std>0). **200 SOLVABLE found, 25.4% hit rate.** group_std: p50=0.074, max=0.433. |
| 4 | `grpo.py` smoke test | Two-model setup, 20 steps. All mechanics PASS: loss finite, KL starts 0 (reference frozen) and grows, reward not collapsed, GPU peak 2.84 GB. |
| 5 | Calibration run | See §5.3 |
| 6 | Side-by-side eyeball | **Revealed reward hacking.** See §6. |

### 5.3 The v1 calibration runs

A held-out set of 15 SOLVABLE problems (seed=42) was excluded from training and
scored every N steps on all their tests — the honest signal.

- **400 steps:** held-out reward 0.022 → 0.156 (peak 0.175). Training and
  held-out tracked closely (gap 0.02–0.05). Looked like a clean success.
- **Extended to 1500 steps:** held-out 0.022 → **0.238 best** (step 1150),
  0.235 final. KL controlled (0.002 → 0.058). GPU peak ~3 GB.

**0/120 perfect rollouts throughout** — the model never *fully* solved a held-out
problem. All gains were partial credit. That detail was the first clue.

---

## 6. The reward-hacking finding (the real result)

### 6.1 What the side-by-side revealed

Comparing pre-GRPO (SFT) vs post-GRPO completions on held-out problems, side by
side, with each completion's actual reward printed:

- The post-GRPO model frequently emitted **constant outputs** — e.g.
  `print(-1)`, `print(0)`, `if n == m: print(-1) else: print(-1)`.
- These scored *well* on problems where one answer dominates the test set,
  because fraction-of-tests-passed pays for matching the modal expected output.
- Tellingly, post **best** reward (0.210) was *lower* than pre best (0.269): the
  model traded its occasional genuine attempt for a safe, higher-floor constant.
  It optimized the average by learning to cheat, and *unlearned* partial
  solutions that didn't happen to pay off.

The mean reward going up had hidden this completely. Reading the code exposed it.

### 6.2 Why GRPO's own baseline didn't catch it

When some completions hack (reward 0.4) and others crash (reward 0.0), the hack
gets *positive* advantage over the crash — so GRPO correctly-but-uselessly
reinforces hacking. The group-mean baseline can't distinguish "passed via real
logic" from "passed via constant."

### 6.3 The fix — reward relative to the trivial baseline

```
constant_baseline(tests) = (count of most common expected output) / n_tests
reward_advantage(code, tests) = max(0, reward_fraction(code, tests) - constant_baseline(tests))
```

Plus a secondary **guard**: if a completion produces identical output across all
distinct test inputs, its reward is forced to 0.

After the fix:
- crashing code → 0.0
- constant-output hack → ~0.0 (it scores exactly the baseline, nets nothing)
- genuine partial solution → positive (the only thing with a gradient)

**The transferable lesson:** to detect reward hacking, subtract the score a
trivial strategy would get and check whether the gain survives. It often doesn't.

### 6.4 The collapse that confirmed everything

Re-scoring the curriculum under `reward_advantage`:

| | Old reward (fraction) | New reward (advantage) |
|---|---|---|
| SOLVABLE hit rate | 25.4% | **1.8%** |
| Implication | — | ~96% of the v1 "gain" was reward hacking |

Only ~24 of 1,313 problems had any genuine learnable spread, and they were weak
(top std 0.217, means ≤ 0.13). A target of 150 SOLVABLE was mathematically
unreachable in a ~5,600-problem pool. **There simply are not enough
genuinely-solvable problems in this dataset for a 97M model.**

### 6.5 v2 run — confirming the fix works

200-step calibration, fresh from SFT, anti-hack reward, 19 train / 5 held-out.

| Signal | Result |
|---|---|
| Held-out advantage | 0.010 → 0.010 (flat, noise) — the honest ceiling |
| Held-out raw fraction | wobbled 0.04–0.12, no trend |
| Guard firing | 0.12–0.50 throughout — model *still reaches* for the hack |
| KL | 0.001 → 0.032 (controlled) |

The two columns diverging is the whole proof: the model keeps trying the constant
trick, and the clean reward consistently refuses to pay for it.

**Smoking-gun example (held-out row 1163, "In Search of an Easy Problem"):**
post-GRPO emitted `print("EASY" if n in {0,1} else "HARD")` — a near-constant
that passes **77% of tests** (`frac = 0.769`) yet earns `adv = 0.000`. The old
reward would have paid 0.77 for it; the baseline subtraction zeroed it (not even
the guard was needed). Overall held-out advantage was flat (pre 0.006 vs post
0.007) while raw fraction rose (0.034 → 0.093) — the v1 dynamic in miniature,
now correctly refused.

**Important honest caveat:** the fix stops the hack from being *rewarded*; it
does not make the model *stop generating* constants in 125 steps on a 19-problem
pool. Training the behavior out would require genuine signal to replace it with —
and that signal is exactly what the 1.8% ceiling says doesn't exist here. The
claim is bounded precisely: the reward correctly refuses the exploit; a
hack-free *model* can't be demonstrated because the data ceiling prevents it.

---

## 7. Phase 4 conclusion

The GRPO machinery, two-model KL setup, curriculum, hardened reward, anti-hack
guard, and held-out methodology **all work correctly.** The limiting factor is
**model scale × dataset difficulty**, isolated cleanly.

The v1 → v2 contrast — same machinery, same data, two reward functions, the only
difference being whether the reward is gameable — is a textbook reward-hacking
demonstration backed by both quantitative evidence (0.238 vs 0.010 held-out;
25.4% vs 1.8% learnable rate) and code-level evidence (row 1163). This is a
legitimate, well-documented negative result, and arguably more valuable than the
misleading v1 number would have been.

---

## 8. Phase 5 — Evaluation (optional capstone)

Status: **not run / optional.** With 0/120 perfect rollouts and ~0.01 held-out
advantage, pass@k on standard benchmarks will be ≈0 — known going in. Phase 5 is
therefore a *formal ceiling confirmation*, not a discovery:

- Metric: **pass@k** using the **unbiased combinatorial estimator** (not the
  biased plug-in `1 − (1 − c/n)^k`, which underestimates).
- Benchmarks: held-out `greengerong/leetcode` + HumanEval, reusing Phase 4
  execution machinery.
- Caveat: decontamination was vs AIME/GPQA/MATH-500/LiveCodeBench, *not*
  necessarily LeetCode — watch CodeForces↔LeetCode overlap if scores look high.
- Style note: the model defaults to CodeForces stdin/stdout I/O; LeetCode's
  function-signature style is a mismatch to handle in the eval harness.

---

## 9. Artifacts & key files

```
phase3/
  decoder_only_v3.py            # GPTv3 model class
  pretrain.py                   # pretraining loop
  finetune.py                   # SFT loop
  checkpoints_pretrain_v2/      # base model (best val 1.117)
  checkpoints_finetune_v2/      # SFT model — GRPO policy + reference (step 200)
phase4/
  audit_tests.py
  code_executor.py              # run_solution(), reward_fraction(),
                                #   constant_baseline(), reward_advantage()
  rollouts.py                   # generate_group()
  build_curriculum.py           # → curriculum.jsonl (old reward, 200 SOLVABLE)
                                # → curriculum_v2.jsonl (advantage, 24 SOLVABLE)
  grpo.py                       # the GRPO training loop
  compare_grpo.py               # side-by-side PRE vs POST (--pre/--post flags)
  checkpoints_grpo_v1/          # 1500-step run (reward-hacked) — cautionary artifact
  checkpoints_grpo_v2/          # anti-hack run; best.pt = step 125
```

**Keep `grpo_v1/best.pt` as a labeled cautionary artifact** — it's the
reward-hacked model and the evidence for the write-up. **`grpo_v2/best.pt`** is
the anti-hack result.

---

## 10. Reproducibility notes

- Train/held-out splits are deterministic from **seed=42**; the frozen curriculum
  makes them reproducible without persisting indices.
- All GRPO checkpoints are full-state (model + optimizer + step + RNG + history)
  and `--resume` picks up bit-exactly.
- GRPO hyperparameters that worked: **G=8, lr=1e-6, kl_beta=0.04, clip_eps=0.2.**
- GPU peak stayed ~3 GB throughout GRPO (5+ GB headroom on the 8 GB card).

---

## 11. Lessons worth keeping

1. **A rising reward curve is not proof of learning.** Always read the actual
   outputs. The v1 held-out reward looked like a clean 10× win; the code showed
   the model had learned to cheat.
2. **Reward hacking is the default, not the exception.** Given a gameable reward,
   the model found the exploit fast and abandoned genuine attempts for it.
3. **Diagnostic for hacking:** subtract the score a trivial/constant strategy
   would earn; if the gain vanishes, it was hacking.
4. **A clean negative result can be the real contribution.** Correctly diagnosing
   *why* a number is misleading is worth more than the misleading number.
5. **Separate "does the mechanism work" from "did the model improve."** They are
   different questions with different success criteria.
6. **Know your ceiling.** At 97M params on CodeForces-difficulty problems, the
   honest learnable signal was ~1.8% — model scale, not pipeline quality, was the
   binding constraint.
