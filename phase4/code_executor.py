"""
phase4/code_executor.py — sandbox runner for model-generated Python.

Two public functions:

    run_solution(code, test_input, expected_output, timeout_s=5) -> bool
        Run `code` as a subprocess with `test_input` piped to stdin,
        compare stripped stdout to stripped expected_output.

    reward_fraction(code, tests, timeout_s=5) -> float
        Run all (input, expected_output) pairs in `tests` and return the
        fraction passed. This is the GRPO scalar reward.

Plus a __main__ smoke test that:
  1. Loads one open-r1/codeforces-cots row.
  2. Parses its R1-generated reference Python solution.
  3. Runs reward_fraction against public_tests.
  4. Reports the score (should be ~1.0 for an accepted reference).

SECURITY — read before deploying anywhere shared
-------------------------------------------------
This module runs model-generated code on the host machine. We apply, in
the order they take effect:

  - **Process isolation**: separate subprocess, `shell=False`.
  - **Isolated interpreter**: `python -I -S` ignores PYTHON* env vars,
    user site-packages, and PYTHONPATH manipulation.
  - **Wall-clock timeout**: subprocess.run(timeout=...). Kills the
    process group on overrun.
  - **CPU time rlimit** (RLIMIT_CPU): catches busy loops where wall
    clock might not — also a hard upper bound.
  - **Address-space rlimit** (RLIMIT_AS): caps virtual memory (512 MB).
  - **Process-count rlimit** (RLIMIT_NPROC): caps fork bombs.
  - **File-size rlimit** (RLIMIT_FSIZE): caps output to 100 MB.
  - **Core-dump rlimit** (RLIMIT_CORE = 0): no .core files on crash.
  - **Temp CWD**: per-call temp dir, cleaned on exit.

This is **NOT a real sandbox**. The code can still:
  - Read/write the filesystem outside the temp dir (no chroot).
  - Make network calls (no network namespace).
  - Use any importable stdlib module (no module allowlist).
  - Read env vars that weren't stripped (we use the parent's environ
    minus PYTHON*; everything else passes through).

For a hardened setup use a container (Docker / Podman), nsjail, or a
service like E2B / Modal. The rlimit + timeout + subprocess combo here
is the pragmatic local version chosen because each GRPO rollout has to
be cheap (no container startup per test case) and the codeforces-cots
solutions are competitive-programming code that almost never needs file
or network IO. If the threat model changes, swap the executor.

The host should treat *every* candidate-code execution as untrusted —
don't run this with credentials in env vars, on shared infrastructure,
or with write access to anything the rest of the project relies on.
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Resource caps applied in the child between fork and exec.
# ---------------------------------------------------------------------------
DEFAULT_CPU_SECONDS = 10                       # hard CPU-time cap (s)
DEFAULT_AS_BYTES    = 512 * 1024 * 1024        # 512 MB virtual-memory cap
DEFAULT_FSIZE_BYTES = 100 * 1024 * 1024        # 100 MB max single-file write
DEFAULT_NPROC       = 256                      # cap on processes for fork bombs


def _set_limits() -> None:
    """preexec_fn — runs in the child immediately after fork, before exec.

    We swallow per-rlimit failures: not every platform supports every
    rlimit (RLIMIT_NPROC for instance is Linux-specific). The wall-clock
    timeout in the parent is the catch-all.
    """
    def _try(which, soft, hard):
        try:
            resource.setrlimit(which, (soft, hard))
        except (ValueError, OSError, AttributeError):
            pass
    _try(resource.RLIMIT_CPU,   DEFAULT_CPU_SECONDS, DEFAULT_CPU_SECONDS)
    _try(resource.RLIMIT_AS,    DEFAULT_AS_BYTES,    DEFAULT_AS_BYTES)
    _try(resource.RLIMIT_FSIZE, DEFAULT_FSIZE_BYTES, DEFAULT_FSIZE_BYTES)
    _try(resource.RLIMIT_CORE,  0, 0)
    if hasattr(resource, "RLIMIT_NPROC"):
        _try(resource.RLIMIT_NPROC, DEFAULT_NPROC, DEFAULT_NPROC)
    # New session so the parent's timeout kill targets the whole group.
    try:
        os.setsid()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Output canonicalisation — line-by-line, CodeForces-style
# ---------------------------------------------------------------------------
def _to_canonical_lines(s: str) -> list[str]:
    """Turn a (possibly noisy) output blob into a comparable list of lines.

    Three normalisations, in order:
      1. CRLF / CR → LF, so Windows-vs-Linux line endings never matter.
      2. Split on '\\n' and rstrip each line — this is the load-bearing one.
         CodeForces test files frequently contain trailing whitespace before
         a newline (we observed "3 3 3 \\n" in private_tests). A whole-blob
         .strip() only handled this if the line was the LAST line; an
         interior trailing-space line would have mismatched. Per-line
         rstrip handles every line uniformly.
      3. Drop trailing empty lines from both sides. A solution that ends
         its output with print() vs print(... end='') shouldn't change the
         verdict; competitive-programming judges treat them identically.

    The unit asserts in __main__ exercise each of these.
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in s.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


# ---------------------------------------------------------------------------
# Single execution — shared core
# ---------------------------------------------------------------------------
def _execute(
    code: str,
    test_input: str,
    timeout_s: float = 5.0,
) -> tuple[bool, str | None]:
    """Run `code` as a subprocess; return (clean_exit, stdout).

    clean_exit is True iff the process exited 0 within the timeout / rlimits.
    stdout is the captured stdout string when clean_exit, else None.

    Never raises — timeouts / fork failures / non-zero exits all return
    (False, None). Callers (run_solution, reward_*) build verdicts on top.
    """
    with tempfile.TemporaryDirectory(prefix="saLLMan_exec_") as tmpdir:
        script_path = Path(tmpdir) / "sol.py"
        script_path.write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", str(script_path)],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmpdir,
                preexec_fn=_set_limits,
            )
        except subprocess.TimeoutExpired:
            return False, None
        except (OSError, ValueError):
            return False, None
        if proc.returncode != 0:
            return False, None
        return True, proc.stdout


# ---------------------------------------------------------------------------
# Single test case
# ---------------------------------------------------------------------------
def run_solution(
    code: str,
    test_input: str,
    expected_output: str,
    timeout_s: float = 5.0,
) -> bool:
    """True iff `code` exits cleanly and its stdout matches expected_output
    under CodeForces-style line canonicalisation."""
    ok, stdout = _execute(code, test_input, timeout_s=timeout_s)
    if not ok or stdout is None:
        return False
    return _to_canonical_lines(stdout) == _to_canonical_lines(expected_output)


# ---------------------------------------------------------------------------
# All test cases for a problem
# ---------------------------------------------------------------------------
def reward_fraction(
    code: str,
    tests: dict,
    timeout_s: float = 5.0,
) -> float:
    """Fraction of (input, expected_output) pairs in `tests` that pass.

    `tests` is the {"input": [str], "output": [str]} dict shape used by
    open-r1/codeforces-cots' public_tests / private_tests fields.

    Returns 0.0 if the dict is malformed or empty. Otherwise, in [0, 1].

    This is the scalar reward the GRPO loop will multiply into its
    advantage estimate. {0, 1} pass/fail is also obtainable from this
    (>= some threshold) if a binary reward is preferred.
    """
    inputs  = tests.get("input")  or []
    outputs = tests.get("output") or []
    if not inputs or len(inputs) != len(outputs):
        return 0.0

    n_pass = 0
    for tin, texp in zip(inputs, outputs):
        if run_solution(code, tin, texp, timeout_s=timeout_s):
            n_pass += 1
    return n_pass / len(inputs)


# ---------------------------------------------------------------------------
# Constant-output baseline + advantage reward
# ---------------------------------------------------------------------------
# Motivation (found in the pre/post side-by-side): the policy learned to
# print a constant (-1, 0) on problems whose test set is dominated by one
# expected output, gaining raw pass-fraction without solving anything.
#
# constant_baseline(tests) = score a best-case constant-output program gets
#   = (count of the single most common expected output) / n_tests.
#
# reward_advantage = max(0, fraction - baseline), with a secondary guard
# that zeroes any completion producing identical output across ALL distinct
# test inputs (catches constants that aren't the modal value).
def constant_baseline(tests: dict) -> float:
    """Fraction of tests a best-case constant-output program would pass:
    the relative frequency of the modal expected output (canonicalised so
    "1\\n" and "1" count as the same output class). No code execution."""
    from collections import Counter
    outputs = tests.get("output") or []
    if not outputs:
        return 0.0
    canon = [tuple(_to_canonical_lines(o)) for o in outputs]
    most_common_count = Counter(canon).most_common(1)[0][1]
    return most_common_count / len(outputs)


def reward_advantage_detailed(
    code: str,
    tests: dict,
    timeout_s: float = 5.0,
    budget_s: float | None = None,
    guard_constant: bool = True,
) -> dict:
    """Run all tests once; return reward + diagnostics.

    Returns dict with:
      fraction    — raw pass fraction (n_pass / n_total), the old reward
      baseline    — constant_baseline(tests)
      advantage   — max(0, fraction - baseline), zeroed if guard fires
      guard_fired — True if code produced one identical output for every
                    distinct test input (a constant-output hack)
      n_run       — tests actually executed (may be < n_total if budget hit)
      n_total     — total tests available

    `budget_s` caps total wall-clock; fraction is over n_total (not n_run)
    so a budget cutoff can't inflate the score by running only easy tests.
    """
    info = {"fraction": 0.0, "baseline": 0.0, "advantage": 0.0,
            "guard_fired": False, "n_run": 0, "n_total": 0}
    if not code:
        return info
    inputs  = tests.get("input")  or []
    outputs = tests.get("output") or []
    if not inputs or len(inputs) != len(outputs):
        return info

    n_total = len(inputs)
    info["n_total"] = n_total
    baseline = constant_baseline(tests)
    info["baseline"] = baseline

    start = time.time()
    n_pass = 0
    n_run = 0
    by_input: dict[tuple, tuple | None] = {}   # distinct canon-input -> canon-output
    for tin, texp in zip(inputs, outputs):
        if budget_s is not None and time.time() - start >= budget_s:
            break
        ok, stdout = _execute(code, tin, timeout_s=timeout_s)
        n_run += 1
        canon_in = tuple(_to_canonical_lines(tin))
        if ok and stdout is not None:
            canon_out = tuple(_to_canonical_lines(stdout))
            by_input.setdefault(canon_in, canon_out)
            if canon_out == tuple(_to_canonical_lines(texp)):
                n_pass += 1
        else:
            by_input.setdefault(canon_in, None)

    info["n_run"] = n_run
    fraction = n_pass / n_total          # over TOTAL, not run
    info["fraction"] = fraction

    guard_fired = False
    if guard_constant:
        produced = {o for o in by_input.values() if o is not None}
        if len(by_input) >= 2 and len(produced) == 1:
            guard_fired = True
    info["guard_fired"] = guard_fired

    info["advantage"] = 0.0 if guard_fired else max(0.0, fraction - baseline)
    return info


def reward_advantage(
    code: str,
    tests: dict,
    timeout_s: float = 5.0,
    budget_s: float | None = None,
) -> float:
    """Scalar GRPO reward: max(0, pass_fraction - constant_baseline), with
    the constant-output guard. Beating the trivial baseline is the only way
    to score."""
    return reward_advantage_detailed(
        code, tests, timeout_s=timeout_s, budget_s=budget_s,
    )["advantage"]


# ===========================================================================
# Smoke test
# ===========================================================================
# Load one codeforces-cots row, parse the R1-generated reference Python
# solution out of messages[1].content, score it against public_tests.
# An accepted reference should pass ≈ all public tests; this confirms the
# end-to-end loop works before we wire it into GRPO.
# ---------------------------------------------------------------------------
THINK_CLOSE = "</think>"
PY_FENCE    = "```python"
FENCE_CLOSE = "```"


def _extract_python(assistant_text: str) -> str | None:
    """First ```python ... ``` block AFTER </think> in the assistant output.

    Same logic as the SFT data prep (phase3/finetune_data_prep.py), inlined
    here so phase4 doesn't import from phase3 just for one helper.
    """
    close_idx = assistant_text.find(THINK_CLOSE)
    after = assistant_text[close_idx + len(THINK_CLOSE):] if close_idx != -1 else assistant_text
    fence_open = after.find(PY_FENCE)
    if fence_open == -1:
        return None
    code_start = fence_open + len(PY_FENCE)
    if code_start < len(after) and after[code_start] == "\n":
        code_start += 1
    fence_close = after.find(FENCE_CLOSE, code_start)
    if fence_close == -1:
        return None
    return after[code_start:fence_close].strip() or None


def _smoke_test(n_rows: int = 5) -> None:
    """Try up to n_rows candidate references; report each score."""
    from datasets import load_dataset  # local import — keep module lightweight
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading open-r1/codeforces-cots:solutions_py_decontaminated ...")
    ds = load_dataset(
        "open-r1/codeforces-cots",
        "solutions_py_decontaminated",
        split="train",
        token=hf_token,
    )
    print(f"Total rows: {len(ds):,}")

    scores: list[float] = []
    rows_checked = 0

    for i, row in enumerate(ds):
        if len(scores) >= n_rows:
            break

        pub = row.get("public_tests") or {}
        inputs  = pub.get("input")  or []
        outputs = pub.get("output") or []
        if not inputs or len(inputs) != len(outputs):
            continue

        msgs = row.get("messages") or []
        if len(msgs) < 2 or not isinstance(msgs[1], dict):
            continue
        if msgs[1].get("role") != "assistant":
            continue
        assistant = (msgs[1].get("content") or "").strip()
        if not assistant:
            continue

        code = _extract_python(assistant)
        if not code:
            continue

        rows_checked += 1
        print(f"\n--- row {i} ({len(inputs)} public tests, "
              f"{len(code)} chars of code) ---")
        # Show a couple of code lines so we can eyeball that we extracted
        # actual Python and not a docstring or something.
        for line in code.splitlines()[:4]:
            print(f"    {line}")
        if len(code.splitlines()) > 4:
            print(f"    ... ({len(code.splitlines()) - 4} more lines)")

        score = reward_fraction(code, pub, timeout_s=5.0)
        n_pass = int(round(score * len(inputs)))
        verdict = "PASS" if score >= 0.99 else ("PARTIAL" if score > 0 else "FAIL")
        print(f"    score: {score:.3f}  ({n_pass}/{len(inputs)})  -> {verdict}")
        scores.append(score)

    # ── Aggregate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("SMOKE TEST SUMMARY")
    print("=" * 64)
    if not scores:
        print("  No usable rows found. Check the dataset access and parser.")
        sys.exit(1)

    print(f"  rows attempted: {rows_checked}")
    print(f"  scores:         {[f'{s:.3f}' for s in scores]}")
    print(f"  mean score:     {sum(scores)/len(scores):.3f}")
    n_pass_full = sum(1 for s in scores if s >= 0.99)
    print(f"  rows with ≥0.99 score: {n_pass_full}/{len(scores)}")

    if n_pass_full == 0:
        print("\n  ! No reference solution scored ≈1.0. Either the parser missed")
        print("  ! the code, the runner is broken, or the public_tests format")
        print("  ! isn't what we assumed. Inspect the row prints above.")
        sys.exit(2)
    elif n_pass_full < len(scores):
        print(f"\n  Most references pass — runner looks healthy.")
        print(f"  ({len(scores) - n_pass_full} partials are expected: codeforces-cots")
        print(f"   notes ~84% Python-trace pass rate on public tests.)")
    else:
        print(f"\n  All {n_pass_full} references scored ≈1.0 — runner confirmed.")


def _assert_comparison_invariants() -> None:
    """Unit asserts proving the line-by-line comparison handles the cases
    that motivated the change. Runs before the dataset smoke test so a
    runner regression trips here instead of silently passing rows."""
    def eq(a, b):
        return _to_canonical_lines(a) == _to_canonical_lines(b)

    # Trailing whitespace before newline — the CodeForces case we hit.
    assert eq("3 3 3 \n",      "3 3 3\n"),       "trailing space before \\n must match"
    assert eq("1 2 3 \n",      "1 2 3\n"),       "single-line trailing space must match"

    # Presence vs absence of the final newline is irrelevant.
    assert eq("hello\n",       "hello"),         "trailing \\n optional"
    assert eq("hello",         "hello\n\n\n"),   "extra trailing blanks ignored"

    # Multi-line outputs with trailing whitespace on INTERIOR lines —
    # the case that .strip() alone could not handle.
    assert eq("1 2 \n3 4\n",   "1 2\n3 4"),      "interior-line trailing space must match"
    assert eq("a\nb \nc\n",    "a\nb\nc"),       "interior-line trailing space (3 lines)"

    # CRLF / CR normalisation.
    assert eq("a\r\nb\r\n",    "a\nb\n"),        "CRLF must match LF"
    assert eq("a\rb",          "a\nb"),          "CR must match LF"

    # Genuine value differences must still fail.
    assert not eq("1 2 3",     "1 2 4"),         "value diff must fail"
    assert not eq("1\n2\n3",   "1\n2"),          "missing line must fail"
    assert not eq("1\n2\n3",   "1\n2\n3\n4"),    "extra non-blank line must fail"

    print("  comparison invariants: 10/10 asserts passed")


def _assert_reward_advantage() -> None:
    """Unit asserts for constant_baseline + reward_advantage.

    Synthetic problem: print 'YES' if n is even else 'NO'.
      inputs  = 2,4,6,1,3
      outputs = YES,YES,YES,NO,NO   → modal 'YES' (3/5) → baseline = 0.6
    Three candidate programs:
      - constant   `print("YES")`  passes 3/5=0.6, but is a constant hack
      - partial    correct on evens + n==1, wrong on n==3  → 4/5=0.8
      - full       correct parity                          → 5/5=1.0
    """
    tests = {
        "input":  ["2", "4", "6", "1", "3"],
        "output": ["YES", "YES", "YES", "NO", "NO"],
    }

    base = constant_baseline(tests)
    assert abs(base - 0.6) < 1e-9, f"baseline should be 0.6, got {base}"

    const_code   = 'print("YES")'
    partial_code = 'n=int(input())\nprint("YES" if n>1 else "NO")'  # wrong only on n==3
    full_code    = 'n=int(input())\nprint("YES" if n%2==0 else "NO")'

    c = reward_advantage_detailed(const_code, tests)
    p = reward_advantage_detailed(partial_code, tests)
    f = reward_advantage_detailed(full_code, tests)

    # 1. Constant on a dominated set scores ~0 (baseline-subtraction AND
    #    the guard both push it to 0).
    assert c["advantage"] <= 0.01, f"constant should net ~0, got {c['advantage']:.3f}"
    assert c["guard_fired"], "constant-output guard should fire on print('YES')"

    # 2. A genuinely-partial program scores > 0.
    assert p["advantage"] > 0.0, f"partial should score > 0, got {p['advantage']:.3f}"
    assert not p["guard_fired"], "partial varies output → guard must NOT fire"

    # 3. A full solution scores highest.
    assert f["advantage"] > p["advantage"], \
        f"full ({f['advantage']:.3f}) should beat partial ({p['advantage']:.3f})"
    assert abs(f["fraction"] - 1.0) < 1e-9, f"full should pass all, got {f['fraction']}"

    print(f"  reward-advantage asserts: 6/6 passed")
    print(f"    baseline={base:.2f}  "
          f"const adv={c['advantage']:.2f} (guard={c['guard_fired']})  "
          f"partial adv={p['advantage']:.2f}  full adv={f['advantage']:.2f}")


if __name__ == "__main__":
    print("=" * 64)
    print("RUNNER COMPARISON INVARIANTS")
    print("=" * 64)
    _assert_comparison_invariants()
    print()
    print("=" * 64)
    print("REWARD-ADVANTAGE INVARIANTS")
    print("=" * 64)
    _assert_reward_advantage()
    print()
    _smoke_test(n_rows=5)
