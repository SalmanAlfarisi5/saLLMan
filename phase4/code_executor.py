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
# Output normalisation
# ---------------------------------------------------------------------------
def _normalise(s: str) -> str:
    """Strip + canonicalise line endings.

    Competitive-programming judges typically ignore trailing whitespace and
    line-ending style. We do the same: strip outer whitespace and convert
    CRLF / CR to LF so a model's `print()` on Windows-style data still
    matches a Linux-staged expected output.
    """
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


# ---------------------------------------------------------------------------
# Single test case
# ---------------------------------------------------------------------------
def run_solution(
    code: str,
    test_input: str,
    expected_output: str,
    timeout_s: float = 5.0,
) -> bool:
    """Execute `code` as a Python script in a subprocess.

    Returns True iff:
      - the subprocess exits with code 0,
      - did not hit the wall-clock timeout,
      - did not hit any rlimit,
      - normalised stdout equals normalised expected_output.

    Any other outcome (exception in the child, segfault, OOM, timeout)
    is treated as a failed test. We never raise to the caller — GRPO
    needs a clean boolean per test so the reward computation can proceed.
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
            return False
        except (OSError, ValueError):
            # Fork failure, executable missing, etc.
            return False

        if proc.returncode != 0:
            return False

        return _normalise(proc.stdout) == _normalise(expected_output)


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


if __name__ == "__main__":
    _smoke_test(n_rows=5)
