"""Check the Phase 4 goals locally (no API key, no Langfuse account, no network except OpenML on first use).

    venv\\Scripts\\python tests\\verify\\verify_phase4.py

Runs: lint, the full test suite with coverage, the tracing tests, and a no-secrets run.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # tests/verify/ -> project root
PY = sys.executable
results: list[tuple[str, bool]] = []


def run(cmd: list[str], env_extra: dict | None = None, drop: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update({"PYTHONPATH": str(ROOT), **(env_extra or {})})
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")


if __name__ == "__main__":
    r = run([PY, "-m", "ruff", "check", "."])
    check("lint is clean (ruff)", r.returncode == 0, (r.stdout.strip().splitlines() or [""])[-1])

    r = run([PY, "-m", "pytest", "tests/test_tracing.py", "-q"])
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    check("tracing: real Langfuse SDK spans have the right tree, tokens, cost, errors", r.returncode == 0, tail)

    secrets = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
    r = run([PY, "-m", "pytest", "--cov", "--cov-report=term", "-q"], {"AGENTICSDK_TRACING": "off"}, drop=secrets)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    m = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", r.stdout)
    check("whole suite passes with no API keys in the environment", r.returncode == 0,
          f"{tail}; coverage {m.group(1) + '%' if m else 'n/a'}")

    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    uses_secret = re.search(r"\$\{\{\s*secrets\.", ci) is not None      # a real ${{ secrets.X }} reference
    check("CI workflow runs ruff and pytest and uses no secrets", "ruff check" in ci and "pytest" in ci
          and not uses_secret, "no `${{ secrets.* }}` reference in .github/workflows/ci.yml")

    r = run([PY, "-m", "agenticsdk.tracing"], drop=secrets, env_extra={"AGENTICSDK_TRACING": "off"})
    check("without keys, tracing is a clean no-op", r.returncode == 1 and "Tracing is off" in r.stdout,
          r.stdout.strip().splitlines()[0][:100] if r.stdout.strip() else r.stderr[-100:])

    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    print("Still needs YOU: (1) Langfuse keys + `python -m agenticsdk.tracing`, (2) pushing to GitHub for the real CI run.")
    sys.exit(1 if bad else 0)
