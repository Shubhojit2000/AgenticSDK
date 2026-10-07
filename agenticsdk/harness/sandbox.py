"""Run one agent-written script in an isolated working directory.

Per attempt:
  * a fresh directory that contains ONLY ``train.csv`` (features + ``target``),
    ``val_features.csv`` and ``test_features.csv``; no hidden labels exist there;
  * the script runs as a separate interpreter process with a stripped environment (no API keys);
  * a wall-clock timeout kills the whole process *tree* (psutil; works on Windows);
  * stdout/stderr are captured and truncated for the LLM.

The agent's script must write ``predictions_val.csv`` and ``predictions_test.csv`` to the working
directory; the harness (not the agent) scores them. See ``agenticsdk/harness/scorer.py`` for the format.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from agenticsdk.harness.allowlist import check_script
from agenticsdk.harness.datasets import TARGET, load_split

# Names of the files the agent sees (kept in sync with allowlist.py).
TRAIN_FILE = "train.csv"
VAL_FEATURES_FILE = "val_features.csv"
TEST_FEATURES_FILE = "test_features.csv"
PRED_VAL_FILE = "predictions_val.csv"
PRED_TEST_FILE = "predictions_test.csv"
AGENT_TARGET = "target"  # the target column is called plainly "target" for the agent

MAX_OUTPUT_CHARS = 3000


@dataclass
class ExecResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    blocked: bool = False            # rejected by the allow-list, never executed
    violations: list[str] = field(default_factory=list)
    workdir: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.blocked


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = limit // 3
    return text[:head] + f"\n... [{len(text) - limit} chars truncated] ...\n" + text[-(limit - head):]


def _redact(text: str, workdir: Path) -> str:
    """Remove machine-specific paths so LLM prompts (and therefore the replay cache) are stable."""
    wd = str(workdir)
    for variant in {wd, wd.replace("\\", "/"), str(workdir.resolve()), str(workdir.resolve()).replace("\\", "/")}:
        text = text.replace(variant, "<workdir>")
    return text


def _clean_env() -> dict[str, str]:
    """Only what Python needs to start; in particular no GOOGLE_API_KEY / GEMINI_API_KEY."""
    keep = ("SYSTEMROOT", "SYSTEMDRIVE", "PATH", "PATHEXT", "TEMP", "TMP", "COMSPEC", "WINDIR",
            "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "TMPDIR")   # the last five matter on Linux/macOS
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    return env


def kill_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=5)


def prepare_workdir(key: str, seed: int, workdir: Path) -> None:
    """Materialize exactly what the agent may see. Never writes val/test targets."""
    data = load_split(key, seed)
    workdir.mkdir(parents=True, exist_ok=True)
    train = data["train"].rename(columns={TARGET: AGENT_TARGET})
    train.to_csv(workdir / TRAIN_FILE, index=False)
    data["val"].drop(columns=TARGET).to_csv(workdir / VAL_FEATURES_FILE, index=False)
    data["test"].drop(columns=TARGET).to_csv(workdir / TEST_FEATURES_FILE, index=False)


def run_script(code: str, workdir: Path, timeout: float, screen: bool = True) -> ExecResult:
    """Run ``code`` as ``solution.py`` inside ``workdir`` (already prepared)."""
    if screen:
        verdict = check_script(code)
        if not verdict.ok:
            return ExecResult(None, "", "Script rejected before execution by the safety allow-list:\n- "
                              + "\n- ".join(verdict.violations), 0.0, blocked=True,
                              violations=verdict.violations, workdir=str(workdir))
    script = workdir / "solution.py"
    script.write_text(code, encoding="utf-8")
    t0 = time.time()
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        [sys.executable, "-W", "ignore", "solution.py"], cwd=workdir, env=_clean_env(),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", creationflags=flags,
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(proc.pid)
        try:
            out, err = proc.communicate(timeout=5)
        except Exception:
            out, err = "", ""
        err = (err or "") + f"\n[sandbox] killed: exceeded the {timeout:.0f}s wall-clock limit"
    return ExecResult(
        exit_code=None if timed_out else proc.returncode,
        stdout=_truncate(_redact(out or "", workdir)), stderr=_truncate(_redact(err or "", workdir)),
        duration=round(time.time() - t0, 2), timed_out=timed_out, workdir=str(workdir),
    )
