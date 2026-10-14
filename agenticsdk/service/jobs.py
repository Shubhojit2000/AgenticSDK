"""Job bookkeeping for the service: one directory per run, plain JSON files, no database.

    runs/<run_id>/job.json        params + coarse status (written by the API and the worker)
    runs/<run_id>/events.jsonl    graph events (written by the graph, one JSON object per line)
    runs/<run_id>/decision.json   the human's answer while a run waits for approval (written by the API)
    runs/<run_id>/report.json     the final report (written by the graph)

Status: queued -> preparing (baselines) -> running <-> awaiting_approval -> done | failed
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from agenticsdk.harness.datasets import CUSTOM_DIR, RAW_DIR, SPLIT_DIR
from agenticsdk.llm import quota_day_start

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = ROOT / "runs"
RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")
ACTIVE = ("queued", "preparing", "running", "awaiting_approval")


@dataclass(frozen=True)
class Limits:
    """Caps for a public demo (DEMO_MODE=1) so one visitor cannot spend the API key or fill the disk."""
    max_bytes: int | None = None
    max_rows: int | None = None
    max_total_attempts: int = 6
    planner_tool_budget: int = 3
    feature_tool_budget: int = 2
    budget_seconds: int = 60
    baseline_seconds: int = 60
    max_concurrent: int = 2
    runs_per_hour_per_client: int | None = None
    daily_live_calls: int | None = None
    approval_timeout_s: int = 1800

    @staticmethod
    def from_env() -> Limits:
        if os.environ.get("DEMO_MODE", "0") not in ("1", "true", "True"):
            return Limits()
        return Limits(max_bytes=5_000_000, max_rows=20_000, max_total_attempts=4, planner_tool_budget=2,
                      feature_tool_budget=1, budget_seconds=30, baseline_seconds=20, max_concurrent=1,
                      runs_per_hour_per_client=3, daily_live_calls=int(os.environ.get("DEMO_DAILY_CALLS", "300")),
                      approval_timeout_s=600)


def run_dir(run_id: str, runs_dir: Path | None = None) -> Path:
    runs_dir = runs_dir or RUNS_DIR
    if not RUN_ID_RE.match(run_id):
        raise ValueError("bad run id")
    return runs_dir / run_id


def read_job(run_id: str, runs_dir: Path | None = None) -> dict | None:
    runs_dir = runs_dir or RUNS_DIR
    f = run_dir(run_id, runs_dir) / "job.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):   # caught mid-replace (Windows refuses the read); the next poll succeeds
        return None


def write_job(run_id: str, update: dict, runs_dir: Path | None = None) -> dict:
    runs_dir = runs_dir or RUNS_DIR
    d = run_dir(run_id, runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    job = {**(read_job(run_id, runs_dir) or {}), **update, "updated": time.time()}
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(job, indent=1, default=str), encoding="utf-8")
    for attempt in range(40):   # Windows refuses to replace a file that a reader has open at that instant
        try:
            os.replace(tmp, d / "job.json")
            break
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)
    return job


def read_events(run_id: str, start: int = 0, runs_dir: Path | None = None) -> list[dict]:
    runs_dir = runs_dir or RUNS_DIR
    f = run_dir(run_id, runs_dir) / "events.jsonl"
    if not f.exists():
        return []
    out = []
    for ln in f.read_text(encoding="utf-8").splitlines()[start:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            break   # a half-written last line
    return out


def active_jobs(runs_dir: Path | None = None) -> list[str]:
    runs_dir = runs_dir or RUNS_DIR
    out = []
    for f in runs_dir.glob("*/job.json"):
        j = read_job(f.parent.name, runs_dir) if RUN_ID_RE.match(f.parent.name) else None
        if j and j.get("status") in ACTIVE:
            out.append(f.parent.name)
    return out


def live_calls_today(runs_dir: Path | None = None) -> int:
    runs_dir = runs_dir or RUNS_DIR
    since = quota_day_start()
    n = 0
    for f in runs_dir.glob("*/llm_calls.jsonl"):
        if f.stat().st_mtime >= since:
            n += sum(1 for ln in f.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not json.loads(ln).get("cached") and json.loads(ln).get("ts", 0) >= since)
    return n


def delete_run(run_id: str, runs_dir: Path | None = None) -> None:
    """Remove a run and the uploaded data behind it (raw table, splits, spec)."""
    runs_dir = runs_dir or RUNS_DIR
    key = f"custom_{run_id}"
    shutil.rmtree(run_dir(run_id, runs_dir), ignore_errors=True)
    shutil.rmtree(SPLIT_DIR / key, ignore_errors=True)
    for p in (RAW_DIR / f"{key}.parquet", CUSTOM_DIR / f"{key}.json", CUSTOM_DIR / f"{key}.baselines.json"):
        p.unlink(missing_ok=True)
