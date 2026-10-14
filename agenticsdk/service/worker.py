"""One run as its own process: ``python -m agenticsdk.service.worker <run_id>``.

The API starts this after an upload. It computes the three frozen baselines for the uploaded table, runs the agent
graph (pausing for human approval when asked) and records the outcome in job.json. A separate process, so a crash
or a runaway script never takes the API down, and the checkpointed run could be resumed by hand.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

from agenticsdk.service.jobs import RUNS_DIR, read_job, run_dir, write_job

_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


_PATH_RE = re.compile(r"""[A-Za-z]:[\\/][^\s'"]+|/(?:home|Users|root)/[^\s'"]+""")


def scrub(msg: str) -> str:
    """Error text that is safe to show to a visitor: no API keys, no machine paths."""
    return _PATH_RE.sub("<path>", _KEY_RE.sub("[redacted]", msg))[:400]


def compute_baselines(key: str, seed: int, budget: int) -> dict:
    """B0 default random forest, B1 default LightGBM, B2 FLAML with a time budget, scored by the harness exactly as
    for the benchmark datasets: {"B0": {"val": .., "test": ..}, ...}."""
    from agenticsdk.harness.baselines import fit_predict
    from agenticsdk.harness.datasets import REGISTRY, load_split
    from agenticsdk.harness.scorer import score_frame
    data = load_split(key, seed)
    spec = REGISTRY[key]
    out = {}
    for b in ("B0", "B1", "B2"):
        vf, tf_, _, _ = fit_predict(b, spec.kind, data["meta"]["classes"], data["train"], data["val"], data["test"],
                                    seed, budget)
        sv, st = score_frame(vf, key, seed, "val"), score_frame(tf_, key, seed, "test")
        if not (sv.valid and st.valid):
            raise RuntimeError(f"baseline {b} produced invalid predictions")
        out[b] = {"val": sv.score, "test": st.score}
    return out


def make_approver(run_id: str, timeout: int, runs_dir: Path | None = None):
    """Blocks until the API writes decision.json (or the timeout passes, which counts as a veto: a run nobody is
    watching must not keep spending the key)."""
    runs_dir = runs_dir or RUNS_DIR
    decision_file = run_dir(run_id, runs_dir) / "decision.json"

    def approver(proposal: dict) -> dict:
        decision_file.unlink(missing_ok=True)
        write_job(run_id, {"status": "awaiting_approval", "proposal": proposal}, runs_dir)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if decision_file.exists():
                try:
                    d = json.loads(decision_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    time.sleep(0.2)
                    continue
                decision_file.unlink(missing_ok=True)
                write_job(run_id, {"status": "running", "proposal": None}, runs_dir)
                return {"approve": bool(d.get("approve")), "feedback": str(d.get("feedback", ""))[:500]}
            time.sleep(0.5)
        write_job(run_id, {"status": "running", "proposal": None}, runs_dir)
        return {"approve": False, "feedback": "No reviewer answered in time."}

    return approver


def publish_result(run_id: str, runs_dir: Path | None = None) -> dict:
    """Copy the winning script and its test predictions to runs/<id>/final/ and return the summary."""
    runs_dir = runs_dir or RUNS_DIR
    d = run_dir(run_id, runs_dir)
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    res = {"status": rep["status"], "val": rep.get("final_val_score"), "test": rep.get("final_test_score"),
           "normalized": rep.get("final_normalized"), "best_attempt": rep.get("best_attempt")}
    if rep.get("best_attempt"):
        src = d / f"attempt_{rep['best_attempt']}"
        (d / "final").mkdir(exist_ok=True)
        for name in ("solution.py", "predictions_test.csv"):
            if (src / name).exists():
                shutil.copy(src / name, d / "final" / name)
    return res


def main(run_id: str, runs_dir: Path | None = None) -> int:
    runs_dir = runs_dir or RUNS_DIR
    job = read_job(run_id, runs_dir)
    if not job:
        print("no such job", file=sys.stderr)
        return 2
    p, key = job["params"], job["dataset_key"]
    try:
        from agenticsdk.agents.run import run_dataset
        write_job(run_id, {"status": "preparing"}, runs_dir)
        baselines = compute_baselines(key, 0, p["baseline_seconds"])
        write_job(run_id, {"status": "running", "baselines": baselines}, runs_dir)
        run_dataset(key, 0, run_id, model=p["model"], max_code_retries=2, budget=p["budget_seconds"],
                    max_strategy_retries=2, max_total_attempts=p["max_total_attempts"], use_tools=True,
                    hitl=p["hitl"], approver=make_approver(run_id, p["approval_timeout_s"], runs_dir) if p["hitl"] else None,
                    baselines=baselines, overrides={"planner_tool_budget": p["planner_tool_budget"],
                                                    "feature_tool_budget": p["feature_tool_budget"]})
        result = publish_result(run_id, runs_dir)
        write_job(run_id, {"status": "done" if result["status"] in ("passed", "exhausted") else "failed",
                           "result": result, **({} if result["status"] != "failed" else {"error": "no valid predictions"})},
                  runs_dir)
        return 0
    except Exception as e:  # noqa: BLE001 - whatever went wrong must reach the job file, not just a dead process
        write_job(run_id, {"status": "failed", "error": scrub(f"{type(e).__name__}: {e}"), "proposal": None}, runs_dir)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
