"""The AgenticSDK web service (Phase 8).

    uvicorn agenticsdk.service.app:app --port 7860          # API docs at /docs, UI at /ui
    DEMO_MODE=1 uvicorn ...                                  # public-demo caps (see service/jobs.py: Limits)

    POST   /runs                  upload a CSV + the target column (multipart); returns {"run_id": ...}
    GET    /runs/{id}             status, strategies, attempts, baselines, result
    GET    /runs/{id}/events      Server-Sent Events: every graph event as it happens
    POST   /runs/{id}/approve     human-in-the-loop: let the proposed strategy run
    POST   /runs/{id}/reject      ... or veto it, with feedback for the planner
    GET    /runs/{id}/script      the winning script
    GET    /runs/{id}/predictions its predictions on the held-out test rows
    DELETE /runs/{id}             delete the run and the uploaded data

Generated code is screened by the allow-list and runs as a subprocess with a stripped environment; set
AGENTICSDK_SANDBOX=docker (see Dockerfile.sandbox) to run it in a container with no network instead.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import time
import uuid
from collections import defaultdict, deque

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agenticsdk.harness.datasets import UploadError, register_custom
from agenticsdk.llm import DEFAULT_MODEL
from agenticsdk.service import jobs

app = FastAPI(title="AgenticSDK", version="0.1.0",
              description="Upload a table, watch a critic-guided multi-agent system build a model, "
                          "and compare it with frozen baselines.")
_recent: dict[str, deque] = defaultdict(deque)   # client -> start times of its recent runs (demo rate limit)


class Decision(BaseModel):
    feedback: str = ""


def _limits() -> jobs.Limits:
    return jobs.Limits.from_env()


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _spawn(run_id: str) -> None:
    subprocess.Popen([sys.executable, "-m", "agenticsdk.service.worker", run_id], cwd=jobs.ROOT,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def _job_or_404(run_id: str) -> dict:
    try:
        job = jobs.read_job(run_id)
    except ValueError:
        job = None
    if not job:
        raise HTTPException(404, "no such run")
    return job


def key_configured() -> bool:
    """Is a Gemini key available to the worker processes (environment or the project's .env)?"""
    from dotenv import load_dotenv

    from agenticsdk.llm import ROOT
    load_dotenv(ROOT / ".env")
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


@app.get("/health")
def health() -> dict:
    lim = _limits()
    return {"ok": True, "demo_mode": lim.max_rows is not None, "active_runs": len(jobs.active_jobs()),
            "model": DEFAULT_MODEL, "key_configured": key_configured()}


@app.post("/runs", status_code=202)
async def create_run(request: Request, file: UploadFile = File(...), target: str = Form(...),
                     task: str = Form("auto"), hitl: bool = Form(False)) -> dict:
    lim = _limits()
    if len(jobs.active_jobs()) >= lim.max_concurrent:
        raise HTTPException(429, "the service is busy with another run; try again in a few minutes")
    if lim.daily_live_calls is not None and jobs.live_calls_today() >= lim.daily_live_calls:
        raise HTTPException(503, "the demo's daily model-call allowance is used up; it resets within 24 hours")
    if lim.runs_per_hour_per_client is not None:
        q, now = _recent[_client(request)], time.time()
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= lim.runs_per_hour_per_client:
            raise HTTPException(429, f"at most {lim.runs_per_hour_per_client} runs per hour per visitor in the demo")
        q.append(now)
    raw = await file.read((lim.max_bytes or 50_000_000) + 1)
    if lim.max_bytes and len(raw) > lim.max_bytes:
        raise HTTPException(413, f"the file is larger than {lim.max_bytes // 1_000_000} MB")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001 - any parse failure is the uploader's to fix
        raise HTTPException(400, f"could not read the file as CSV: {type(e).__name__}") from e
    run_id = uuid.uuid4().hex[:12]
    key = f"custom_{run_id}"
    try:
        spec = register_custom(key, df, target, task, max_rows=lim.max_rows)
    except UploadError as e:
        raise HTTPException(400, str(e)) from e
    params = {"target": target, "kind": spec.kind, "hitl": hitl, "model": DEFAULT_MODEL,
              "max_total_attempts": lim.max_total_attempts, "planner_tool_budget": lim.planner_tool_budget,
              "feature_tool_budget": lim.feature_tool_budget, "budget_seconds": lim.budget_seconds,
              "baseline_seconds": lim.baseline_seconds, "approval_timeout_s": lim.approval_timeout_s}
    jobs.write_job(run_id, {"run_id": run_id, "dataset_key": key, "status": "queued", "params": params,
                            "filename": (file.filename or "upload.csv")[:80], "created": time.time(),
                            "rows": len(df), "columns": int(df.shape[1])})
    _spawn(run_id)
    return {"run_id": run_id, "status": "queued", "task": spec.kind}


def _summary(run_id: str, job: dict) -> dict:
    events = jobs.read_events(run_id)
    strategies: list[dict] = []
    for e in events:   # in order: an approval event refers to the strategy proposed just before it
        if e["node"] == "planner" and "strategy" in e.get("data", {}):
            strategies.append({**e["data"]["strategy"], "index": e["data"]["strategy_index"], "human": None})
        elif e["node"] == "approval" and strategies:
            strategies[-1]["human"] = "rejected" if "rejected" in e["detail"] else "approved"
    attempts = [{"index": i + 1, "detail": e["detail"],
                 **{k: e["data"].get(k) for k in ("strategy_index", "val_score", "error", "timed_out", "blocked")}}
                for i, e in enumerate(x for x in events if x["node"] == "execute")]
    verdicts = [{"decision": e["data"].get("decision"), "detail": e["detail"]} for e in events if e["node"] == "critic"]
    return {"run_id": run_id, "status": job["status"], "kind": job["params"]["kind"], "hitl": job["params"]["hitl"],
            "rows": job.get("rows"), "columns": job.get("columns"), "baselines": job.get("baselines"),
            "proposal": job.get("proposal"), "strategies": strategies, "attempts": attempts, "verdicts": verdicts,
            "events": len(events), "result": job.get("result"), "error": job.get("error")}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _summary(run_id, _job_or_404(run_id))


@app.get("/runs/{run_id}/events")
async def stream_events(run_id: str) -> StreamingResponse:
    _job_or_404(run_id)

    async def gen():
        sent, last_status = 0, None
        while True:
            job = jobs.read_job(run_id) or {}
            fresh = jobs.read_events(run_id, sent)
            for ev in fresh:
                sent += 1
                yield f"event: {ev['node']}\ndata: {json.dumps(ev)}\n\n"
            status = job.get("status")
            if status != last_status:
                last_status = status
                yield f"event: status\ndata: {json.dumps({'status': status, 'proposal': job.get('proposal')})}\n\n"
            if status in ("done", "failed") and not fresh:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _decide(run_id: str, approve: bool, body: Decision) -> dict:
    job = _job_or_404(run_id)
    if job["status"] != "awaiting_approval":
        raise HTTPException(409, f"the run is '{job['status']}', not waiting for a decision")
    (jobs.run_dir(run_id) / "decision.json").write_text(
        json.dumps({"approve": approve, "feedback": body.feedback[:500]}), encoding="utf-8")
    return {"run_id": run_id, "approved": approve}


@app.post("/runs/{run_id}/approve")
def approve(run_id: str, body: Decision | None = None) -> dict:
    return _decide(run_id, True, body or Decision())


@app.post("/runs/{run_id}/reject")
def reject(run_id: str, body: Decision | None = None) -> dict:
    return _decide(run_id, False, body or Decision())


def _artifact(run_id: str, name: str, media: str) -> FileResponse:
    _job_or_404(run_id)
    f = jobs.run_dir(run_id) / "final" / name
    if not f.exists():
        raise HTTPException(404, "not available (the run has no valid result yet)")
    return FileResponse(f, media_type=media, filename=f"{run_id}_{name}")


@app.get("/runs/{run_id}/script")
def script(run_id: str) -> FileResponse:
    return _artifact(run_id, "solution.py", "text/x-python")


@app.get("/runs/{run_id}/predictions")
def predictions(run_id: str) -> FileResponse:
    return _artifact(run_id, "predictions_test.csv", "text/csv")


@app.delete("/runs/{run_id}")
def delete(run_id: str) -> dict:
    job = _job_or_404(run_id)
    if job["status"] in jobs.ACTIVE:
        raise HTTPException(409, "the run is still active")
    jobs.delete_run(run_id)
    return {"deleted": run_id}


def _mount_ui() -> None:
    if os.environ.get("AGENTICSDK_NO_UI") == "1":
        return
    try:
        import gradio as gr

        from agenticsdk.service.ui import build_ui
    except ImportError:   # gradio is an optional extra ("serve")
        return
    gr.mount_gradio_app(app, build_ui(), path="/ui")


_mount_ui()
