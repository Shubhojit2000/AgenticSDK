"""Phase 8: the web service end to end (FastAPI TestClient, scripted LLM, real sandbox, real harness scoring)."""
import io
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agenticsdk.agents import run as agent_run
from agenticsdk.harness import datasets, sandbox
from agenticsdk.llm import RawResult, ToolStep
from agenticsdk.service import app as app_mod
from agenticsdk.service import jobs, worker
from agenticsdk.service.jobs import Limits
from tests.test_graph_mocked import FEATURES, GOOD, PASS, fence, strat


@pytest.fixture
def env(tmp_path, monkeypatch):
    for mod in (datasets, jobs):
        monkeypatch.setattr(mod, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(mod, "SPLIT_DIR", tmp_path / "splits")
        monkeypatch.setattr(mod, "CUSTOM_DIR", tmp_path / "custom")
    for mod in (jobs, worker, agent_run):
        monkeypatch.setattr(mod, "RUNS_DIR", tmp_path / "runs")
    spawned = []
    monkeypatch.setattr(app_mod, "_spawn", lambda run_id: spawned.append(run_id))
    monkeypatch.setattr(app_mod, "_limits", lambda: Limits(baseline_seconds=3, budget_seconds=30))
    monkeypatch.delitem(datasets.REGISTRY, "x", raising=False)
    yield TestClient(app_mod.app), spawned, tmp_path
    for k in [k for k in datasets.REGISTRY if k.startswith("custom_")]:
        datasets.REGISTRY.pop(k)


def csv_bytes(n=300, seed=0, **over) -> bytes:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "c": rng.choice(["a", "b", "c"], n),
                       "y": np.where(x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n) > 0, "yes", "no")})
    for k, v in over.items():
        df[k] = v
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def upload(client, content=None, target="y", **form):
    return client.post("/runs", files={"file": ("data.csv", content or csv_bytes(), "text/csv")},
                       data={"target": target, **form})


# ---- upload validation -------------------------------------------------------------------------------------
def test_upload_creates_a_job_registers_the_dataset_and_starts_a_worker(env):
    client, spawned, _ = env
    r = upload(client, hitl="true")
    assert r.status_code == 202 and r.json()["task"] == "binary"
    run_id = r.json()["run_id"]
    assert spawned == [run_id] and datasets.REGISTRY[f"custom_{run_id}"].kind == "binary"
    job = jobs.read_job(run_id)
    assert job["status"] == "queued" and job["params"]["hitl"] is True and job["rows"] == 300
    got = client.get(f"/runs/{run_id}").json()
    assert got["status"] == "queued" and got["strategies"] == [] and got["hitl"] is True


@pytest.mark.parametrize("content,target,msg", [
    (csv_bytes(), "nope", "not in the file"),
    (csv_bytes(n=50), "y", "at least 100"),
    (csv_bytes(y="same"), "y", "single value"),
    (b"\x00\x01\x02 not a csv \xff", "y", "could not read"),
])
def test_bad_uploads_are_refused_with_a_readable_reason(env, content, target, msg):
    client, spawned, _ = env
    r = upload(client, content, target)
    assert r.status_code == 400 and msg in r.json()["detail"] and spawned == []


def test_demo_limits_cap_rows_size_concurrency_and_the_daily_call_allowance(env, monkeypatch):
    client, spawned, tmp = env
    monkeypatch.setattr(app_mod, "_limits", lambda: Limits(max_rows=200, max_bytes=100_000, max_concurrent=1,
                                                           runs_per_hour_per_client=2, daily_live_calls=5))
    assert "at most 200" in upload(client).json()["detail"]
    assert upload(client, csv_bytes(n=150)).status_code == 202
    assert upload(client, csv_bytes(n=150)).status_code == 429          # one run at a time
    jobs.write_job(spawned[0], {"status": "done"})
    d = tmp / "runs" / "aaaaaaaaaaaa"
    d.mkdir(parents=True)
    (d / "llm_calls.jsonl").write_text("\n".join(json.dumps({"cached": False, "ts": time.time()}) for _ in range(6)))
    assert upload(client, csv_bytes(n=150)).status_code == 503           # the key's daily allowance is spent
    monkeypatch.setattr(app_mod, "_limits", lambda: Limits(max_bytes=1000))
    assert upload(client).status_code == 413


def test_unknown_and_malformed_run_ids_are_404(env):
    client, _, _ = env
    assert client.get("/runs/abcdefabcdef").status_code == 404
    assert client.get("/runs/..%2f..%2fetc").status_code == 404
    assert client.get("/runs/abcdefabcdef/script").status_code == 404


# ---- status, events, decisions, deletion ---------------------------------------------------------------------
def fake_run(tmp, run_id="abcdef123456", status="running", events=()):
    jobs.write_job(run_id, {"run_id": run_id, "dataset_key": f"custom_{run_id}", "status": status,
                            "params": {"kind": "binary", "hitl": True}})
    ev = tmp / "runs" / run_id / "events.jsonl"
    ev.write_text("\n".join(json.dumps(e) for e in events) + ("\n" if events else ""))
    return run_id


EVENTS = [
    {"node": "planner", "detail": "strategy 1", "data": {"strategy_index": 1, "strategy": {
        "model_family": "linear", "feature_approach": "minimal", "summary": "logistic regression"}}},
    {"node": "execute", "detail": "attempt 1 (strategy 1): val -0.5", "data": {"strategy_index": 1, "val_score": -0.5}},
    {"node": "critic", "detail": "PASS (llm): fine", "data": {"decision": "PASS"}},
]


def test_run_summary_is_assembled_from_the_event_log(env):
    client, _, tmp = env
    rid = fake_run(tmp, events=EVENTS)
    got = client.get(f"/runs/{rid}").json()
    assert got["strategies"][0]["model_family"] == "linear" and got["strategies"][0]["index"] == 1
    assert got["attempts"][0]["val_score"] == -0.5 and got["verdicts"][0]["decision"] == "PASS" and got["events"] == 3


def test_event_stream_replays_every_event_then_closes_when_the_run_is_over(env):
    client, _, tmp = env
    rid = fake_run(tmp, status="done", events=EVENTS)
    with client.stream("GET", f"/runs/{rid}/events") as r:
        body = "".join(r.iter_text())
    assert r.status_code == 200 and "text/event-stream" in r.headers["content-type"]
    assert body.count("event: planner") == 1 and "event: execute" in body and "event: critic" in body
    assert '"status": "done"' in body


def test_decisions_need_a_run_that_is_waiting(env):
    client, _, tmp = env
    rid = fake_run(tmp, status="running")
    assert client.post(f"/runs/{rid}/approve").status_code == 409
    jobs.write_job(rid, {"status": "awaiting_approval", "proposal": {"strategy": {}}})
    assert client.post(f"/runs/{rid}/reject", json={"feedback": "too slow"}).json()["approved"] is False
    assert json.loads((tmp / "runs" / rid / "decision.json").read_text()) == {"approve": False, "feedback": "too slow"}


def test_delete_removes_the_run_and_the_uploaded_data_but_not_an_active_run(env):
    client, _, tmp = env
    rid = upload(client).json()["run_id"]
    assert client.delete(f"/runs/{rid}").status_code == 409            # still queued
    jobs.write_job(rid, {"status": "done"})
    datasets.load_split(f"custom_{rid}", 0)                           # materialise the splits
    assert (tmp / "raw" / f"custom_{rid}.parquet").exists() and (tmp / "splits" / f"custom_{rid}").exists()
    assert client.delete(f"/runs/{rid}").status_code == 200
    assert not (tmp / "raw" / f"custom_{rid}.parquet").exists() and not (tmp / "splits" / f"custom_{rid}").exists()
    assert not (tmp / "runs" / rid).exists() and not (tmp / "custom" / f"custom_{rid}.json").exists()


# ---- a whole run through the worker, with a human in the loop -----------------------------------------------------
class Backend:
    """Planner proposes two strategies, the Coding Agent always writes the same good script, tools say 'enough'."""

    def __init__(self):
        self.strategies = [strat("bagged_trees", "encoding_focused").model_dump_json(),
                           strat("linear", "minimal", "plain logistic regression").model_dump_json()]
        self.planner_prompts = []

    def __call__(self, model, system, prompt, schema, thinking_budget, temperature, tools=None):
        name = schema.__name__ if schema else "text"
        if tools:
            return RawResult(ToolStep(text="enough").model_dump_json(), 1, 1)
        if name == "Strategy":
            self.planner_prompts.append(prompt)
            return RawResult(self.strategies.pop(0) if len(self.strategies) > 1 else self.strategies[0], 1, 1)
        if name == "FeaturePlan":
            return RawResult(FEATURES, 1, 1)
        if name == "Verdict":
            return RawResult(PASS.model_dump_json(), 1, 1)
        return RawResult(fence(GOOD), 1, 1)


def wait_for(client, rid, want, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        got = client.get(f"/runs/{rid}").json()
        if got["status"] in want:
            return got
        time.sleep(0.4)
    raise AssertionError(f"timed out waiting for {want}; last: {got}")


def test_a_full_run_waits_for_the_human_then_produces_a_script_and_predictions(env, monkeypatch):
    client, _, tmp = env
    backend = Backend()
    real = agent_run.LLMClient
    monkeypatch.setattr(agent_run, "LLMClient", lambda **kw: real(**kw, cache_dir=tmp / "cache", backend=backend,
                                                                  sleep=lambda s: None))

    class FakeTools:
        specs = [{"name": "profile_dataset", "description": "d", "parameters": {"type": "object", "properties": {}}}]

        def __init__(self, *a, **k):
            pass

        def start(self):
            return self

        def call(self, name, arguments=None):
            return "ok"

        def close(self):
            pass

    monkeypatch.setattr(agent_run, "ToolSet", FakeTools)
    monkeypatch.setattr(agent_run, "MCPToolClient", lambda *a, **k: None)
    monkeypatch.setattr(app_mod, "_spawn", lambda rid: threading.Thread(target=worker.main, args=(rid,), daemon=True).start())

    rid = upload(client, hitl="true").json()["run_id"]
    first = wait_for(client, rid, {"awaiting_approval", "failed", "done"})
    assert first["status"] == "awaiting_approval" and first["proposal"]["strategy"]["model_family"] == "bagged_trees"
    assert set(first["baselines"]) == {"B0", "B1", "B2"} and first["attempts"] == []     # nothing has run yet

    assert client.post(f"/runs/{rid}/reject", json={"feedback": "use something linear"}).status_code == 200
    second, t0 = client.get(f"/runs/{rid}").json(), time.time()
    while time.time() - t0 < 120:   # the old proposal stays visible until the worker has read the decision
        second = client.get(f"/runs/{rid}").json()
        if second["proposal"] and second["proposal"]["strategy"]["model_family"] == "linear":
            break
        time.sleep(0.3)
    assert second["proposal"]["strategy"]["model_family"] == "linear" and second["attempts"] == []
    assert "use something linear" in backend.planner_prompts[1]

    assert client.post(f"/runs/{rid}/approve").status_code == 200
    done = wait_for(client, rid, {"done", "failed"})
    assert done["status"] == "done", done
    assert done["result"]["test"] is not None and done["result"]["normalized"] is not None
    assert done["attempts"][0]["val_score"] is not None
    assert [(s["model_family"], s["human"]) for s in done["strategies"]] == [("bagged_trees", "rejected"), ("linear", "approved")]
    script = client.get(f"/runs/{rid}/script")
    preds = client.get(f"/runs/{rid}/predictions")
    assert script.status_code == 200 and "RandomForestClassifier" in script.text
    assert preds.status_code == 200 and preds.text.splitlines()[0].startswith("id,")
    n_test = len(datasets.load_split(f"custom_{rid}", 0)["test"])
    assert len(preds.text.strip().splitlines()) == n_test + 1


# ---- worker helpers and the container sandbox -------------------------------------------------------------------------
def test_error_text_shown_to_visitors_has_no_keys_or_paths():
    msg = worker.scrub("ClientError: key AIzaSyA1234567890abcdefghijklmnop failed at C:\\Users\\me\\proj\\x.py and /home/u/y.py")
    assert "AIza" not in msg and "Users" not in msg and "/home" not in msg and "<path>" in msg


def test_approval_timeout_counts_as_a_veto(env):
    _, _, tmp = env
    jobs.write_job("abcdef123456", {"status": "running"})
    out = worker.make_approver("abcdef123456", timeout=0)({"strategy": {}})
    assert out["approve"] is False and "No reviewer" in out["feedback"]
    assert jobs.read_job("abcdef123456")["status"] == "running"


def test_docker_sandbox_command_has_no_network_no_capabilities_and_limits(tmp_path, monkeypatch):
    cmd = sandbox.docker_command(tmp_path, "n1")
    joined = " ".join(cmd)
    for flag in ("--network none", "--cap-drop ALL", "--read-only", "--memory 2g", "--pids-limit 256",
                 "no-new-privileges", "-w /work"):
        assert flag in joined
    assert cmd[-4:] == ["python", "-W", "ignore", "solution.py"] and f"{tmp_path.resolve()}:/work" in joined
    assert "GOOGLE_API_KEY" not in joined

    seen = {}

    class P:
        returncode = 0
        pid = 1

        def communicate(self, timeout=None):
            return "out", ""

    monkeypatch.setenv("AGENTICSDK_SANDBOX", "docker")
    monkeypatch.setattr(sandbox.subprocess, "Popen", lambda argv, **kw: seen.update(argv=argv) or P())
    (tmp_path / "w").mkdir()
    res = sandbox.run_script("print(1)", tmp_path / "w", timeout=5)
    assert res.ok and seen["argv"][:2] == ["docker", "run"] and "--network" in seen["argv"]
