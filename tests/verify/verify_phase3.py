"""Offline check of the Phase 3 goals (MCP tools + native tool calling). No API key, no LLM calls.

    venv\\Scripts\\python tests\\verify\\verify_phase3.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # tests/verify/ -> project root
sys.path.insert(0, str(ROOT))

from agenticsdk.agents.run import run_dataset  # noqa: E402
from agenticsdk.tools.client import MCPToolClient  # noqa: E402

results: list[tuple[str, bool]] = []


def check(name: str, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")


def server_tools():
    with MCPToolClient("credit_g", 0) as c:
        names = sorted(s["name"] for s in c.specs)
        profile = c.call("profile_dataset", {})
        bad = c.call("inspect_column", {"name": "no_such_column"})
        cv = c.call("run_quick_cv", {"model_family": "linear", "folds": 2})
    ok = (names == ["inspect_column", "profile_dataset", "run_quick_cv"] and "rows (train): 600" in profile
          and bad.startswith("ERROR") and "neg_log_loss mean" in cv)
    return ok, f"MCP server over stdio lists {names}; a bad column gives a readable error, not a crash; quick CV: {cv[:90]}..."


def tools_reject_bad_input():
    with MCPToolClient("credit_g", 0) as c:
        outs = [c.call("run_quick_cv", {"model_family": "svm_rbf"}),
                c.call("run_quick_cv", {"model_family": "linear", "params_json": '{"C": 1e9}'}),
                c.call("run_quick_cv", {"model_family": "linear", "params_json": '{"os": "rm -rf"}'})]
    return all(o.startswith("ERROR") for o in outs), "unknown family, out-of-range value and unknown parameter are all refused: " + " | ".join(o[:45] for o in outs)


def tools_use_train_only():
    src = (ROOT / "agenticsdk" / "tools" / "dataset_tools.py").read_text(encoding="utf-8")
    ok = "val.parquet" not in src.replace("never val.parquet", "") and "test.parquet" not in src.replace("test.parquet\n", "")
    return ok, "agenticsdk/tools/dataset_tools.py opens train.parquet and meta.json only (tests/test_mcp_tools.py also runs it on a split folder with no val/test files)"


def recorded_trace():
    ev = [json.loads(x) for x in (ROOT / "runs" / "cp3b_eucalyptus" / "events.jsonl").read_text().splitlines()]
    first_strategy = next(i for i, e in enumerate(ev) if e["node"] == "planner" and "strategy" in e["data"])
    before = [e["data"]["tool"] for e in ev[:first_strategy] if e["node"] == "tool"]
    return ("profile_dataset" in before and "run_quick_cv" in before), f"before committing to strategy 1 the planner called: {before}"


def replay():
    rid = "verify3_replay"
    shutil.rmtree(ROOT / "runs" / rid, ignore_errors=True)
    f = run_dataset("eucalyptus", 0, rid, mode="replay_only", use_tools=True)
    u = json.loads((ROOT / "runs" / rid / "report.json").read_text())["llm_usage"]
    return f.status == "passed" and u["api_calls"] == 0, (
        f"recorded tool-using run replayed: {u['api_calls']} API calls, {u['cache_hits']} cache hits, "
        f"{len(f.tool_log)} tool calls re-executed locally, test log-loss score {f.final_test_score:.4f}")


if __name__ == "__main__":
    for n, f in (("MCP server works over stdio", server_tools), ("tools refuse bad input", tools_reject_bad_input),
                 ("tools read the train split only", tools_use_train_only),
                 ("recorded run: tools called before the strategy was chosen", recorded_trace),
                 ("recorded tool-using run replays with 0 API calls", replay)):
        check(n, f)
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)
