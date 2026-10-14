"""Ablation C: ONE agent does everything (investigate with the MCP tools, write the script, read the result,
improve it) in a single ReAct-style loop. Same tools, same sandbox, same harness scoring, same attempt budget as
the multi-agent graph; what is removed is the separation into Planner / Feature / Coding / Critic agents and the
strategy-escalation rule.

    python -m agenticsdk.agents.react --dataset credit_approval --seed 0 --run-id react_demo

The report it writes has the same shape as the graph's report.json (``strategy_history`` is empty: there is no
Planner), so the benchmark statistics treat both alike.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agenticsdk.agents import prompts
from agenticsdk.agents.run import RUNS_DIR, initial_state
from agenticsdk.agents.schemas import AgentState, Attempt, ExecSummary
from agenticsdk.harness.baselines import BUDGET_SECONDS
from agenticsdk.harness.datasets import REGISTRY
from agenticsdk.harness.sandbox import (
    PRED_TEST_FILE,
    PRED_VAL_FILE,
    TEST_FEATURES_FILE,
    TRAIN_FILE,
    VAL_FEATURES_FILE,
    prepare_workdir,
    run_script,
)
from agenticsdk.harness.scorer import normalized_score, score_file
from agenticsdk.llm import DEFAULT_MODEL, LLMClient
from agenticsdk.tools.client import MCPToolClient

REACT_SYSTEM = """You are a single autonomous machine-learning engineer. You work in a loop:
1. Optionally investigate the data with tools (profile_dataset, inspect_column, run_quick_cv). Tool results are
   real measurements on the training split. Call at most one tool per turn.
2. Write ONE complete, self-contained Python script that trains a model and writes predictions.
3. You are then shown the harness validation score (computed by the evaluation harness, not your script) next to
   frozen baselines. Improve the script, or reply with the single word DONE if further changes are unlikely to help.

Hard rules for scripts (violations are rejected before the script runs):
- Allowed imports only: numpy, pandas, scipy, sklearn, lightgbm, and harmless stdlib modules (math, json, re,
  time, warnings, collections, itertools, functools, typing, random, statistics, copy, pathlib). No os, sys, subprocess.
- Read only train.csv, val_features.csv, test_features.csv. Write only predictions_val.csv and predictions_test.csv.
  No other paths, no eval/exec/getattr/dunder attributes.
- The script must finish within the wall-clock limit given in the task. Fix every random seed.
Reply with a short plan and ONE fenced ```python code block when you write a script."""


def _prompt(s: AgentState, notes: list[str], attempts: list[Attempt], left: int, tools_left: int) -> str:
    base = ", ".join(f"{k}={v:.6f}" for k, v in sorted(s.baselines_val.items()))
    out = [prompts.task_block(s, with_profile=not (tools_left > 0 and not attempts)),
           f"\nFrozen baselines on the validation split (B0 default random forest, B1 default LightGBM, B2 FLAML): {base}"]
    if notes:
        out.append("\nWhat your investigation found so far:\n" + "\n".join(notes))
    for a in attempts:
        res = (f"harness validation score {a.val_score:.6f}" if a.val_valid else f"NO valid predictions: {a.error}")
        out.append(f"\nYour script {a.index}:\n```python\n{a.code}\n```\nResult: {res}\n"
                   + (f"Its stdout:\n{a.exec.stdout.strip()}" if a.exec.stdout.strip() else ""))
    out.append(f"\nScripts left: {left}. " + (f"Investigation calls left: {tools_left}. " if tools_left else "")
               + ("Write the first script now." if not attempts else
                  "Write an improved complete script, or reply DONE if you are satisfied."))
    return "\n".join(out)


def _execute(s: AgentState, code: str, idx: int, run_dir: Path) -> Attempt:
    wd = run_dir / f"attempt_{idx}"
    prepare_workdir(s.dataset_key, s.seed, wd)
    r = run_script(code, wd, timeout=s.budget_seconds)
    res = ExecSummary(exit_code=r.exit_code, stdout=r.stdout, stderr=r.stderr, duration=r.duration,
                      timed_out=r.timed_out, blocked=r.blocked)
    att = Attempt(index=idx, strategy_index=1, code=code, exec=res, workdir=str(wd))
    if r.blocked:
        att.error = r.stderr
    elif r.timed_out:
        att.error = f"the script exceeded the {s.budget_seconds}s wall-clock limit and was killed. Make it faster."
    elif r.exit_code != 0:
        att.error = f"the script crashed (exit code {r.exit_code}). stderr:\n{r.stderr.strip()[-1500:]}"
    else:
        missing = [f for f in (PRED_VAL_FILE, PRED_TEST_FILE) if not (wd / f).exists()]
        if missing:
            att.error = f"the script finished but did not write: {missing}"
        else:
            v = score_file(wd / PRED_VAL_FILE, s.dataset_key, s.seed, "val")
            t = score_file(wd / PRED_TEST_FILE, s.dataset_key, s.seed, "test")   # format check only
            if not v.valid:
                att.error = f"predictions_val.csv is invalid: {v.error}"
            elif not t.valid:
                att.error = f"predictions_test.csv is invalid: {t.error}"
            else:
                att.val_valid, att.val_score = True, v.score
    for f in (TRAIN_FILE, VAL_FEATURES_FILE, TEST_FEATURES_FILE):
        (wd / f).unlink(missing_ok=True)
    return att


def run_react(key: str, seed: int = 0, run_id: str | None = None, mode: str = "cache", model: str = DEFAULT_MODEL,
              max_total_attempts: int = 6, tool_budget: int = 5, budget: int = BUDGET_SECONDS) -> AgentState:
    run_id = run_id or f"react_{key}_s{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = LLMClient(run_dir=run_dir, mode=mode, model=model)
    s = initial_state(key, seed, run_id, 0, budget, max_total_attempts=max_total_attempts, use_tools=True)
    tools = MCPToolClient(key, seed).start()
    notes: list[str] = []
    tool_log: list[dict] = []
    attempts: list[Attempt] = []
    status = "exhausted"
    try:
        while len(attempts) < max_total_attempts:
            tools_left = tool_budget - len(tool_log)
            prompt = _prompt(s, notes, attempts, max_total_attempts - len(attempts), tools_left)
            if tools_left > 0 and not attempts:   # investigation is offered until the first script is written
                step = llm.tool_step(REACT_SYSTEM, prompt, tools.specs, "react", thinking_budget=1024)
                if step.tool:
                    try:
                        result = tools.call(step.tool, step.arguments)
                    except Exception as e:  # noqa: BLE001 - a failing tool must not kill the run
                        result = f"ERROR: tool call failed: {type(e).__name__}: {e}"
                    notes.append(f"- {step.tool}({json.dumps(step.arguments)}): {result[:prompts.MAX_TOOL_RESULT_CHARS]}")
                    tool_log.append({"agent": "react", "tool": step.tool, "arguments": step.arguments,
                                     "ok": not result.startswith("ERROR")})
                    continue
                reply = step.text
                if not prompts.extract_code(reply):   # no tool and no script: ask for the script explicitly
                    reply = llm.text(REACT_SYSTEM, prompt + "\nNo more investigation: write the script now.", "react",
                                     thinking_budget=2048)
            else:
                reply = llm.text(REACT_SYSTEM, prompt, "react", thinking_budget=2048)
            code = prompts.extract_code(reply)
            if not code:
                if attempts and "DONE" in reply.upper():
                    status = "passed"
                    break
                attempts.append(Attempt(index=len(attempts) + 1, code="", exec=ExecSummary(), workdir="",
                                        error="your reply contained no ```python code block"))
                continue
            attempts.append(_execute(s, code, len(attempts) + 1, run_dir))
    finally:
        tools.close()
    valid = [a for a in attempts if a.val_valid]
    out: dict = {"attempts": attempts, "tool_log": tool_log, "status": status if valid else "failed"}
    if valid:
        best = max(valid, key=lambda a: a.val_score)
        t = score_file(Path(best.workdir) / PRED_TEST_FILE, key, seed, "test")
        out.update(best_attempt=best.index, final_val_score=best.val_score, final_test_score=t.score)
        if s.baselines_test and {"B0", "B1", "B2"} <= set(s.baselines_test):
            out["final_normalized"] = normalized_score(t.score, s.baselines_test["B0"],
                                                       max(s.baselines_test["B1"], s.baselines_test["B2"]))
    final = s.model_copy(update=out)
    (run_dir / "report.json").write_text(json.dumps(
        {**final.model_dump(exclude={"trace", "attempts"}),
         "attempts": [{k: v for k, v in a.model_dump().items() if k != "code"} for a in final.attempts],
         "llm_usage": llm.usage(), "variant": "react"}, indent=1, ensure_ascii=False), encoding="utf-8")
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-attempts", type=int, default=6)
    a = ap.parse_args()
    f = run_react(a.dataset, a.seed, a.run_id, model=a.model, max_total_attempts=a.max_attempts)
    print(f"run_id={f.run_id} status={f.status} attempts={len(f.attempts)} val={f.final_val_score} "
          f"test={f.final_test_score} normalized={f.final_normalized}")
