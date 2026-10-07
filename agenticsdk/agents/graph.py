"""LangGraph wiring (Phase 2):

    planner -> feature -> training -> execute (sandbox + harness scoring) -> critic
                 ^                        ^                                     |
                 |                        +----------- RETRY_CODE --------------+
                 +-------------------------------- RETRY_STRATEGY ---------------+
                                                   PASS / budget spent -> report

The Critic's verdict is checked against the budget (code retries per strategy, strategy changes,
total attempts) in code, so the LLM can ask for more than the budget allows but cannot get it.
A new strategy must differ from every earlier one on (model_family, feature_approach); that is
enforced by `_violation`, not by prompt wording."""
from __future__ import annotations

import json
from pathlib import Path

from langgraph.graph import END, StateGraph

from agenticsdk.agents import prompts
from agenticsdk.agents.schemas import AgentState, Attempt, Event, ExecSummary, FeaturePlan, Strategy, Verdict
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
from agenticsdk.llm import LLMClient
from agenticsdk.tracing import get_tracer

PLANNER_TRIES = 3   # asks per strategy before the run gives up for lack of a novel strategy


def _emit(run_dir: Path, node: str, detail: str, **data) -> Event:
    ev = Event(node=node, detail=detail, data=data)
    with open(run_dir / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ev.model_dump(), ensure_ascii=False) + "\n")
    return ev


def _clip(data: dict, limit: int = 2000) -> dict:
    """Event data for trace output: long strings are truncated so spans stay readable."""
    return {k: (v[:limit] + " ..." if isinstance(v, str) and len(v) > limit else v) for k, v in data.items()}


def _violation(s: AgentState, strat: Strategy) -> str | None:
    """Why a proposed strategy is not acceptable, or None."""
    for i, h in enumerate(s.strategy_history, 1):
        if strat.key == h.key:
            return (f"your proposal repeats strategy {i} (model_family={h.model_family}, "
                    f"feature_approach={h.feature_approach}). Choose a different pair.")
    if not s.strategy_history and s.first_family and strat.model_family != s.first_family:
        return f"model_family must be '{s.first_family}' for the first strategy."
    return None


def build_graph(llm: LLMClient, run_dir: Path, checkpointer=None, tools=None):
    """``tools``: an object with ``.specs`` (list of {name, description, parameters}) and
    ``.call(name, arguments) -> str``, e.g. agenticsdk.tools.client.MCPToolClient. Needed iff state.use_tools."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def explore(s: AgentState, role: str, budget: int) -> tuple[str, list[dict]]:
        """Tool-calling loop: the model picks tools (native function calling) until it answers in text or
        the per-agent budget is spent. Returns the rendered findings and the tool log entries."""
        if tools is None:
            raise RuntimeError("state.use_tools is set but no tool client was given to build_graph")
        steps: list[dict] = []
        notes, log = "", []
        while len(steps) < budget:
            step = llm.tool_step(prompts.EXPLORE_SYSTEM, prompts.explore_prompt(s, role, steps, budget - len(steps)),
                                 tools.specs, role, thinking_budget=1024)
            if not step.tool:
                notes = step.text
                break
            with get_tracer().observe(f"tool:{step.tool}", as_type="tool", input=step.arguments,
                                      metadata={"agent": role}) as tspan:
                try:
                    result = tools.call(step.tool, step.arguments)
                except Exception as e:  # noqa: BLE001 - a failing tool must not kill the run
                    result = f"ERROR: tool call failed: {type(e).__name__}: {e}"
                tspan.update(output=result[:4000], level="ERROR" if result.startswith("ERROR") else None,
                             status_message=result[:200] if result.startswith("ERROR") else None)
            steps.append({"tool": step.tool, "arguments": step.arguments, "result": result})
            log.append({"agent": role, "tool": step.tool, "arguments": step.arguments,
                        "ok": not result.startswith("ERROR")})
            _emit(run_dir, "tool", f"{role} called {step.tool}", agent=role, tool=step.tool,
                  arguments=step.arguments, result=result[:600], ok=log[-1]["ok"])
        return prompts.findings_text(steps, notes), log

    # ---- nodes ------------------------------------------------------------------------------
    def planner_node(s: AgentState) -> dict:
        note, strat, findings, log = "", None, "", []
        if s.use_tools:
            findings, log = explore(s, "planner", s.planner_tool_budget)
        for attempt in range(PLANNER_TRIES):
            cand = llm.structured(Strategy, prompts.PLANNER_SYSTEM, prompts.planner_prompt(s, note, findings),
                                  "planner", thinking_budget=2048)
            why = _violation(s, cand)
            if why is None:
                strat = cand
                break
            _emit(run_dir, "planner", f"proposal rejected: {why}", rejected=cand.model_dump(), try_=attempt + 1)
            note = f"Your previous proposal was rejected: {why}"
        if strat is None:
            ev = _emit(run_dir, "planner", f"no novel strategy after {PLANNER_TRIES} tries; stopping")
            return {"status": "exhausted" if s.attempts else "failed", "tool_log": s.tool_log + log,
                    "trace": s.trace + [ev]}
        ev = _emit(run_dir, "planner",
                   f"strategy {len(s.strategy_history) + 1}: {strat.model_family} + {strat.feature_approach}: "
                   f"{strat.summary}", strategy=strat.model_dump(), strategy_index=len(s.strategy_history) + 1)
        return {"current_strategy": strat, "strategy_history": s.strategy_history + [strat],
                "tool_log": s.tool_log + log, "trace": s.trace + [ev]}

    def feature_node(s: AgentState) -> dict:
        findings, log = explore(s, "feature", s.feature_tool_budget) if s.use_tools else ("", [])
        fp = llm.structured(FeaturePlan, prompts.FEATURE_SYSTEM, prompts.feature_prompt(s, findings), "feature",
                            thinking_budget=1024)
        ev = _emit(run_dir, "feature", f"{len(fp.steps)} feature steps", plan=fp.model_dump())
        return {"feature_plan": fp, "tool_log": s.tool_log + log, "trace": s.trace + [ev]}

    def training_node(s: AgentState) -> dict:
        reply = llm.text(prompts.TRAINING_SYSTEM, prompts.training_prompt(s), "training", thinking_budget=2048)
        code = prompts.extract_code(reply)
        ev = _emit(run_dir, "training", f"attempt {len(s.attempts) + 1}: "
                   + ("script written" if code else "NO code block in reply"), chars=len(code or ""))
        return {"current_code": code or "", "trace": s.trace + [ev]}

    def execute_node(s: AgentState) -> dict:
        idx = len(s.attempts) + 1
        sidx = len(s.strategy_history)
        wd = run_dir / f"attempt_{idx}"
        prepare_workdir(s.dataset_key, s.seed, wd)
        if not s.current_code:
            res = ExecSummary(stderr="The reply contained no ```python code block.")
            att = Attempt(index=idx, strategy_index=sidx, code="", exec=res,
                          error="your reply contained no ```python code block", workdir=str(wd))
        else:
            r = run_script(s.current_code, wd, timeout=s.budget_seconds)
            res = ExecSummary(exit_code=r.exit_code, stdout=r.stdout, stderr=r.stderr, duration=r.duration,
                              timed_out=r.timed_out, blocked=r.blocked)
            att = Attempt(index=idx, strategy_index=sidx, code=s.current_code, exec=res, workdir=str(wd))
            if r.blocked:
                att.error = r.stderr
            elif r.timed_out:
                att.error = (f"the script exceeded the {s.budget_seconds}s wall-clock limit and was killed. "
                             "Make it faster (fewer/lighter models, fewer trees or folds, subsample for tuning).")
            elif r.exit_code != 0:
                att.error = f"the script crashed (exit code {r.exit_code}). stderr:\n{r.stderr.strip()[-1500:]}"
            else:
                missing = [f for f in (PRED_VAL_FILE, PRED_TEST_FILE) if not (wd / f).exists()]
                if missing:
                    att.error = f"the script finished but did not write: {missing}"
                else:
                    v = score_file(wd / PRED_VAL_FILE, s.dataset_key, s.seed, "val")
                    t = score_file(wd / PRED_TEST_FILE, s.dataset_key, s.seed, "test")  # format check only
                    if not v.valid:
                        att.error = f"predictions_val.csv is invalid: {v.error}"
                    elif not t.valid:
                        att.error = f"predictions_test.csv is invalid: {t.error}"
                    else:
                        att.val_valid, att.val_score = True, v.score
        for f in (TRAIN_FILE, VAL_FEATURES_FILE, TEST_FEATURES_FILE):  # keep run dirs small
            (wd / f).unlink(missing_ok=True)
        detail = (f"attempt {idx} (strategy {sidx}): val {att.val_score:.5f}" if att.val_valid
                  else f"attempt {idx} (strategy {sidx}): FAILED")
        ev = _emit(run_dir, "execute", detail, strategy_index=sidx, exit_code=res.exit_code,
                   duration=res.duration, timed_out=res.timed_out, blocked=res.blocked,
                   val_score=att.val_score, error=att.error)
        return {"attempts": s.attempts + [att], "trace": s.trace + [ev]}

    def critic_node(s: AgentState) -> dict:
        att = s.attempts[-1]
        total_left = len(s.attempts) < s.max_total_attempts      # s.attempts already holds this attempt
        code_left = total_left and s.code_retry_count < s.max_code_retries
        strat_left = total_left and s.strategy_retry_count < s.max_strategy_retries

        if att.error:  # deterministic fast path: nothing for an LLM to judge yet
            verdict = Verdict(decision="RETRY_CODE", reasoning="No valid scored predictions.", feedback=att.error)
            att.verdict_source = "harness"
        else:
            verdict = llm.structured(Verdict, prompts.CRITIC_SYSTEM, prompts.critic_prompt(s, att), "critic",
                                     thinking_budget=1024)
            att.verdict_source = "llm"
            floor = min(s.baselines_val.values()) if s.baselines_val else None
            if verdict.decision == "PASS" and floor is not None and att.val_score < floor and (code_left or strat_left):
                verdict = Verdict(decision="RETRY_CODE", reasoning=(
                    f"Overridden: harness val score {att.val_score:.5f} is below the weakest baseline {floor:.5f}."),
                    feedback=f"Your validation score {att.val_score:.5f} is below even the default random forest "
                             f"({floor:.5f}). Find out why (leakage in CV, bad preprocessing, underfitting) and fix it.")
                att.verdict_source = "override"

        # the budget decides what is allowed, whatever the Critic asked for
        if verdict.decision == "RETRY_CODE" and not code_left and strat_left:
            verdict = Verdict(decision="RETRY_STRATEGY", reasoning=(
                f"Code retries for this strategy are used up. {verdict.reasoning}"), feedback=verdict.feedback)
            att.verdict_source = "override" if att.verdict_source == "llm" else att.verdict_source
        elif verdict.decision == "RETRY_STRATEGY" and not strat_left and code_left:
            verdict = Verdict(decision="RETRY_CODE", reasoning=(
                f"No strategy changes left. {verdict.reasoning}"), feedback=verdict.feedback)
            att.verdict_source = "override"

        att.verdict_decision = verdict.decision
        update: dict = {"verdict": verdict}
        if verdict.decision == "PASS":
            update["status"] = "passed"
        elif verdict.decision == "RETRY_CODE" and code_left:
            update["code_retry_count"] = s.code_retry_count + 1
        elif verdict.decision == "RETRY_STRATEGY" and strat_left:
            update["strategy_retry_count"] = s.strategy_retry_count + 1
            update["code_retry_count"] = 0
        else:
            update["status"] = "exhausted"
        ev = _emit(run_dir, "critic", f"{verdict.decision} ({att.verdict_source}): {verdict.reasoning}",
                   decision=verdict.decision, source=att.verdict_source, feedback=verdict.feedback,
                   strategy_index=att.strategy_index)
        update["attempts"] = s.attempts[:-1] + [att]
        update["trace"] = s.trace + [ev]
        return update

    def report_node(s: AgentState) -> dict:
        valid = [a for a in s.attempts if a.val_valid]
        out: dict = {}
        if not valid:
            out["status"] = "failed"
        else:
            best = max(valid, key=lambda a: a.val_score)  # model selection uses VALIDATION only
            t = score_file(Path(best.workdir) / PRED_TEST_FILE, s.dataset_key, s.seed, "test")
            out.update(best_attempt=best.index, final_val_score=best.val_score, final_test_score=t.score)
            if s.baselines_test and {"B0", "B1", "B2"} <= set(s.baselines_test):
                ref = max(s.baselines_test["B1"], s.baselines_test["B2"])
                out["final_normalized"] = normalized_score(t.score, s.baselines_test["B0"], ref)
        status = out.get("status", s.status)
        out["status"] = "running" if status == "running" else status
        ev = _emit(run_dir, "report", f"status={out['status']} test={out.get('final_test_score')}",
                   **{k: v for k, v in out.items() if k != "status"})
        out["trace"] = s.trace + [ev]
        final = s.model_copy(update=out)
        (run_dir / "report.json").write_text(
            json.dumps({**final.model_dump(exclude={"trace", "attempts"}),
                        "attempts": [{k: v for k, v in a.model_dump().items() if k != "code"} for a in final.attempts],
                        "llm_usage": llm.usage()}, indent=1, ensure_ascii=False), encoding="utf-8")
        return out

    # ---- wiring -----------------------------------------------------------------------------
    def after_planner(s: AgentState) -> str:
        return "report" if s.status != "running" else "feature"

    def after_critic(s: AgentState) -> str:
        if s.status != "running":
            return "report"
        return "planner" if s.verdict.decision == "RETRY_STRATEGY" else "training"

    def traced(name: str, fn):
        """Wrap a node in a Langfuse span whose output is the node's own event (decision, score, error)."""
        def run(s: AgentState) -> dict:
            with get_tracer().observe(name, as_type="chain",
                                      metadata={"strategy_index": len(s.strategy_history),
                                                "attempts_so_far": len(s.attempts)}) as span:
                out = fn(s)
                if out.get("trace"):
                    ev = out["trace"][-1]
                    span.update(output={"detail": ev.detail, **_clip(ev.data)})
                return out
        return run

    g = StateGraph(AgentState)
    for name, fn in (("planner", planner_node), ("feature", feature_node), ("training", training_node),
                     ("execute", execute_node), ("critic", critic_node), ("report", report_node)):
        g.add_node(name, traced(name, fn))
    g.set_entry_point("planner")
    g.add_conditional_edges("planner", after_planner, {"feature": "feature", "report": "report"})
    g.add_edge("feature", "training")
    g.add_edge("training", "execute")
    g.add_edge("execute", "critic")
    g.add_conditional_edges("critic", after_critic,
                            {"planner": "planner", "training": "training", "report": "report"})
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)
