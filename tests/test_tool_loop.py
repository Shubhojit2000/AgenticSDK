"""Phase 3: the planner/feature tool loop inside the graph (scripted fake LLM and fake tools)."""
import json

import pytest
from test_graph_mocked import GOOD, PASS, Script, fence

from agenticsdk.agents.graph import build_graph
from agenticsdk.agents.run import initial_state
from agenticsdk.agents.schemas import AgentState
from agenticsdk.llm import LLMClient, RawResult, ToolStep

CALL_PROFILE = ToolStep(tool="profile_dataset", arguments={})
CALL_CV = ToolStep(tool="run_quick_cv", arguments={"model_family": "gradient_boosting"})
DONE = ToolStep(text="boosting looks better than linear; use it")


class FakeTools:
    specs = [{"name": "profile_dataset", "description": "d", "parameters": {"type": "object", "properties": {}}},
             {"name": "run_quick_cv", "description": "d", "parameters": {"type": "object", "properties": {}}}]

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail:
            raise self.fail
        return {"profile_dataset": "rows (train): 600 PROFILE_MARKER",
                "run_quick_cv": "quick CV: neg_log_loss mean -0.4100 CV_MARKER"}[name]


class ToolScript(Script):
    """Script backend that also answers native tool-calling turns from a queue."""

    def __init__(self, tool_steps, *a, **kw):
        super().__init__(*a, **kw)
        self.tool_steps = list(tool_steps)

    def __call__(self, model, system, prompt, schema, thinking_budget, temperature, tools=None):
        if tools:
            self.seen.append(("tool_step", prompt))
            step = self.tool_steps.pop(0) if self.tool_steps else DONE
            return RawResult(step.model_dump_json(), 1, 1)
        return super().__call__(model, system, prompt, schema, thinking_budget, temperature)


def run(tmp_path, tool_steps, tools, verdicts=(PASS,), trainings=None, use_tools=True, **kw):
    backend = ToolScript(tool_steps, trainings or [fence(GOOD)], verdicts)
    llm = LLMClient(run_dir=tmp_path / "run", cache_dir=tmp_path / "cache", backend=backend, rpm=0, sleep=lambda s: None)
    g = build_graph(llm, tmp_path / "run", tools=tools)
    st = initial_state("credit_g", 0, "t", 2, 60, use_tools=use_tools, **kw)
    return AgentState.model_validate(g.invoke(st, config={"recursion_limit": 100})), backend, tmp_path / "run"


def test_planner_investigates_then_plans_with_the_findings(tmp_path):
    tools = FakeTools()
    # planner: profile, cv, then stops; feature agent: stops at once (default DONE)
    final, backend, run_dir = run(tmp_path, [CALL_PROFILE, CALL_CV, DONE], tools)
    assert final.status == "passed"
    assert [c[0] for c in tools.calls] == ["profile_dataset", "run_quick_cv"]
    strategy_prompt = [p for k, p in backend.seen if k == "Strategy"][0]
    assert "INVESTIGATION SO FAR" in strategy_prompt and "PROFILE_MARKER" in strategy_prompt
    assert "CV_MARKER" in strategy_prompt and "boosting looks better" in strategy_prompt
    assert "The full profile is not shown" in strategy_prompt            # the planner had to use the tool
    assert "DATASET PROFILE (train split)" not in strategy_prompt
    # the second exploration turn saw the first tool result: a real multi-turn loop
    steps = [p for k, p in backend.seen if k == "tool_step"]
    assert "PROFILE_MARKER" not in steps[0] and "PROFILE_MARKER" in steps[1]
    assert [e["tool"] for e in final.tool_log] == ["profile_dataset", "run_quick_cv"]
    events = [json.loads(x) for x in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [e["data"]["tool"] for e in events if e["node"] == "tool"] == ["profile_dataset", "run_quick_cv"]


def test_tool_budget_is_enforced_in_code(tmp_path):
    tools = FakeTools()
    steps = [CALL_PROFILE] * 10                       # a model that never stops calling tools
    backend = ToolScript(steps, [fence(GOOD)], [PASS])
    llm = LLMClient(run_dir=tmp_path / "run", cache_dir=tmp_path / "cache", backend=backend, rpm=0, sleep=lambda s: None)
    g = build_graph(llm, tmp_path / "run", tools=tools)
    st = initial_state("credit_g", 0, "t", 2, 60, use_tools=True).model_copy(
        update={"planner_tool_budget": 2, "feature_tool_budget": 1})
    out = AgentState.model_validate(g.invoke(st, config={"recursion_limit": 100}))
    planner_calls = [e for e in out.tool_log if e["agent"] == "planner"]
    feature_calls = [e for e in out.tool_log if e["agent"] == "feature"]
    assert len(planner_calls) == 2 and len(feature_calls) == 1 and out.status == "passed"


def test_a_failing_tool_is_reported_to_the_model_not_fatal(tmp_path):
    tools = FakeTools(fail=TimeoutError("server stalled"))
    final, backend, _ = run(tmp_path, [CALL_PROFILE, DONE], tools)
    assert final.status == "passed"
    assert "ERROR: tool call failed: TimeoutError: server stalled" in [p for k, p in backend.seen if k == "Strategy"][0]
    assert final.tool_log[0]["ok"] is False


def test_without_use_tools_no_tool_turn_is_ever_made(tmp_path):
    tools = FakeTools()
    final, backend, _ = run(tmp_path, [CALL_PROFILE], tools, use_tools=False)
    assert tools.calls == [] and "tool_step" not in [k for k, _ in backend.seen]
    assert "DATASET PROFILE (train split)" in [p for k, p in backend.seen if k == "Strategy"][0]
    assert final.tool_log == []


def test_use_tools_without_a_tool_client_fails_loudly(tmp_path):
    with pytest.raises(RuntimeError, match="no tool client"):
        run(tmp_path, [], None)


def test_tool_results_are_truncated_in_prompts():
    from agenticsdk.agents import prompts
    long = "x" * 10_000
    out = prompts.findings_text([{"tool": "profile_dataset", "arguments": {}, "result": long}])
    assert len(out) < prompts.MAX_TOOL_RESULT_CHARS + 200 and out.rstrip().endswith("...")


def test_replanning_investigation_sees_the_rejected_strategy(tmp_path):
    from test_graph_mocked import strat

    from agenticsdk.agents.schemas import Verdict
    retry = Verdict(decision="RETRY_STRATEGY", reasoning="forest too weak", feedback="try boosting")
    backend = ToolScript([CALL_PROFILE, DONE, DONE, CALL_CV, DONE, DONE], [fence(GOOD), fence(GOOD)], [retry, PASS],
                         strategies=[strat("linear", "minimal", "OLD_STRATEGY_MARKER"),
                                     strat("gradient_boosting", "minimal", "new")])
    llm = LLMClient(run_dir=tmp_path / "run", cache_dir=tmp_path / "cache", backend=backend, rpm=0, sleep=lambda s: None)
    g = build_graph(llm, tmp_path / "run", tools=FakeTools())
    out = AgentState.model_validate(g.invoke(initial_state("credit_g", 0, "t", 2, 60, use_tools=True),
                                             config={"recursion_limit": 100}))
    assert len(out.strategy_history) == 2
    explore = [p for k, p in backend.seen if k == "tool_step" and "about to design the modelling strategy" in p]
    assert "EARLIER STRATEGIES" not in explore[0]
    assert any("EARLIER STRATEGIES" in p and "OLD_STRATEGY_MARKER" in p and "forest too weak" in p for p in explore[1:])
