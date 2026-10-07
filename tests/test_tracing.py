"""Phase 4: Langfuse tracing. The real Langfuse SDK emits the spans; an in-memory OpenTelemetry exporter
captures them, so the trace structure is verified without any Langfuse account or network."""
import json

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_graph_mocked import GOOD, PASS, Script, fence, strat
from test_tool_loop import CALL_CV, CALL_PROFILE, DONE, FakeTools, ToolScript

import agenticsdk.agents.run as graph_run
from agenticsdk.agents.graph import build_graph
from agenticsdk.agents.run import initial_state
from agenticsdk.agents.schemas import AgentState, Verdict
from agenticsdk.llm import LLMClient
from agenticsdk.tracing import LangfuseTracer, NoopTracer, from_env, set_tracer

RETRY_STRAT = Verdict(decision="RETRY_STRATEGY", reasoning="forest too weak", feedback="try boosting")


@pytest.fixture(scope="module")
def exporter():
    exp = InMemorySpanExporter()
    # Langfuse keeps one resource manager per public key, so create the client once per module
    client = Langfuse(public_key="pk-test", secret_key="sk-test", host="http://localhost:9", span_exporter=exp,
                      flush_at=1)
    exp.client = client
    return exp


@pytest.fixture
def traced(exporter):
    exporter.clear()
    tracer = LangfuseTracer(exporter.client)
    set_tracer(tracer)
    yield tracer, exporter
    exporter.client.flush()
    set_tracer(NoopTracer())


def spans_of(exporter):
    exporter.client.flush()
    spans = exporter.get_finished_spans()
    return {s.context.span_id: s for s in spans}, list(spans)


def attrs(span, key):
    v = span.attributes.get(f"langfuse.observation.{key}")
    try:
        return json.loads(v) if isinstance(v, str) else v
    except json.JSONDecodeError:
        return v


def name_of(by_id, span):
    return by_id[span.parent.span_id].name if span.parent and span.parent.span_id in by_id else None


def run_graph(tmp_path, tracer_exporter, backend, tools=None, use_tools=False, mode="cache", **state_kw):
    llm = LLMClient(run_dir=tmp_path / "run", cache_dir=tmp_path / "cache", backend=backend, rpm=0,
                    sleep=lambda s: None, mode=mode)
    g = build_graph(llm, tmp_path / "run", tools=tools)
    st = initial_state("credit_g", 0, "t", 2, 60, use_tools=use_tools, **state_kw)
    with get_root():
        return AgentState.model_validate(g.invoke(st, config={"recursion_limit": 100}))


def get_root():
    from agenticsdk.tracing import get_tracer
    return get_tracer().observe("agent_run", as_type="agent", input={"dataset": "credit_g"})


def test_span_tree_for_a_tool_using_run_with_escalation(tmp_path, traced):
    _, exp = traced
    backend = ToolScript([CALL_PROFILE, CALL_CV, DONE, DONE, CALL_PROFILE, DONE, DONE],
                         [fence(GOOD), fence(GOOD)], [RETRY_STRAT, PASS],
                         strategies=[strat("linear", "minimal", "lin"), strat("gradient_boosting", "minimal", "gb")])
    final = run_graph(tmp_path, exp, backend, tools=FakeTools(), use_tools=True)
    assert final.status == "passed" and len(final.strategy_history) == 2

    by_id, spans = spans_of(exp)
    roots = [s for s in spans if s.parent is None]
    assert [s.name for s in roots] == ["agent_run"]
    assert len({s.context.trace_id for s in spans}) == 1                       # one trace per run

    nodes = [s for s in spans if attrs(s, "type") == "chain"]
    assert {s.name for s in nodes} == {"planner", "feature", "training", "execute", "critic", "report"}
    assert all(name_of(by_id, s) == "agent_run" for s in nodes)
    assert [s.name for s in nodes].count("planner") == 2                       # re-planned after RETRY_STRATEGY

    gens = [s for s in spans if attrs(s, "type") == "generation"]
    assert {name_of(by_id, g) for g in gens} == {"planner", "feature", "training", "critic"}
    assert {g.name for g in gens} >= {"llm:planner", "llm:feature", "llm:training", "llm:critic"}
    g0 = next(g for g in gens if g.name == "llm:training")
    assert attrs(g0, "model.name") and attrs(g0, "usage_details") == {"input": 1, "output": 1}
    assert "cost_details" in {k.split("observation.")[1] for k in g0.attributes if "cost_details" in k}
    assert "prompt" in attrs(g0, "input") and attrs(g0, "model.parameters")["temperature"] == 0.0

    tools = [s for s in spans if attrs(s, "type") == "tool"]
    assert {t.name for t in tools} == {"tool:profile_dataset", "tool:run_quick_cv"}
    assert all(name_of(by_id, t) == "planner" for t in tools)
    assert "PROFILE_MARKER" in attrs(next(t for t in tools if t.name == "tool:profile_dataset"), "output")


def test_the_decisions_are_readable_from_the_spans_alone(tmp_path, traced):
    _, exp = traced
    backend = ToolScript([], [fence(GOOD), fence(GOOD)], [RETRY_STRAT, PASS],
                         strategies=[strat("linear", "minimal", "lin"), strat("gradient_boosting", "minimal", "gb")])
    run_graph(tmp_path, exp, backend)
    by_id, spans = spans_of(exp)
    critic = sorted((s for s in spans if s.name == "critic"), key=lambda s: s.start_time)
    outs = [attrs(s, "output") for s in critic]
    assert [o["decision"] for o in outs] == ["RETRY_STRATEGY", "PASS"]
    assert "forest too weak" in outs[0]["detail"]
    execute = sorted((s for s in spans if s.name == "execute"), key=lambda s: s.start_time)
    assert all("val_score" in attrs(s, "output") for s in execute)
    planner = [attrs(s, "output") for s in spans if s.name == "planner"]
    assert {o["strategy"]["model_family"] for o in planner} == {"linear", "gradient_boosting"}


def test_replayed_calls_are_marked_cached_and_not_double_counted(tmp_path, traced):
    _, exp = traced
    run_graph(tmp_path, exp, Script([fence(GOOD)], [PASS]))
    exp.clear()
    run_graph(tmp_path, exp, Script([], []), mode="replay_only")        # every call now comes from the cache
    _, spans = spans_of(exp)
    gens = [s for s in spans if attrs(s, "type") == "generation"]
    assert gens
    assert all(g.attributes["langfuse.observation.metadata.cached"] is True for g in gens)
    assert all(attrs(g, "usage_details") == {"input": 0, "output": 0} for g in gens)
    assert not any("langfuse.observation.cost_details" in g.attributes for g in gens)


def test_a_failing_tool_is_marked_as_an_error_span(tmp_path, traced):
    _, exp = traced
    run_graph(tmp_path, exp, ToolScript([CALL_PROFILE, DONE], [fence(GOOD)], [PASS]),
              tools=FakeTools(fail=TimeoutError("stalled")), use_tools=True)
    _, spans = spans_of(exp)
    tool = next(s for s in spans if s.name == "tool:profile_dataset")
    assert attrs(tool, "level") == "ERROR" and "stalled" in attrs(tool, "status_message")


def test_a_broken_tracing_backend_never_breaks_a_run(tmp_path):
    class Boom:
        def start_as_current_observation(self, **kw):
            raise RuntimeError("langfuse is down")

        def flush(self):
            raise RuntimeError("langfuse is down")

    set_tracer(LangfuseTracer(Boom()))
    llm = LLMClient(run_dir=tmp_path / "run", cache_dir=tmp_path / "c", backend=Script([fence(GOOD)], [PASS]),
                    rpm=0, sleep=lambda s: None)
    g = build_graph(llm, tmp_path / "run")
    out = AgentState.model_validate(g.invoke(initial_state("credit_g", 0, "t", 2, 60), config={"recursion_limit": 100}))
    assert out.status == "passed"


def test_an_exception_inside_a_span_is_recorded_and_re_raised(traced):
    tracer, exp = traced
    with pytest.raises(ValueError), tracer.observe("boom", as_type="span"):
        raise ValueError("bad thing")
    _, spans = spans_of(exp)
    s = next(s for s in spans if s.name == "boom")
    assert attrs(s, "level") == "ERROR" and "bad thing" in attrs(s, "status_message")


def test_run_dataset_opens_the_root_span_with_session_tags_and_final_result(tmp_path, traced, monkeypatch):
    _, exp = traced
    monkeypatch.setattr(graph_run, "RUNS_DIR", tmp_path / "runs")
    real = graph_run.LLMClient

    def fake_llm(run_dir, mode, model):
        return real(run_dir=run_dir, cache_dir=tmp_path / "cache", backend=Script([fence(GOOD)], [PASS]),
                    rpm=0, sleep=lambda s: None)

    monkeypatch.setattr(graph_run, "LLMClient", fake_llm)
    final = graph_run.run_dataset("credit_g", 0, "rid-1")
    assert final.status == "passed"
    _, spans = spans_of(exp)
    root = next(s for s in spans if s.parent is None)
    assert root.name == "agent_run"
    out = attrs(root, "output")
    assert out["status"] == "passed" and out["final_test_score"] == final.final_test_score
    assert out["llm_usage"]["api_calls"] >= 3
    assert root.attributes.get("session.id") == "rid-1" or "rid-1" in json.dumps(dict(root.attributes))
    assert "credit_g" in json.dumps(dict(root.attributes))


# ---------------------------------------------------------------- configuration
def test_no_keys_means_a_noop_tracer(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    assert isinstance(from_env(), NoopTracer) and not from_env().enabled


def test_tracing_can_be_switched_off_even_with_keys(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("AGENTICSDK_TRACING", "off")
    assert not from_env().enabled
