"""Phase 8: human-in-the-loop strategy approval (graph interrupt), with a scripted LLM and an in-memory checkpointer."""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agenticsdk.agents.graph import build_graph
from agenticsdk.agents.run import initial_state
from agenticsdk.agents.schemas import AgentState
from agenticsdk.llm import LLMClient
from tests.test_graph_mocked import GOOD, PASS, Script, fence, strat


def drive(tmp_path, decisions, strategies, **kw):
    backend = Script([fence(GOOD)] * 3, [PASS] * 3, strategies)
    llm = LLMClient(run_dir=tmp_path / "run", cache_dir=tmp_path / "cache", backend=backend, sleep=lambda s: None)
    graph = build_graph(llm, tmp_path / "run", checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}, "recursion_limit": 100}
    graph.invoke(initial_state("credit_g", 0, "t", 3, 60, hitl=True, **kw), config=cfg)
    proposals = []
    for d in decisions:
        pending = [i for t in graph.get_state(cfg).tasks for i in t.interrupts]
        assert pending, "the run should be paused waiting for a human"
        proposals.append(pending[0].value)
        graph.invoke(Command(resume=d), config=cfg)
    final = AgentState.model_validate(graph.get_state(cfg).values)
    return final, proposals, backend, graph.get_state(cfg)


def test_run_pauses_before_any_script_and_continues_after_approval(tmp_path):
    final, proposals, backend, snap = drive(tmp_path, [{"approve": True, "feedback": ""}], [strat()])
    assert proposals[0]["kind"] == "approve_strategy" and proposals[0]["strategy"]["model_family"] == "bagged_trees"
    assert final.status == "passed" and len(final.attempts) == 1 and not snap.tasks
    nodes = [e.node for e in final.trace]
    assert nodes.index("approval") < nodes.index("training")    # nothing was written before the human said yes


def test_rejection_removes_the_strategy_and_sends_the_reviewer_feedback_to_the_planner(tmp_path):
    second = strat("linear", "minimal", "plain logistic regression")
    final, proposals, backend, _ = drive(tmp_path, [{"approve": False, "feedback": "too slow, use something linear"},
                                                    {"approve": True, "feedback": ""}], [strat(), second])
    assert [p["strategy"]["model_family"] for p in proposals] == ["bagged_trees", "linear"]
    assert [s.model_family for s in final.strategy_history] == ["linear"] and final.human_rejections == 1
    planner_prompts = [p for n, p in backend.seen if n == "Strategy"]
    assert "too slow, use something linear" in planner_prompts[1] and "human reviewer" in planner_prompts[1]
    assert final.status == "passed"


def test_repeated_vetoes_end_the_run_instead_of_looping(tmp_path):
    no = {"approve": False, "feedback": "no"}
    strategies = [strat("linear", "minimal"), strat("knn_svm", "minimal"), strat("mlp", "minimal")]
    final, proposals, _, snap = drive(tmp_path, [no, no, no], strategies, max_human_rejections=3)
    assert final.status == "failed" and final.human_rejections == 3 and not final.attempts and not snap.tasks
