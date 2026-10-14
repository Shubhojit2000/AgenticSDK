"""Phase 2: strategy escalation, budget enforcement and resumability (scripted fake LLM, real sandbox)."""
import json
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from test_graph_mocked import CRASH, GOOD, PASS, Script, fence, run, strat

from agenticsdk.agents.graph import build_graph
from agenticsdk.agents.run import initial_state
from agenticsdk.agents.schemas import AgentState, Verdict
from agenticsdk.llm import LLMClient

RETRY_STRAT = Verdict(decision="RETRY_STRATEGY", reasoning="linear model cannot fit this", feedback="try boosting")
RETRY_CODE = Verdict(decision="RETRY_CODE", reasoning="tune", feedback="tune more")
LINEAR = strat("linear", "minimal", "plain logistic regression")
BOOST = strat("gradient_boosting", "interactions_polynomial", "lightgbm with interactions")

CONST = """
import numpy as np, pandas as pd
tr = pd.read_csv("train.csv"); y = tr["target"].astype(str)
classes = sorted(y.unique()); prior = y.value_counts(normalize=True)
for name in ("val", "test"):
    X = pd.read_csv(f"{name}_features.csv")
    p = pd.DataFrame({c: prior[c] for c in classes}, index=range(len(X)))
    p.insert(0, "id", np.arange(len(X))); p.to_csv(f"predictions_{name}.csv", index=False)
"""


def test_retry_strategy_goes_back_to_planner_and_new_strategy_must_differ(tmp_path):
    final, backend, run_dir = run(tmp_path, [fence(GOOD), fence(GOOD)], [RETRY_STRAT, PASS],
                                  strategies=[LINEAR, BOOST])
    assert final.status == "passed" and len(final.strategy_history) == 2
    assert [a.strategy_index for a in final.attempts] == [1, 2]
    assert final.attempts[0].verdict_decision == "RETRY_STRATEGY" and final.strategy_retry_count == 1
    assert final.strategy_history[0].key != final.strategy_history[1].key
    planner_prompts = [p for k, p in backend.seen if k == "Strategy"]
    assert "EARLIER STRATEGIES" not in planner_prompts[0]
    assert "EARLIER STRATEGIES" in planner_prompts[1] and "plain logistic regression" in planner_prompts[1]
    assert "linear model cannot fit this" in planner_prompts[1]          # the Critic's reasoning reaches the planner
    # the second strategy starts a fresh script: the old code is not shown to the coding agent
    training_prompts = [p for k, p in backend.seen if k == "text"]
    assert "The previous script was" not in training_prompts[1]
    assert "do not reuse the earlier approach" in training_prompts[1]
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [e["node"] for e in events].count("feature") == 2
    assert any(e["node"] == "critic" and e["data"]["decision"] == "RETRY_STRATEGY" for e in events)


def test_repeated_strategy_is_rejected_and_the_planner_is_asked_again(tmp_path):
    final, backend, run_dir = run(tmp_path, [fence(GOOD), fence(GOOD)], [RETRY_STRAT, PASS],
                                  strategies=[LINEAR, LINEAR, BOOST])
    assert final.status == "passed" and final.strategy_history[1].key == BOOST.key
    planner_prompts = [p for k, p in backend.seen if k == "Strategy"]
    assert len(planner_prompts) == 3 and "was rejected" in planner_prompts[2]
    assert "repeats strategy 1" in planner_prompts[2]
    assert "proposal rejected" in (run_dir / "events.jsonl").read_text()


def test_planner_that_never_produces_a_novel_strategy_ends_the_run_cleanly(tmp_path):
    final, _, _ = run(tmp_path, [fence(GOOD)], [RETRY_STRAT], strategies=[LINEAR])
    assert final.status == "exhausted" and len(final.strategy_history) == 1 and final.final_test_score is not None


def test_first_family_constraint_is_enforced_in_code(tmp_path):
    final, backend, _ = run(tmp_path, [fence(GOOD)], [PASS], strategies=[BOOST, LINEAR], first_family="linear")
    assert final.strategy_history[0].model_family == "linear"
    assert len([1 for k, _ in backend.seen if k == "Strategy"]) == 2   # the first proposal was refused


def test_critic_cannot_exceed_the_strategy_change_budget(tmp_path):
    # asks for a new strategy but none are allowed: coerced to a code retry, and the override is visible
    final, _, _ = run(tmp_path, [fence(GOOD), fence(GOOD)], [RETRY_STRAT, PASS], max_strategy_retries=0)
    assert len(final.strategy_history) == 1
    assert final.attempts[0].verdict_decision == "RETRY_CODE" and final.attempts[0].verdict_source == "override"


def test_exhausted_code_retries_escalate_to_a_new_strategy(tmp_path):
    final, _, _ = run(tmp_path, [fence(GOOD), fence(GOOD), fence(GOOD)], [RETRY_CODE, RETRY_CODE, PASS],
                      retries=1, strategies=[LINEAR, BOOST])
    assert [a.verdict_decision for a in final.attempts[:2]] == ["RETRY_CODE", "RETRY_STRATEGY"]
    assert final.attempts[1].verdict_source == "override" and final.attempts[2].strategy_index == 2
    assert final.code_retry_count == 0  # reset by the new strategy


def test_a_strategy_that_only_crashes_escalates_instead_of_retrying_forever(tmp_path):
    final, _, _ = run(tmp_path, [fence(CRASH), fence(GOOD)], [PASS], retries=0, strategies=[LINEAR, BOOST])
    assert final.attempts[0].verdict_decision == "RETRY_STRATEGY" and final.attempts[0].verdict_source == "harness"
    assert final.status == "passed" and final.best_attempt == 2


def test_total_attempt_cap_stops_the_run(tmp_path):
    final, _, _ = run(tmp_path, [fence(GOOD)] * 3, [RETRY_CODE] * 3, retries=5, max_total_attempts=3)
    assert len(final.attempts) == 3 and final.status == "exhausted"


def test_best_attempt_is_chosen_across_strategies_by_validation_score(tmp_path):
    final, _, _ = run(tmp_path, [fence(GOOD), fence(CONST)], [RETRY_STRAT, RETRY_STRAT],
                      strategies=[LINEAR, BOOST], max_strategy_retries=1, retries=0)
    assert final.best_attempt == 1  # strategy 1's script was better on validation; selection ignores order


def test_resume_after_a_crash_does_not_redo_finished_nodes(tmp_path):
    class Boom(Script):
        def __call__(self, model, system, prompt, schema, thinking_budget, temperature):
            if schema is None and not self.trainings:          # the training call after the crash point
                raise RuntimeError("simulated crash")
            return super().__call__(model, system, prompt, schema, thinking_budget, temperature)

    run_dir, db = tmp_path / "run", tmp_path / "run" / "ck.sqlite"
    run_dir.mkdir()
    cfg = {"configurable": {"thread_id": "t"}, "recursion_limit": 100}
    b1 = Boom([fence(GOOD)], [RETRY_CODE])
    conn = sqlite3.connect(db, check_same_thread=False)
    g1 = build_graph(LLMClient(run_dir=run_dir, cache_dir=tmp_path / "c1", backend=b1, sleep=lambda s: None),
                     run_dir, checkpointer=SqliteSaver(conn))
    with pytest.raises(RuntimeError, match="simulated crash"):
        g1.invoke(initial_state("credit_g", 0, "t", 2, 60), config=cfg)
    assert [k for k, _ in b1.seen].count("Strategy") == 1 and len(g1.get_state(cfg).values["attempts"]) == 1
    conn.close()

    b2 = Script([fence(GOOD)], [PASS])
    conn = sqlite3.connect(db, check_same_thread=False)
    g2 = build_graph(LLMClient(run_dir=run_dir, cache_dir=tmp_path / "c2", backend=b2, sleep=lambda s: None),
                     run_dir, checkpointer=SqliteSaver(conn))
    final = AgentState.model_validate(g2.invoke(None, config=cfg))
    conn.close()
    kinds = [k for k, _ in b2.seen]
    assert final.status == "passed" and len(final.attempts) == 2
    assert "Strategy" not in kinds and "FeaturePlan" not in kinds
