"""Schema and prompt unit tests: structure, determinism, and that hidden scores never reach a prompt."""
import tempfile

import pytest
from pydantic import ValidationError

from agenticsdk.agents import prompts
from agenticsdk.agents.run import initial_state
from agenticsdk.agents.schemas import AgentState, Attempt, ExecSummary, FeaturePlan, Strategy, Verdict

TEST_ONLY_B0, TEST_ONLY_B1, TEST_ONLY_B2 = -7.31415, -6.27182, -5.16180


def strategy(**kw):
    base = dict(model_family="linear", feature_approach="minimal", summary="s", preprocessing=["p"],
                model="m", validation="v", risks=["r"])
    return Strategy(**{**base, **kw})


def state(**kw):
    s = initial_state("credit_g", 0, "t", 2, 60, **kw)
    s.baselines_test = {"B0": TEST_ONLY_B0, "B1": TEST_ONLY_B1, "B2": TEST_ONLY_B2}
    s.strategy_history = [strategy()]
    s.current_strategy = s.strategy_history[0]
    s.feature_plan = FeaturePlan(steps=["a"], drop_columns=[], rationale="r")
    return s


def attempt(i=1, val=-0.5, error=None):
    return Attempt(index=i, code="print(1)", exec=ExecSummary(exit_code=0, stdout="cv -0.51"), val_valid=error is None,
                   val_score=None if error else val, error=error)


# ---- schemas
def test_verdict_accepts_exactly_three_decisions():
    for d in ("PASS", "RETRY_CODE", "RETRY_STRATEGY"):
        assert Verdict(decision=d, reasoning="r", feedback="f").decision == d
    with pytest.raises(ValidationError):
        Verdict(decision="MAYBE", reasoning="r", feedback="f")


def test_strategy_requires_known_family_and_approach_and_exposes_a_key():
    assert strategy().key == ("linear", "minimal")
    with pytest.raises(ValidationError):
        strategy(model_family="transformer")
    with pytest.raises(ValidationError):
        strategy(feature_approach="magic")


def test_agent_state_defaults_are_the_documented_budgets():
    s = initial_state("credit_g", 0, "t", 2, 60)
    assert (s.max_code_retries, s.max_strategy_retries, s.max_total_attempts) == (2, 2, 6)
    assert s.use_tools is False and (s.planner_tool_budget, s.feature_tool_budget) == (3, 2)
    assert s.status == "running" and s.strategy_history == [] and s.attempts == []


def test_state_round_trips_through_json_for_the_checkpointer():
    s = state()
    s.attempts = [attempt()]
    again = AgentState.model_validate_json(s.model_dump_json())
    assert again == s


# ---- prompts: determinism and integrity
def test_prompts_are_deterministic_and_free_of_machine_specific_text():
    s = state()
    s.attempts = [attempt(error="boom")]
    texts = [prompts.planner_prompt(s), prompts.feature_prompt(s), prompts.training_prompt(s),
             prompts.critic_prompt(s, attempt(2))]
    assert texts == [prompts.planner_prompt(s), prompts.feature_prompt(s), prompts.training_prompt(s),
                     prompts.critic_prompt(s, attempt(2))]
    tmp = tempfile.gettempdir().replace("\\", "/").lower()
    for t in texts:
        assert tmp not in t.replace("\\", "/").lower() and "run_id" not in t


@pytest.mark.parametrize("use_tools", [False, True])
def test_test_scores_never_appear_in_any_prompt(use_tools):
    s = state(use_tools=use_tools)
    s.attempts = [attempt(1), attempt(2, val=-0.4)]
    s.verdict = Verdict(decision="RETRY_CODE", reasoning="r", feedback="f")
    steps = [{"tool": "profile_dataset", "arguments": {}, "result": "rows 600"}]
    texts = [prompts.task_block(s), prompts.planner_prompt(s), prompts.feature_prompt(s), prompts.training_prompt(s),
             prompts.critic_prompt(s, s.attempts[-1]), prompts.explore_prompt(s, "planner", steps, 2),
             prompts.explore_prompt(s, "feature", steps, 1), prompts.findings_text(steps, "notes")]
    blob = " ".join(texts)
    for marker in ("7.31415", "6.27182", "5.16180", "__target__"):
        assert marker not in blob


def test_validation_baselines_do_reach_the_critic():
    s = state()
    s.baselines_val = {"B0": -0.5, "B1": -0.4, "B2": -0.3}
    assert "B0=-0.500000" in prompts.critic_prompt(s, attempt())


def test_tool_mode_hides_the_profile_from_the_planner_but_not_the_feature_agent():
    s = state(use_tools=True)
    s.profile = "rows (train): 600\nfeature columns: 20\n  - secret_column_name: numeric"
    planner = prompts.planner_prompt(s)
    assert "secret_column_name" not in planner and "The full profile is not shown" in planner
    assert "secret_column_name" in prompts.feature_prompt(s)
    assert "secret_column_name" in prompts.planner_prompt(state(use_tools=False).model_copy(update={"profile": s.profile}))


def test_a_new_strategy_prompt_does_not_show_the_old_script():
    s = state()
    s.strategy_history = [strategy(), strategy(model_family="gradient_boosting")]
    s.current_strategy = s.strategy_history[1]
    s.attempts = [attempt(1)]                       # strategy_index defaults to 1, current strategy is 2
    out = prompts.training_prompt(s)
    assert "The previous script was" not in out and "do not reuse the earlier approach" in out


# ---- extract_code
def test_extract_code_takes_the_longest_block_and_handles_missing_blocks():
    assert prompts.extract_code("no code here") is None
    text = "plan\n```python\nx = 1\n```\nmore\n```python\nx = 1\ny = 2\n```"
    assert prompts.extract_code(text) == "x = 1\ny = 2"
    assert prompts.extract_code("```py\nz = 3\n```") == "z = 3"
