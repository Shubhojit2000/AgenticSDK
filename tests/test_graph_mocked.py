"""End-to-end graph tests with a scripted fake LLM: no API calls, real sandbox, real scorer."""
import json

from agenticsdk.agents.graph import build_graph
from agenticsdk.agents.run import initial_state
from agenticsdk.agents.schemas import AgentState, FeaturePlan, Strategy, Verdict
from agenticsdk.llm import LLMClient, RawResult


def strat(family="bagged_trees", approach="encoding_focused", summary="RF on encoded features"):
    return Strategy(model_family=family, feature_approach=approach, summary=summary,
                    preprocessing=["ordinal-encode categoricals"], model="RandomForest",
                    validation="5-fold CV", risks=["time"])


STRATEGY = strat().model_dump_json()
FEATURES = FeaturePlan(steps=["ordinal-encode categoricals"], drop_columns=[], rationale="trees").model_dump_json()

GOOD = '''
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
tr = pd.read_csv("train.csv"); va = pd.read_csv("val_features.csv"); te = pd.read_csv("test_features.csv")
y = tr.pop("target").astype(str)
cats = [c for c in tr.columns if tr[c].dtype == object or str(tr[c].dtype) == "str"]
def enc(df):
    df = df.copy()
    for c in cats: df[c] = df[c].astype("category").cat.codes
    return df.fillna(-1)
allx = pd.concat([tr, va, te]);
for c in cats: allx[c] = allx[c].astype("category")
codes = {c: allx[c].cat.categories for c in cats}
def enc2(df):
    df = df.copy()
    for c in cats: df[c] = pd.Categorical(df[c], categories=codes[c]).codes
    return df.fillna(-1)
m = RandomForestClassifier(n_estimators=200, random_state=0).fit(enc2(tr), y)
classes = list(m.classes_)
for name, X in (("val", va), ("test", te)):
    p = pd.DataFrame(m.predict_proba(enc2(X)), columns=classes)
    p.insert(0, "id", np.arange(len(X)))
    p.to_csv(f"predictions_{name}.csv", index=False)
print("done")
'''

GOOD_FAST = GOOD.replace("n_estimators=200", "n_estimators=20")   # for tests with a tight wall-clock budget
CRASH = "import pandas as pd\nraise RuntimeError('boom in training')\n"
LOOP = "while True:\n    pass\n"
BLOCKED = "import os\nprint(os.listdir('.'))\n"
NO_PRED = "print('I forgot to write predictions')\n"


def fence(code):
    return f"Plan: do the thing.\n```python\n{code}\n```"


class Script:
    """Scripted backend: planner JSON, a queue of training replies, a queue of critic verdicts."""

    def __init__(self, trainings, verdicts, strategies=None):
        self.trainings, self.verdicts, self.seen = list(trainings), list(verdicts), []
        self.strategies = [s.model_dump_json() for s in strategies] if strategies else [STRATEGY]

    def __call__(self, model, system, prompt, schema, thinking_budget, temperature):
        self.seen.append((schema.__name__ if schema else "text", prompt))
        if schema is Strategy:   # the last scripted strategy repeats if the planner is asked more often
            return RawResult(self.strategies.pop(0) if len(self.strategies) > 1 else self.strategies[0], 1, 1)
        if schema is FeaturePlan:
            return RawResult(FEATURES, 1, 1)
        if schema is Verdict:
            return RawResult(self.verdicts.pop(0).model_dump_json(), 1, 1)
        return RawResult(self.trainings.pop(0), 1, 1)


def run(tmp_path, trainings, verdicts, retries=3, budget=60, key="credit_g", strategies=None, **kw):
    backend = Script(trainings, verdicts, strategies)
    llm = LLMClient(run_dir=tmp_path / "run", cache_dir=tmp_path / "cache", backend=backend, sleep=lambda s: None)
    graph = build_graph(llm, tmp_path / "run")
    state = initial_state(key, 0, "t", retries, budget, **kw)
    return AgentState.model_validate(graph.invoke(state, config={"recursion_limit": 100})), backend, tmp_path / "run"


PASS = Verdict(decision="PASS", reasoning="good", feedback="")


def test_crash_then_fix_then_pass(tmp_path):
    final, backend, run_dir = run(tmp_path, [fence(CRASH), fence(GOOD)], [PASS])
    assert final.status == "passed" and len(final.attempts) == 2
    a1, a2 = final.attempts
    assert a1.error and "boom in training" in a1.error and a1.verdict_source == "harness"
    assert a2.val_valid and a2.verdict_source == "llm"
    assert final.best_attempt == 2 and final.final_test_score is not None and final.final_normalized is not None
    # the crash message reached the training agent on the retry, and no tmp paths leaked into prompts
    retry_prompt = [p for k, p in backend.seen if k == "text"][1]
    assert "boom in training" in retry_prompt and str(tmp_path) not in retry_prompt
    assert (run_dir / "report.json").exists() and (run_dir / "events.jsonl").exists()


def test_infinite_loop_is_killed_and_reported_to_agent(tmp_path):
    final, backend, _ = run(tmp_path, [fence(LOOP), fence(GOOD_FAST)], [PASS], budget=12)
    assert final.attempts[0].exec.timed_out and "wall-clock limit" in final.attempts[0].error
    assert final.status == "passed"


def test_blocked_import_never_runs_and_is_explained(tmp_path):
    final, _, _ = run(tmp_path, [fence(BLOCKED), fence(GOOD)], [PASS])
    assert final.attempts[0].exec.blocked and "import of 'os'" in final.attempts[0].error


def test_reply_without_code_block_is_a_retryable_failure(tmp_path):
    final, _, _ = run(tmp_path, ["I think you should use random forests.", fence(GOOD)], [PASS])
    assert "no ```python code block" in final.attempts[0].error and final.status == "passed"


def test_missing_prediction_files_are_caught(tmp_path):
    final, _, _ = run(tmp_path, [fence(NO_PRED), fence(GOOD)], [PASS])
    assert "did not write" in final.attempts[0].error


def test_retry_budget_exhausted_without_any_valid_attempt_is_failed(tmp_path):
    final, _, _ = run(tmp_path, [fence(CRASH)] * 3, [], retries=2)
    assert final.status == "failed" and len(final.attempts) == 3 and final.final_test_score is None


def test_exhausted_but_valid_attempt_is_still_scored(tmp_path):
    retry = Verdict(decision="RETRY_CODE", reasoning="try harder", feedback="tune more")
    final, _, _ = run(tmp_path, [fence(GOOD), fence(GOOD)], [retry, retry], retries=1)
    assert final.status == "exhausted" and len(final.attempts) == 2 and final.final_test_score is not None


def test_critic_cannot_pass_a_score_below_the_weakest_baseline(tmp_path):
    # a constant-probability model is far below B0 on credit_g, so a PASS must be overridden
    const = '''
import numpy as np, pandas as pd
tr = pd.read_csv("train.csv"); y = tr["target"].astype(str)
classes = sorted(y.unique()); prior = y.value_counts(normalize=True)
for name in ("val", "test"):
    X = pd.read_csv(f"{name}_features.csv")
    p = pd.DataFrame({c: prior[c] for c in classes}, index=range(len(X)))
    p.insert(0, "id", np.arange(len(X))); p.to_csv(f"predictions_{name}.csv", index=False)
'''
    final, _, _ = run(tmp_path, [fence(const), fence(GOOD)], [PASS, PASS], retries=2)
    assert final.attempts[0].verdict_source == "override" and final.attempts[0].verdict_decision == "RETRY_CODE"
    assert final.best_attempt == 2


def test_test_labels_never_reach_the_agent_or_the_prompts(tmp_path):
    final, backend, _ = run(tmp_path, [fence(GOOD)], [PASS])
    assert final.status == "passed"
    blob = json.dumps(backend.seen)
    assert "__target__" not in blob and "test_labels" not in blob
