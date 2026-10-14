"""Phase 3: the tool logic, the MCP server over real stdio, and tool-step caching in the LLM client."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import agenticsdk.tools.dataset_tools as tools_mod
from agenticsdk.harness.datasets import SPLIT_DIR
from agenticsdk.llm import LLMClient, RawResult, ToolStep
from agenticsdk.tools.client import MCPToolClient
from agenticsdk.tools.dataset_tools import TaskTools, ToolError


@pytest.fixture(scope="module")
def credit():
    return TaskTools("credit_g", 0)


def test_profile_reports_checks_and_target_relationships(credit):
    out = credit.profile_dataset()
    assert "rows (train): 600" in out and "duplicate feature rows" in out and "constant columns" in out
    assert "|Spearman|" in out and "duration=" in out


def test_inspect_numeric_and_categorical_columns(credit):
    num = credit.inspect_column("duration")
    assert "numeric" in num and "IQR outliers" in num and "positive rate by quintile" in num
    cat = credit.inspect_column("checking_status")
    assert "categorical" in cat and "positive rate" in cat


def test_inspect_unknown_column_suggests_a_fix(credit):
    with pytest.raises(ToolError, match="Did you mean: duration"):
        credit.inspect_column("duratoin")


@pytest.mark.parametrize("kwargs, msg", [
    (dict(model_family="svm_rbf"), "not supported"),
    (dict(model_family="linear", params={"C": 1e9}), "must be a number in"),
    (dict(model_family="linear", params={"bogus": 1}), "not allowed"),
    (dict(model_family="linear", encoding="target"), "encoding must be"),
    (dict(model_family="linear", folds=50), "folds must be"),
])
def test_quick_cv_rejects_bad_specs_without_running_anything(credit, kwargs, msg):
    with pytest.raises(ToolError, match=msg):
        credit.run_quick_cv(**kwargs)


@pytest.mark.parametrize("family", ["linear", "bagged_trees", "gradient_boosting", "knn", "mlp"])
def test_quick_cv_runs_every_family_in_the_sandbox(credit, family):
    params = {"n_estimators": 20} if family in ("bagged_trees", "gradient_boosting") else {}
    out = credit.run_quick_cv(family, params, scale=True, folds=2)
    assert "neg_log_loss mean -" in out and "2-fold on 600 train rows" in out    # log-loss, not the stale roc_auc


def test_quick_cv_handles_regression_and_onehot_with_missing_indicators():
    t = TaskTools("moneyball", 0)
    out = t.run_quick_cv("bagged_trees", {"n_estimators": 20}, encoding="onehot", missing_indicators=True, folds=2)
    assert "r2 mean" in out


def test_tools_only_ever_open_the_train_split(tmp_path, monkeypatch):
    # a split directory that has no val/test files at all: the tools must still work
    d = tmp_path / "credit_g" / "seed0"
    d.mkdir(parents=True)
    for f in ("train.parquet", "meta.json"):
        shutil.copy(SPLIT_DIR / "credit_g" / "seed0" / f, d / f)
    monkeypatch.setattr(tools_mod, "make_splits", lambda spec, seed: d)
    t = TaskTools("credit_g", 0)
    assert "rows (train): 600" in t.profile_dataset()
    assert "neg_log_loss" in t.run_quick_cv("linear", folds=2)


def test_quick_cv_workdir_contains_only_train_csv(credit, monkeypatch):
    seen = []
    real = tools_mod.run_script

    def spy(code, workdir, timeout, screen=True):
        seen.append(sorted(p.name for p in Path(workdir).iterdir()))
        return real(code, workdir, timeout, screen)

    monkeypatch.setattr(tools_mod, "run_script", spy)
    credit.run_quick_cv("linear", folds=2)
    assert seen == [["train.csv"]]


def test_quick_cv_timeout_is_a_readable_error(credit, monkeypatch):
    monkeypatch.setattr(tools_mod, "CV_TIMEOUT_S", 1)
    with pytest.raises(ToolError, match="timed out"):
        credit.run_quick_cv("mlp", {"hidden": 256, "max_iter": 300}, folds=5)


# ---- the MCP server, over real stdio, as any MCP client sees it ------------------------------------
@pytest.fixture(scope="module")
def mcp_client():
    with MCPToolClient("credit_g", 0) as c:
        yield c


def test_server_lists_the_three_tools_with_schemas(mcp_client):
    by = {s["name"]: s for s in mcp_client.specs}
    assert set(by) == {"profile_dataset", "inspect_column", "run_quick_cv"}
    assert by["run_quick_cv"]["parameters"]["required"] == ["model_family"]
    assert by["inspect_column"]["parameters"]["properties"]["name"]["type"] == "string"
    assert all(s["description"] for s in by.values())
    assert "additionalProperties" not in json.dumps(by["run_quick_cv"]["parameters"])   # Gemini-safe


def test_server_calls_return_text_and_errors_are_not_exceptions(mcp_client):
    assert "rows (train): 600" in mcp_client.call("profile_dataset", {})
    assert mcp_client.call("inspect_column", {"name": "nope"}).startswith("ERROR: no column named")
    assert mcp_client.call("run_quick_cv", {"model_family": "linear", "params_json": "{not json"}).startswith("ERROR")
    ok = mcp_client.call("run_quick_cv", {"model_family": "linear", "folds": 2})
    assert "neg_log_loss mean" in ok


def test_server_environment_has_no_secrets(mcp_client):
    env = mcp_client.params.env
    assert not [k for k in env if any(w in k.upper() for w in ("KEY", "TOKEN", "SECRET", "GOOGLE", "GEMINI"))]


# ---- LLM client: tool steps are cacheable and do not disturb existing cache keys ---------------------
SPECS = [{"name": "profile_dataset", "description": "d", "parameters": {"type": "object", "properties": {}}}]


class ToolBackend:
    def __init__(self, replies):
        self.replies, self.calls, self.kwargs = list(replies), 0, []

    def __call__(self, model, system, prompt, schema, thinking_budget, temperature, tools=None):
        self.calls += 1
        self.kwargs.append(tools)
        return RawResult(self.replies.pop(0), 5, 5)


def make(tmp_path, backend, **kw):
    return LLMClient(run_dir=tmp_path / "run", cache_dir=tmp_path / "cache", backend=backend, rpm=0,
                     sleep=lambda s: None, **kw)


def test_tool_step_parses_call_and_text(tmp_path):
    b = ToolBackend([ToolStep(tool="profile_dataset", arguments={}).model_dump_json(),
                     ToolStep(text="done").model_dump_json()])
    c = make(tmp_path, b)
    s1 = c.tool_step("sys", "p1", SPECS, "planner")
    s2 = c.tool_step("sys", "p2", SPECS, "planner")
    assert (s1.tool, s2.tool, s2.text) == ("profile_dataset", None, "done") and b.kwargs[0] == SPECS


def test_tool_step_is_cached_and_replayable(tmp_path):
    b = ToolBackend([ToolStep(tool="profile_dataset").model_dump_json()])
    c = make(tmp_path, b)
    c.tool_step("sys", "p", SPECS, "planner")
    c.tool_step("sys", "p", SPECS, "planner")
    assert b.calls == 1 and c.usage()["cache_hits"] == 1
    r = make(tmp_path, ToolBackend([]), mode="replay_only")
    assert r.tool_step("sys", "p", SPECS, "planner").tool == "profile_dataset"


def test_tool_list_is_part_of_the_cache_key(tmp_path):
    b = ToolBackend([ToolStep(text="a").model_dump_json(), ToolStep(text="b").model_dump_json()])
    c = make(tmp_path, b)
    c.tool_step("sys", "p", SPECS, "x")
    c.tool_step("sys", "p", SPECS + [{"name": "other", "description": "", "parameters": {}}], "x")
    assert b.calls == 2


def test_existing_cache_keys_are_unchanged_by_the_tools_feature(tmp_path):
    c = make(tmp_path, ToolBackend([]))
    blob = json.dumps({"model": c.model, "system": "s", "prompt": "p", "schema": None, "thinking_budget": 7,
                       "temperature": 0.0}, sort_keys=True, ensure_ascii=False)
    assert c._key("s", "p", None, 7) == hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]
