"""Phase 5: experience cards, the Chroma store, leave-one-dataset-out, and memory inside the graph
(scripted fake LLM, hashed-word embeddings: no API key, no network beyond the cached OpenML datasets)."""
import json
import sys

import pytest
from test_escalation import BOOST, LINEAR, RETRY_STRAT
from test_graph_mocked import GOOD, PASS, fence, run
from test_tool_loop import CALL_CV, CALL_PROFILE, DONE, FakeTools, ToolScript

from agenticsdk import memory
from agenticsdk.agents.graph import build_graph
from agenticsdk.agents.run import initial_state, run_dataset
from agenticsdk.agents.schemas import AgentState, CitedStrategy, Strategy
from agenticsdk.llm import LLMClient, RawResult, ToolStep
from agenticsdk.memory import ExperienceStore, GeminiEmbedder, HashEmbedder, MemoryTools, build_card
from agenticsdk.tools.client import MCPToolClient, ToolSet


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    """A real card, condensed from a mocked two-strategy run (linear rejected, boosting passed)."""
    tmp = tmp_path_factory.mktemp("cardrun")
    final, _, run_dir = run(tmp, [fence(GOOD), fence(GOOD)], [RETRY_STRAT, PASS], strategies=[LINEAR, BOOST])
    assert final.status == "passed"
    return build_card(run_dir)


def other(card, dataset, run_id):
    return card.model_copy(update={"dataset_key": dataset, "run_id": run_id, "card_id": f"{dataset}:{run_id}"})


# ---- the card -------------------------------------------------------------------------------------
def test_card_condenses_strategies_failures_and_the_critics_reason(card):
    assert card.dataset_key == "credit_g" and card.status == "passed" and len(card.strategies) == 2
    s1, s2 = card.strategies
    assert (s1.model_family, s1.verdict) == ("linear", "RETRY_STRATEGY")
    assert "linear model cannot fit this" in s1.critic_note
    assert s2.verdict == "PASS" and card.best_strategy in (1, 2)   # identical scripts tie, so either may win
    assert card.meta["rows"] == 600 and card.meta["classes"] == 2
    text = card.render()
    assert text.startswith("CARD credit_g:t") and "Strategy 2: gradient_boosting" in text


def test_card_never_renders_or_embeds_the_test_side_score(card):
    assert card.final_test_normalized is not None          # kept in the file for analysis ...
    text = card.render()
    assert str(card.final_test_normalized)[:6] not in text and "test" not in text.lower()


def test_run_that_predates_strategy_tracking_is_refused(tmp_path):
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "report.json").write_text(json.dumps({"attempts": [{"index": 1}], "strategy_history": []}), "utf-8")
    with pytest.raises(ValueError, match="predates strategy tracking"):
        build_card(tmp_path)


# ---- the store and the leakage rule ------------------------------------------------------------------
def test_store_search_roundtrip_ranks_by_similarity_and_upserts(tmp_path, card):
    store = ExperienceStore(tmp_path / "chroma", HashEmbedder())
    cards = [other(card, "eucalyptus", "r1"), other(card, "moneyball", "r2"), other(card, "diamonds", "r3")]
    for c in cards:
        store.add(c)
    store.add(cards[0])                                      # same id again: replaced, not duplicated
    assert len(store) == 3
    hits = store.search(cards[1].render(), k=3)
    assert hits[0][0].card_id == "moneyball:r2" and hits[0][1] > 0.99
    assert len(store.search("anything", k=2)) == 2


def test_search_never_returns_an_excluded_dataset(tmp_path, card):
    store = ExperienceStore(tmp_path / "chroma", HashEmbedder())
    for ds in ("eucalyptus", "moneyball", "credit_g"):
        store.add(other(card, ds, "r"))
    exact = store.cards()[0].render()                        # a query that matches an excluded card verbatim
    assert "credit_g" not in {c.dataset_key for c, _ in store.search(exact, 5, exclude_datasets={"credit_g"})}
    everything = {c.dataset_key for c, _ in store.search(exact, 5, exclude_datasets={"credit_g", "moneyball"})}
    assert everything == {"eucalyptus"}


def test_memory_tool_hides_the_current_dataset_whatever_the_query_says(tmp_path, card):
    store = ExperienceStore(tmp_path / "chroma", HashEmbedder())
    store.add(other(card, "credit_g", "mine"))
    tools = MemoryTools(store, "credit_g", 0)
    assert "No experience cards" in tools.search_experience("credit_g:mine exact copy of the card", 5)
    store.add(other(card, "moneyball", "r2"))
    out = tools.search_experience("anything", 99)            # k is clamped
    assert "CARD moneyball:r2" in out and "credit_g:mine" not in out and out.count("CARD ") <= MemoryTools.MAX_K


def test_ingest_cli_skips_unseen_datasets_without_the_flag(tmp_path, monkeypatch, capsys, card):
    run_dir = tmp_path / "runs" / "adult_run"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(memory, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(memory, "build_card", lambda d: other(card, "adult", "adult_run"))
    argv = ["memory", "--memory-dir", str(tmp_path / "chroma"), "--embedder", "hash", "ingest", "adult_run"]
    monkeypatch.setattr(sys, "argv", argv)
    memory.main()
    assert "final-unseen" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", argv + ["--allow-unseen"])
    memory.main()
    assert "added adult:adult_run" in capsys.readouterr().out


# ---- embeddings ---------------------------------------------------------------------------------
def test_gemini_embedder_caches_by_text_and_task(tmp_path):
    calls = []

    class Counting(GeminiEmbedder):
        def _embed_one(self, text, kind):
            calls.append((text, kind))
            return [1.0, 0.0]

    e = Counting(cache_dir=tmp_path)
    assert e.embed(["a", "b"]) == [[1.0, 0.0]] * 2 and e.embed(["a"]) == [[1.0, 0.0]]
    assert len(calls) == 2
    e.embed(["a"], "query")                                   # a different task type is a different embedding
    assert len(calls) == 3


# ---- tools: routing and the real MCP server -----------------------------------------------------------
def test_toolset_routes_calls_to_the_owning_server():
    class C:
        def __init__(self, names):
            self.specs = [{"name": n} for n in names]
            self.started = self.closed = False

        def start(self):
            self.started = True

        def call(self, name, arguments=None):
            return f"{name}@{id(self)}"

        def close(self):
            self.closed = True

    a, b = C(["profile_dataset"]), C(["search_experience"])
    ts = ToolSet(a, b).start()
    assert [s["name"] for s in ts.specs] == ["profile_dataset", "search_experience"]
    assert ts.call("search_experience", {}).endswith(str(id(b))) and ts.call("nope").startswith("ERROR")
    ts.close()
    assert a.closed and b.closed


def test_memory_server_over_stdio_never_returns_the_current_dataset(tmp_path, card):
    store = ExperienceStore(tmp_path / "chroma", HashEmbedder())
    store.add(other(card, "credit_g", "mine"))
    store.add(other(card, "moneyball", "r2"))
    del store
    args = ["--memory-dir", str(tmp_path / "chroma"), "--embedder", "hash"]
    with MCPToolClient("credit_g", 0, server_args=args, module="agenticsdk.tools.memory_server") as c:
        assert [s["name"] for s in c.specs] == ["search_experience"]
        out = c.call("search_experience", {"query": "boosting", "k": 3})
    assert "CARD moneyball:r2" in out and "credit_g:mine" not in out


# ---- memory inside the graph ----------------------------------------------------------------------------
CALL_MEM = ToolStep(tool="search_experience", arguments={"query": "small tabular, binary target", "k": 2})
MEM_RESULT = "Experience cards ...\n\nCARD moneyball:r2  (similarity 0.80)\nStrategy 2: ...\n\nCARD diamonds:r3 (similarity 0.5)\n"


class MemTools(FakeTools):
    specs = FakeTools.specs + [{"name": "search_experience", "description": "d",
                                "parameters": {"type": "object", "properties": {}}}]

    def call(self, name, arguments):
        if name == "search_experience":
            self.calls.append((name, arguments))
            return MEM_RESULT
        return super().call(name, arguments)


class MemScript(ToolScript):
    """Also answers the CitedStrategy planner schema; records which tools each tool-step was offered."""

    def __init__(self, cited, *a, **kw):
        super().__init__(*a, **kw)
        self.cited, self.offered, self.systems = cited, [], []

    def __call__(self, model, system, prompt, schema, thinking_budget, temperature, tools=None):
        if tools:
            self.offered.append([t["name"] for t in tools])
            self.systems.append(system)
        if schema is CitedStrategy:
            s = CitedStrategy(**json.loads(self.strategies[0]), cited_cards=self.cited)
            self.seen.append(("CitedStrategy", prompt))
            return RawResult(s.model_dump_json(), 1, 1)
        return super().__call__(model, system, prompt, schema, thinking_budget, temperature, tools)


def run_mem(tmp_path, tool_steps, cited, tools=None, **kw):
    backend = MemScript(cited, tool_steps, [fence(GOOD)], [PASS])
    llm = LLMClient(run_dir=tmp_path / "run", cache_dir=tmp_path / "cache", backend=backend, rpm=0, sleep=lambda s: None)
    g = build_graph(llm, tmp_path / "run", tools=tools or MemTools())
    st = initial_state("credit_g", 0, "t", 2, 60, use_tools=True, use_memory=True, **kw)
    return AgentState.model_validate(g.invoke(st, config={"recursion_limit": 100})), backend, tmp_path / "run"


def test_planner_searches_memory_and_valid_citations_are_recorded(tmp_path):
    tools = MemTools()
    final, backend, run_dir = run_mem(tmp_path, [CALL_MEM, DONE], ["moneyball:r2"], tools)
    assert final.status == "passed" and type(final.strategy_history[0]) is Strategy
    assert [c[0] for c in tools.calls] == ["search_experience"]
    assert final.memory_log == [{"strategy_index": 1, "retrieved": ["moneyball:r2", "diamonds:r3"],
                                 "cited": ["moneyball:r2"], "dropped": []}]
    assert final.tool_log[0]["cards"] == ["moneyball:r2", "diamonds:r3"]
    ev = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert next(e for e in ev if e["node"] == "planner")["data"]["memory"]["cited"] == ["moneyball:r2"]
    assert any(k == "CitedStrategy" for k, _ in backend.seen)    # the memory-aware schema was used


def test_a_citation_of_a_card_that_was_never_retrieved_is_dropped_and_logged(tmp_path):
    final, _, _ = run_mem(tmp_path, [CALL_MEM, DONE], ["eucalyptus:made_up", "diamonds:r3"])
    assert final.memory_log[0]["cited"] == ["diamonds:r3"] and final.memory_log[0]["dropped"] == ["eucalyptus:made_up"]


def test_memory_search_has_its_own_allowance_and_is_offered_only_to_the_planner(tmp_path):
    # the planner spends its 3 measurements and still gets its one memory search; the feature agent gets none
    steps = [CALL_PROFILE, CALL_CV, CALL_PROFILE, CALL_MEM, DONE]
    final, backend, _ = run_mem(tmp_path, steps, [])
    names = [t["tool"] for t in final.tool_log]
    assert names.count("search_experience") == 1 and names.count("profile_dataset") + names.count("run_quick_cv") == 3
    assert backend.offered[3] == ["search_experience"]        # data budget spent: only the memory tool is offered
    assert all("search_experience" not in o for o in backend.offered[-1:])   # feature agent: not offered
    assert "search_experience" in backend.systems[0]


def test_memory_off_changes_nothing_for_the_planner(tmp_path):
    backend = ToolScript([CALL_PROFILE, DONE], [fence(GOOD)], [PASS])
    llm = LLMClient(run_dir=tmp_path / "run", cache_dir=tmp_path / "cache", backend=backend, rpm=0, sleep=lambda s: None)
    final = AgentState.model_validate(build_graph(llm, tmp_path / "run", tools=FakeTools()).invoke(
        initial_state("credit_g", 0, "t", 2, 60, use_tools=True), config={"recursion_limit": 100}))
    assert final.memory_log == [] and "Experience memory" not in json.dumps(backend.seen)


def test_use_memory_requires_use_tools():
    with pytest.raises(ValueError, match="needs use_tools"):
        run_dataset("credit_g", 0, "no_such_run", use_memory=True)


def test_search_returns_one_card_per_dataset(tmp_path, card):
    store = ExperienceStore(tmp_path / "chroma", HashEmbedder())
    for ds, rid in (("eucalyptus", "r1"), ("eucalyptus", "r2"), ("eucalyptus", "r3"), ("moneyball", "r4")):
        store.add(other(card, ds, rid))
    hits = store.search(store.cards()[0].render(), k=3)
    assert sorted(c.dataset_key for c, _ in hits) == ["eucalyptus", "moneyball"]   # not 3 euc seeds
