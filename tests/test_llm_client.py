import json

import pytest
from pydantic import BaseModel

from agenticsdk.llm import CacheMiss, LLMClient, LLMOutputError, QuotaExhausted, RawResult


class Out(BaseModel):
    answer: str


class FakeBackend:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def __call__(self, model, system, prompt, schema, thinking_budget, temperature):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make(tmp_path, backend, **kw):
    sleeps = []
    kw.setdefault("rpm", 0)                      # no throttling unless a test asks for it
    kw.setdefault("model", "gemini-2.5-flash")   # price table entry the cost assertions rely on
    c = LLMClient(run_dir=tmp_path / "run", cache_dir=tmp_path / "cache", backend=backend,
                  sleep=sleeps.append, **kw)
    return c, sleeps


def ok(text='{"answer": "42"}', i=10, o=20):
    return RawResult(text, i, o)


def test_structured_parses_and_accounts(tmp_path):
    c, _ = make(tmp_path, FakeBackend([ok(i=1000, o=2000)]))
    assert c.structured(Out, "sys", "q", "planner").answer == "42"
    u = c.usage()
    assert u["api_calls"] == 1 and u["input_tokens"] == 1000 and u["output_tokens"] == 2000
    assert u["cost_usd_live"] == pytest.approx((1000 * 0.30 + 2000 * 2.50) / 1e6)


def test_second_identical_call_hits_cache(tmp_path):
    b = FakeBackend([ok()])
    c, _ = make(tmp_path, b)
    c.structured(Out, "sys", "q", "planner")
    c.structured(Out, "sys", "q", "planner")
    assert b.calls == 1 and c.usage()["cache_hits"] == 1
    assert c.usage()["cost_usd_replayed"] > 0 and c.usage()["cost_usd_live"] > 0


def test_replay_only_serves_from_cache_with_zero_api_calls(tmp_path):
    c, _ = make(tmp_path, FakeBackend([ok()]))
    c.structured(Out, "sys", "q", "planner")
    b2 = FakeBackend([])  # any API call would raise IndexError
    c2, _ = make(tmp_path, b2, mode="replay_only")
    assert c2.structured(Out, "sys", "q", "planner").answer == "42"
    assert b2.calls == 0


def test_replay_only_raises_on_miss(tmp_path):
    c, _ = make(tmp_path, FakeBackend([]), mode="replay_only")
    with pytest.raises(CacheMiss):
        c.text("sys", "never seen", "training")


def test_different_prompt_or_system_or_schema_is_a_different_key(tmp_path):
    b = FakeBackend([ok(), ok(), ok('plain')])
    c, _ = make(tmp_path, b)
    c.structured(Out, "sys", "q1", "a")
    c.structured(Out, "sys2", "q1", "a")
    c.text("sys", "q1", "a")
    assert b.calls == 3


def test_backoff_on_429_then_success(tmp_path):
    b = FakeBackend([RuntimeError("429 RESOURCE_EXHAUSTED"), RuntimeError("503 UNAVAILABLE"), ok()])
    c, sleeps = make(tmp_path, b, base_delay=2.0)
    assert c.structured(Out, "s", "q", "a").answer == "42"
    assert b.calls == 3 and len(sleeps) == 2
    assert sleeps[1] > sleeps[0] * 0.9  # exponential growth (jitter keeps it within +-25%)
    assert c.usage()["retries"] == 2


def test_server_retry_hint_is_honoured(tmp_path):
    b = FakeBackend([RuntimeError("429 ... Please retry in 17.5s."), ok()])
    c, sleeps = make(tmp_path, b)
    c.structured(Out, "s", "q", "a")
    assert sleeps[0] >= 17.5


def test_daily_quota_raises_immediately_instead_of_sleeping_for_hours(tmp_path):
    msg = "429 You exceeded your current quota ... limit: 20, model: gemini-2.5-flash Please retry in 5h43m13.87243468s."
    b = FakeBackend([RuntimeError(msg)])
    c, sleeps = make(tmp_path, b)
    with pytest.raises(QuotaExhausted):
        c.text("s", "q", "a")
    assert b.calls == 1 and not sleeps


def test_per_day_quota_id_is_not_retried_even_without_a_hint(tmp_path):
    b = FakeBackend([RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")])
    c, sleeps = make(tmp_path, b)
    with pytest.raises(QuotaExhausted):
        c.text("s", "q", "a")
    assert not sleeps


def test_non_retryable_error_propagates_immediately(tmp_path):
    b = FakeBackend([ValueError("400 INVALID_ARGUMENT: bad request")])
    c, sleeps = make(tmp_path, b)
    with pytest.raises(ValueError):
        c.text("s", "q", "a")
    assert b.calls == 1 and not sleeps


def test_gives_up_after_max_retries(tmp_path):
    b = FakeBackend([RuntimeError("429")] * 10)
    c, sleeps = make(tmp_path, b, max_retries=3)
    with pytest.raises(RuntimeError):
        c.text("s", "q", "a")
    assert b.calls == 4 and len(sleeps) == 3


def test_malformed_structured_output_raises_and_is_not_cached_as_success(tmp_path):
    b = FakeBackend([ok('{"wrong": 1}')])
    c, _ = make(tmp_path, b)
    with pytest.raises(LLMOutputError):
        c.structured(Out, "s", "q", "a")


def test_calls_are_logged_as_jsonl(tmp_path):
    c, _ = make(tmp_path, FakeBackend([ok()]))
    c.structured(Out, "s", "q", "planner")
    lines = (tmp_path / "run" / "llm_calls.jsonl").read_text().strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["agent"] == "planner" and rec["cached"] is False and rec["input_tokens"] == 10


def test_client_side_throttle_spaces_live_calls_but_not_cache_hits(tmp_path):
    b = FakeBackend([ok('{"answer": "1"}'), ok('{"answer": "2"}')])
    c, sleeps = make(tmp_path, b, rpm=10)          # one call per 6s (+15% margin)
    c.structured(Out, "s", "q1", "a")
    assert not sleeps                              # first call is immediate
    c.structured(Out, "s", "q2", "a")
    assert len(sleeps) == 1 and sleeps[0] > 6.0    # second live call must wait
    c.structured(Out, "s", "q1", "a")              # cache hit: no wait, no API call
    assert len(sleeps) == 1 and b.calls == 2
