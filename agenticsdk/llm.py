"""Thin LLM client: backoff on rate limits, token/cost accounting, record/replay cache.

* **Backoff** on 429 / 5xx / timeouts with exponential delay + jitter (honours a server
  "retry in Ns" hint when the error carries one).
* **Accounting** per call: input/output tokens (output includes Gemini "thinking" tokens) and a
  USD-equivalent at list price, so cost is tracked even on the free tier.
* **Record/replay cache** keyed by a hash of (model, system, prompt, schema, thinking budget,
  temperature). Modes:
    ``cache``        use the cache, call the API on a miss and record the result (default)
    ``replay_only``  a miss raises ``CacheMiss``; proves a run needs zero API calls
    ``no_cache``     always call the API (never read or write the cache)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from agenticsdk.tracing import get_tracer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "llm_cache"
# Free-tier workhorse: 15 requests/min and 500 requests/day on the key used for development
# (the other free Gemini models are capped at 20 requests/day).
DEFAULT_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_S = 120

# USD per 1M tokens at list price (input, output). Check against Google's current price sheet
# before quoting a cost number in the report.
# Free-tier requests per minute per model (from the AI Studio rate-limit page); used for client-side throttling.
MODEL_RPM = {"gemini-3.5-flash-lite": 15, "gemini-3.1-flash-lite": 15, "gemini-2.5-flash-lite": 10,
             "gemini-2.5-flash": 5, "gemini-3-flash-preview": 5, "gemini-3.8-flash": 5,
             "gemma-4-26b-a4b-it": 30, "gemma-4-31b-it": 30}
# Free-tier tokens per minute where it is the binding limit (not enforced client-side yet; backoff handles 429s).
MODEL_TPM = {"gemma-4-26b-a4b-it": 16_000, "gemma-4-31b-it": 16_000}

PRICES = {
    "gemini-3.5-flash-lite": (0.10, 0.40),   # ASSUMED equal to 2.5 Flash-Lite; verify before quoting a cost
    "gemini-3.1-flash-lite": (0.10, 0.40),   # ASSUMED, same caveat. Gemma 4 (open weights) has no list price: 0.

    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
}

_RETRYABLE = re.compile(r"429|500|502|503|504|RESOURCE_EXHAUSTED|UNAVAILABLE|DEADLINE|overloaded|timed? ?out|"
                        r"temporarily|ServerError|ResourceExhausted|ServiceUnavailable", re.I)
_RETRY_HINT = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s|retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)s", re.I)
# A server-suggested wait longer than this means a daily/hard quota, not a per-minute rate limit.
MAX_SENSIBLE_WAIT_S = 300

T = TypeVar("T", bound=BaseModel)



class CacheMiss(RuntimeError):
    """Raised in ``replay_only`` mode when a call is not in the cache."""


class QuotaExhausted(RuntimeError):
    """A daily (or otherwise long-lived) quota is used up; retrying now is pointless."""


class LLMOutputError(RuntimeError):
    """The model answered, but not in the requested structure."""


@dataclass
class CallRecord:
    agent: str
    model: str
    key: str
    cached: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float          # USD-equivalent at list price (also filled for cache hits)
    latency_s: float
    retries: int = 0
    ts: float = field(default_factory=time.time)


@dataclass
class RawResult:
    content: str
    input_tokens: int
    output_tokens: int


class ToolStep(BaseModel):
    """One turn of a tool-using agent: the model either asks for ONE tool call or answers in text."""
    tool: str | None = None
    arguments: dict = {}
    text: str = ""


# (model, system, prompt, schema, thinking_budget, temperature) -> RawResult; backends that support
# native tool calling also accept a keyword-only ``tools`` list of {name, description, parameters}.
Backend = Callable[..., RawResult]


def _gemini_backend(model: str, system: str, prompt: str, schema, thinking_budget, temperature,
                    tools: list[dict] | None = None) -> RawResult:
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    load_dotenv(ROOT / ".env")
    # A hung request must not hang the whole run: fail after REQUEST_TIMEOUT_S and let our own
    # backoff loop (which treats timeouts as retryable) decide whether to try again.
    kwargs = {"model": model, "temperature": temperature, "timeout": REQUEST_TIMEOUT_S, "max_retries": 1}
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    llm = ChatGoogleGenerativeAI(**kwargs)
    msgs = ([SystemMessage(content=system)] if system else []) + [HumanMessage(content=prompt)]
    if tools:  # native function calling: the model emits a functionCall, or plain text when it is done
        raw = llm.bind_tools(tools).invoke(msgs)
        text = raw.content if isinstance(raw.content, str) else "".join(
            p if isinstance(p, str) else p.get("text", "") for p in raw.content)
        call = (raw.tool_calls or [None])[0]
        content = ToolStep(tool=call["name"] if call else None, arguments=dict(call["args"]) if call else {},
                           text=text).model_dump_json()
        u = getattr(raw, "usage_metadata", None) or {}
        return RawResult(content, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)))
    if schema is not None:
        out = llm.with_structured_output(schema, include_raw=True).invoke(msgs)
        raw = out["raw"]
        if out.get("parsed") is None:
            raise LLMOutputError(f"structured output failed to parse: {out.get('parsing_error')}")
        content = out["parsed"].model_dump_json()
    else:
        raw = llm.invoke(msgs)
        content = raw.content if isinstance(raw.content, str) else "".join(
            p if isinstance(p, str) else p.get("text", "") for p in raw.content)
    u = getattr(raw, "usage_metadata", None) or {}
    return RawResult(content, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)))


class LLMClient:
    def __init__(self, run_dir: Path | None = None, cache_dir: Path | None = None, mode: str = "cache",
                 model: str = DEFAULT_MODEL, temperature: float = 0.0, max_retries: int = 6,
                 base_delay: float = 2.0, backend: Backend | None = None, rpm: float | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        if mode not in ("cache", "replay_only", "no_cache"):
            raise ValueError(mode)
        self.mode, self.model, self.temperature = mode, model, temperature
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.run_dir = Path(run_dir) if run_dir else None
        self.max_retries, self.base_delay, self.sleep = max_retries, base_delay, sleep
        self.backend = backend or _gemini_backend
        rpm = rpm if rpm is not None else MODEL_RPM.get(model)
        self.min_interval = 60.0 / rpm * 1.15 if rpm else 0.0   # 15% safety margin under the limit
        self._last_call = -1e9
        self.calls: list[CallRecord] = []
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    # ---- public API ---------------------------------------------------------------------------
    def structured(self, schema: type[T], system: str, prompt: str, agent: str,
                   thinking_budget: int | None = 1024) -> T:
        content = self._call(system, prompt, agent, schema, thinking_budget)
        try:
            return schema.model_validate_json(content)
        except ValidationError as e:
            raise LLMOutputError(f"{agent}: response did not match {schema.__name__}: {e}") from e

    def text(self, system: str, prompt: str, agent: str, thinking_budget: int | None = 2048) -> str:
        return self._call(system, prompt, agent, None, thinking_budget)

    def tool_step(self, system: str, prompt: str, tools: list[dict], agent: str,
                  thinking_budget: int | None = 1024) -> ToolStep:
        """One native tool-calling turn. The conversation so far is serialized into ``prompt`` (so the
        step is a pure function of its inputs and therefore cacheable and replayable)."""
        content = self._call(system, prompt, agent, None, thinking_budget, tools=tools)
        try:
            return ToolStep.model_validate_json(content)
        except ValidationError as e:
            raise LLMOutputError(f"{agent}: bad tool-step payload: {e}") from e

    def usage(self) -> dict:
        live = [c for c in self.calls if not c.cached]
        hit = [c for c in self.calls if c.cached]
        return {
            "calls": len(self.calls), "api_calls": len(live), "cache_hits": len(hit),
            "input_tokens": sum(c.input_tokens for c in live), "output_tokens": sum(c.output_tokens for c in live),
            "cost_usd_live": round(sum(c.cost_usd for c in live), 6),
            "cost_usd_replayed": round(sum(c.cost_usd for c in hit), 6),
            "retries": sum(c.retries for c in self.calls),
        }

    # ---- internals ----------------------------------------------------------------------------
    def _key(self, system, prompt, schema, thinking_budget, tools=None) -> str:
        d = {"model": self.model, "system": system, "prompt": prompt,
             "schema": schema.model_json_schema() if schema else None,
             "thinking_budget": thinking_budget, "temperature": self.temperature}
        if tools:
            d["tools"] = tools
        blob = json.dumps(d, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _cost(self, in_tok: int, out_tok: int) -> float:
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        return (in_tok * pin + out_tok * pout) / 1_000_000

    def _call(self, system, prompt, agent, schema, thinking_budget, tools=None) -> str:
        """One LLM call, recorded as a Langfuse generation (no-op without Langfuse keys)."""
        with get_tracer().observe(
                f"llm:{agent}", as_type="generation", model=self.model,
                input={"system": system, "prompt": prompt},
                model_parameters={"temperature": self.temperature, "thinking_budget": thinking_budget}) as obs:
            content = self._call_untraced(system, prompt, agent, schema, thinking_budget, tools)
            rec = self.calls[-1]
            # cache hits report zero usage/cost so replayed calls are not double-counted in dashboards;
            # the original numbers stay in the metadata.
            obs.update(output=content,
                       usage_details={"input": 0 if rec.cached else rec.input_tokens,
                                      "output": 0 if rec.cached else rec.output_tokens},
                       cost_details=None if rec.cached else {"total": rec.cost_usd},
                       metadata={"cached": rec.cached, "cache_key": rec.key, "retries": rec.retries,
                                 "latency_s": rec.latency_s, "usd_equivalent": rec.cost_usd,
                                 "recorded_input_tokens": rec.input_tokens,
                                 "recorded_output_tokens": rec.output_tokens,
                                 "schema": schema.__name__ if schema else None,
                                 "tools": [x["name"] for x in tools] if tools else None})
            return content

    def _call_untraced(self, system, prompt, agent, schema, thinking_budget, tools=None) -> str:
        key = self._key(system, prompt, schema, thinking_budget, tools)
        path = self.cache_dir / f"{key}.json"
        if self.mode != "no_cache" and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            self._record(CallRecord(agent, self.model, key, True, d["input_tokens"], d["output_tokens"],
                                    self._cost(d["input_tokens"], d["output_tokens"]), 0.0))
            return d["content"]
        if self.mode == "replay_only":
            raise CacheMiss(f"{agent}: no cached response for key {key}")

        t0, retries = time.time(), 0
        while True:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                self.sleep(wait)  # stay under the per-minute limit instead of provoking 429s
            self._last_call = time.monotonic()
            try:
                args = (self.model, system, prompt, schema, thinking_budget, self.temperature)
                res = self.backend(*args, tools=tools) if tools else self.backend(*args)
                break
            except LLMOutputError:
                raise
            except Exception as e:  # noqa: BLE001 - SDK raises many exception types
                msg = f"{type(e).__name__}: {e}"
                if retries >= self.max_retries or not _RETRYABLE.search(msg):
                    raise
                delay = min(60.0, self.base_delay * 2 ** retries) * (0.75 + 0.5 * random.random())
                hint = _RETRY_HINT.search(msg)
                wait = 0.0
                if hint:
                    wait = (float(hint.group(4)) if hint.group(4) else
                            int(hint.group(1) or 0) * 3600 + int(hint.group(2) or 0) * 60 + float(hint.group(3)))
                if wait > MAX_SENSIBLE_WAIT_S or "PerDay" in msg:
                    raise QuotaExhausted(f"{self.model}: daily quota exhausted (server asks to retry in "
                                         f"{wait / 3600:.1f}h). Use another model, wait, or enable billing. "
                                         f"({msg[:200]})") from e
                if hint:
                    delay = max(delay, wait + 1.0)
                retries += 1
                self.sleep(delay)
        latency = time.time() - t0
        rec = CallRecord(agent, self.model, key, False, res.input_tokens, res.output_tokens,
                         self._cost(res.input_tokens, res.output_tokens), round(latency, 2), retries)
        if self.mode == "cache":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "model": self.model, "agent": agent, "system": system, "prompt": prompt,
                "schema": schema.__name__ if schema else None, "thinking_budget": thinking_budget,
                "tools": [x["name"] for x in tools] if tools else None,
                "content": res.content, "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                "created": time.time()}, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, path)
        self._record(rec)
        return res.content

    def _record(self, rec: CallRecord) -> None:
        self.calls.append(rec)
        if self.run_dir:
            with open(self.run_dir / "llm_calls.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec)) + "\n")
