"""Langfuse tracing behind a tiny facade.

* No Langfuse keys in the environment -> a no-op tracer: nothing is sent, nothing is slower.
* Tracing must never break or change a run: every Langfuse call is wrapped, and a failure
  degrades to "no tracing" for that observation.
* The JSON-lines logs (events.jsonl, llm_calls.jsonl) are written regardless; Langfuse is the
  nested, searchable view of the same run (agent_run > node > llm generation / tool call).

Configure with LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and (optionally) LANGFUSE_HOST in .env.
Set AGENTICSDK_TRACING=off to disable tracing even when keys are present.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("agenticsdk.tracing")


class _NoopHandle:
    def update(self, **kw) -> None:
        pass

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        pass

    def trace_url(self) -> str | None:
        return None


class _Handle(_NoopHandle):
    def __init__(self, obs, client):
        self._obs, self._client = obs, client

    def update(self, **kw) -> None:
        try:
            self._obs.update(**{k: v for k, v in kw.items() if v is not None})
        except Exception as e:  # noqa: BLE001
            log.debug("tracing update failed: %s", e)

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        try:
            self._obs.score_trace(name=name, value=value, comment=comment)
        except Exception as e:  # noqa: BLE001
            log.debug("tracing score failed: %s", e)

    def trace_url(self) -> str | None:
        try:
            return self._client.get_trace_url()
        except Exception:  # noqa: BLE001
            return None


class NoopTracer:
    enabled = False

    @contextlib.contextmanager
    def observe(self, name: str, as_type: str = "span", **kw) -> Iterator[_NoopHandle]:
        yield _NoopHandle()

    @contextlib.contextmanager
    def session(self, session_id: str, tags: list[str] | None = None, metadata: dict | None = None):
        yield

    def flush(self) -> None:
        pass


class LangfuseTracer(NoopTracer):
    enabled = True

    def __init__(self, client):
        self.client = client

    @contextlib.contextmanager
    def observe(self, name: str, as_type: str = "span", **kw) -> Iterator[_NoopHandle]:
        try:
            cm = self.client.start_as_current_observation(
                name=name, as_type=as_type, **{k: v for k, v in kw.items() if v is not None})
            obs = cm.__enter__()
        except Exception as e:  # noqa: BLE001 - never let tracing break a run
            log.debug("tracing start failed: %s", e)
            yield _NoopHandle()
            return
        handle = _Handle(obs, self.client)
        try:
            yield handle
        except BaseException:
            handle.update(level="ERROR", status_message=str(sys.exc_info()[1])[:500])
            with contextlib.suppress(Exception):
                cm.__exit__(*sys.exc_info())
            raise
        else:
            with contextlib.suppress(Exception):
                cm.__exit__(None, None, None)

    @contextlib.contextmanager
    def session(self, session_id: str, tags: list[str] | None = None, metadata: dict | None = None):
        """Attach session id / tags / metadata to every observation created inside."""
        try:
            from langfuse import propagate_attributes
            cm = propagate_attributes(session_id=session_id, tags=tags, metadata=metadata)
            cm.__enter__()
        except Exception as e:  # noqa: BLE001
            log.debug("tracing session failed: %s", e)
            yield
            return
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                cm.__exit__(None, None, None)

    def flush(self) -> None:
        with contextlib.suppress(Exception):
            self.client.flush()


def from_env() -> NoopTracer:
    """A Langfuse tracer if keys are configured, else a no-op tracer."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if os.environ.get("AGENTICSDK_TRACING", "").lower() == "off":
        return NoopTracer()
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return NoopTracer()
    try:
        from langfuse import Langfuse
        return LangfuseTracer(Langfuse())
    except Exception as e:  # noqa: BLE001
        log.warning("Langfuse tracing disabled: %s", e)
        return NoopTracer()


_tracer: NoopTracer | None = None


def get_tracer() -> NoopTracer:
    global _tracer
    if _tracer is None:
        _tracer = from_env()
    return _tracer


def set_tracer(tracer: NoopTracer | None) -> None:
    """Install a tracer (tests, notebooks). ``None`` resets to lazy from-env creation."""
    global _tracer
    _tracer = tracer


# ---------------------------------------------------------------- `python -m agenticsdk.tracing`
def main() -> int:
    """Check your Langfuse setup: sends ONE tiny test trace and prints its URL."""
    tracer = from_env()
    if not tracer.enabled:
        print("Tracing is off: no LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in the environment or .env.\n"
              "Create a free project at https://cloud.langfuse.com, copy its API keys into .env "
              "(see .env.example), and run this again.")
        return 1
    try:
        ok = tracer.client.auth_check()
    except Exception as e:  # noqa: BLE001
        print(f"Langfuse auth check failed: {type(e).__name__}: {e}")
        return 2
    if not ok:
        print("Langfuse rejected the keys (auth check returned False). Check the keys and LANGFUSE_HOST.")
        return 2
    with tracer.session("smoke-test", tags=["smoke"]), tracer.observe("smoke_test", as_type="span",
                                                                      input="hello") as h:
        h.update(output="world")
        url = h.trace_url()
    tracer.flush()
    print("Keys OK. Sent one test trace." + (f"\nOpen it: {url}" if url else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
