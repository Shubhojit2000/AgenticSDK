"""Test isolation: unit tests never talk to Langfuse or Gemini, whatever is in the developer's .env."""
import pytest

from agenticsdk.tracing import NoopTracer, set_tracer


@pytest.fixture(autouse=True)
def _no_external_services(monkeypatch):
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    set_tracer(NoopTracer())
    yield
    set_tracer(None)
