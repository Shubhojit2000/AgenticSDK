"""Synchronous wrapper around the official MCP client, so LangGraph nodes can call MCP tools.

The server runs as a stdio subprocess (a separate process, started with a minimal environment: no
API keys). A background thread owns the asyncio loop and the session; `call` is thread-safe.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

from mcp import Client, StdioServerParameters

from agenticsdk.harness.sandbox import _clean_env

ROOT = Path(__file__).resolve().parent.parent.parent
CALL_TIMEOUT_S = 120

# Fields Gemini function declarations reject; the MCP/JSON-Schema originals are kept in the server.
_STRIP = {"title", "additionalProperties", "$schema"}


def _clean_schema(node):
    if isinstance(node, dict):
        return {k: _clean_schema(v) for k, v in node.items() if k not in _STRIP}
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


class MCPToolClient:
    """Usage: ``with MCPToolClient('credit_g', 0) as t: t.specs; t.call('profile_dataset', {})``."""

    def __init__(self, dataset: str, seed: int = 0, server_args: list[str] | None = None):
        self.params = StdioServerParameters(
            command=sys.executable, cwd=ROOT, env={**_clean_env(), "PYTHONPATH": str(ROOT)},
            args=["-m", "agenticsdk.tools.server", "--dataset", dataset, "--seed", str(seed)] + (server_args or []))
        self.specs: list[dict] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)

    async def _main(self) -> None:
        self._loop, self._stop = asyncio.get_running_loop(), asyncio.Event()
        try:
            async with Client(self.params) as c:
                self._client = c
                listed = await c.list_tools()
                self.specs = [{"name": t.name, "description": t.description or "",
                               "parameters": _clean_schema(t.input_schema)} for t in listed.tools]
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 - surfaced to the caller of start()
            self._error = e
            self._ready.set()

    def start(self) -> MCPToolClient:
        self._thread.start()
        if not self._ready.wait(60) or self._error:
            raise RuntimeError(f"MCP server failed to start: {self._error!r}")
        return self

    def call(self, name: str, arguments: dict | None = None) -> str:
        fut = asyncio.run_coroutine_threadsafe(self._client.call_tool(name, arguments or {}), self._loop)
        res = fut.result(timeout=CALL_TIMEOUT_S)
        text = "\n".join(getattr(c, "text", "") for c in res.content)
        return f"ERROR: {text}" if getattr(res, "is_error", False) and not text.startswith("ERROR") else text

    def close(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=15)

    def __enter__(self) -> MCPToolClient:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
