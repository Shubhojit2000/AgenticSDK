"""MCP server for cross-run memory: one tool, `search_experience`, scoped to ONE task.

    python -m agenticsdk.tools.memory_server --dataset credit_g --seed 0

It is a separate process from the dataset-tools server on purpose. This one needs the embedding API
key (read from .env) but never touches any split, the dataset server touches the train split but has
no key. The leave-one-dataset-out rule lives in `MemoryTools`, not in the tool arguments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from agenticsdk.memory import DEFAULT_MEMORY_DIR, ExperienceStore, MemoryTools, get_embedder


def build_server(dataset: str, seed: int = 0, memory_dir: Path = DEFAULT_MEMORY_DIR,
                 embedder: str = "gemini") -> MCPServer:
    tools = MemoryTools(ExperienceStore(memory_dir, get_embedder(embedder)), dataset, seed)
    server = MCPServer("agenticsdk-memory")

    @server.tool()
    def search_experience(query: str = "", k: int = 3) -> str:
        """Retrieve summaries of earlier runs on OTHER datasets that resemble this one: the strategies
        tried, how each scored against the frozen baselines (validation split), what failed and what the
        Critic said. query: optional free text, e.g. 'many categorical columns, skewed target'; this
        dataset's own description is always added. k: number of cards, 1 to 5. The cards are hints from
        different data, not measurements of this dataset."""
        try:
            return tools.search_experience(query, k)
        except Exception as e:  # noqa: BLE001 - a readable error the model can act on, not a stack trace
            return f"ERROR: memory search failed: {type(e).__name__}: {e}"

    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR))
    ap.add_argument("--embedder", default="gemini", choices=["gemini", "hash"])
    a = ap.parse_args()
    build_server(a.dataset, a.seed, Path(a.memory_dir), a.embedder).run()


if __name__ == "__main__":
    main()
