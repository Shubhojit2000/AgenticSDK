# AgenticSDK

[![CI](https://github.com/Shubhojit2000/AgenticSDK/actions/workflows/ci.yml/badge.svg)](https://github.com/Shubhojit2000/AgenticSDK/actions/workflows/ci.yml)

A multi-agent system (LangGraph) that builds ML pipelines for tabular datasets, plus an evaluation
harness that holds the validation and test labels itself, so an agent cannot grade its own work.
Status: research project in progress (see [`docs/DESIGN.md`](docs/DESIGN.md) for the design and the checkpoint results).

## Layout

```
agenticsdk/
  llm.py            LLM client: retries, rate limiting, record/replay cache, token and cost accounting
  tracing.py        optional Langfuse tracing (no-op without keys); `python -m agenticsdk.tracing` checks your keys
  harness/          the trusted side: datasets, splits, scorer, baselines, and the sandbox that runs agent code
  agents/           the agents: schemas, prompts, the LangGraph graph, the run command, the trace viewer
  tools/            the MCP server the agents call through native function calling
tests/              scripted-LLM tests (no API key needed); tests/verify/ has local replay checks
docs/DESIGN.md      design, evaluation protocol, and the measured result at each checkpoint
results/            frozen baselines and the checkpoint traces
```

## Reading order (about 2,500 lines of source)

1. `harness/datasets.py`, `harness/scorer.py`: what is evaluated and how a score is computed. The agent never sees val/test labels.
2. `harness/sandbox.py`, `harness/allowlist.py`: how agent-written scripts run (temp dir, stripped environment, timeout with process-tree kill, AST allow-list).
3. `agents/schemas.py`, `agents/prompts.py`: the data the agents pass around and what they are told.
4. `agents/graph.py`: the whole loop, planner → feature → training → execute → critic → report, with budgets enforced in code.
5. `llm.py`: the model client; `agents/run.py`: the command that runs one dataset.
6. `tools/`: the MCP server (`server.py`), its tools (`dataset_tools.py`) and the client the graph uses (`client.py`).

## What exists today

- **Harness:** 12 development and 3 unseen OpenML datasets, frozen 60/20/20 splits, scoring (log-loss / R²),
  and three frozen baselines (default RandomForest, default LightGBM, FLAML).
- **Agents:** planner → feature → training → sandboxed execution → critic, with strategy escalation
  (`RETRY_STRATEGY`), budgets enforced in code, and resumable runs.
- **Tools:** an MCP server (dataset profile, column inspection, quick cross-validation) that the agents call
  through native function calling (`--use-tools`).
- **Safety:** per-attempt temp dir, stripped environment, wall-clock timeout with process-tree kill, AST
  allow-list. The allow-list is defense in depth, not a security boundary.
- **Observability:** optional Langfuse tracing; JSON-lines logs are always written.

## Quick start

```bash
python -m venv venv && venv/Scripts/activate        # or: source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                                  # add GOOGLE_API_KEY (free tier) to run live agents
python -m agenticsdk.harness.datasets                 # download and freeze the datasets (needs network)

python -m agenticsdk.agents.run --dataset california_housing --first-family linear --run-id demo
python -m agenticsdk.agents.show demo                 # readable timeline of the run
python -m agenticsdk.agents.run --dataset eucalyptus --use-tools --run-id demo_tools
```

## Tests (no API key needed)

```bash
ruff check .
pytest --cov
```

Every LLM call in the tests is scripted, so the suite needs no key and sends nothing anywhere.
The tests download two small OpenML datasets on first use. `tests/verify/verify_phase0_2.py` and `tests/verify/verify_phase3.py`
replay recorded live runs; those replays depend on numerically identical results, so they are
meant for the machine that recorded them and are not part of CI.

## Tracing (optional)

Put `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (and `LANGFUSE_HOST`) in `.env`, then run
`python -m agenticsdk.tracing` to check them. Each run becomes one trace: `agent_run` → node spans
(planner, feature, training, execute, critic, report) → LLM generations (tokens, USD-equivalent cost,
cache hit or live) and tool calls. Without keys nothing is sent.
