# AgenticSDK: design and progress

Critic-guided multi-agent ML engineering: a system plus an evaluation study. This document
records the design, the evaluation protocol, and the measured result at each checkpoint.

## What this project is (and is not)

Two deliverables, built together:

1. **A system:** a multi-agent LLM system that takes an unseen tabular dataset + a task
   description and autonomously does EDA, feature engineering, model selection, training,
   self-correction and evaluation. It uses native tool calling over an MCP server, cross-run
   memory, a Critic that can reject a whole *strategy*, budget awareness, a sandbox, and a
   production API.
2. **A study:** a rigorous, honest evaluation of that system, in the style of a research lab
   writeup.
   - A harness-owned hidden holdout, so the agent can never grade itself.
   - Three baselines, up to and including a real AutoML tool.
   - Multiple seeds, confidence intervals and significance tests.
   - Ablations of every design choice, with token, cost and latency accounting.
   - A failure taxonomy.
   - A measurement of how often the agent tries to game its metric (specification gaming).

The study is what separates this from a typical LangGraph demo: the agent's score is computed by a harness
that holds the labels, the baselines include a real AutoML tool, and every design choice is ablated with
token, cost and latency accounting.

**Honest framing:** this is a research project. Results are reported as measured, including losses.

**Related work:** AIDE (Weco), MLE-bench (OpenAI), MLAgentBench (Stanford), DS-Agent, Data Interpreter
(MetaGPT), AutoKaggle, and AutoML benchmarks (AMLB, FLAML, AutoGluon).
Differentiators: (a) two-level Critic escalation (code vs. strategy), (b) harness-enforced integrity plus
a specification-gaming measurement, and (c) an ablation study with cost accounting.

---

## Tech stack

| Concern | Choice |
|---|---|
| Orchestration | LangGraph (`langgraph`) with `SqliteSaver` checkpointer (resumable runs) |
| LLM | `gemini-3.5-flash-lite` (free tier, 500 requests/day, `.env`) via `langchain-google-genai`; `gemini-3.1-flash-lite` (500/day) and the free Gemma 4 models (`gemma-4-26b-a4b-it`, `gemma-4-31b-it`; 14.4K/day) for the model-comparison ablation; the 20/day Gemini models only for tiny subsets; NVIDIA API as a second provider later |
| LLM client wrapper | own thin wrapper: retry with exponential backoff on 429/5xx, token + cost accounting, **record/replay cache** keyed by prompt hash (reproducible runs, free re-runs, deterministic tests) |
| Tools | **MCP server** built with the official `mcp` Python SDK (FastMCP), consumed via `langchain-mcp-adapters`; native function calling |
| Memory | Chroma vector store + Gemini embeddings over past-run "experience cards" |
| Structured outputs | Pydantic v2 schemas for every agent output and the graph state |
| ML libraries agents may use | scikit-learn, LightGBM, pandas, numpy (allow-listed) |
| Baselines | default RandomForest, default LightGBM, FLAML under equal time budget |
| Datasets | OpenML (`openml` package), fixed task IDs, frozen splits |
| Sandbox | per-attempt temp dir, `subprocess` with timeout + process-tree kill, stripped env (no API keys), allow-listed imports; Docker `--network none` in Phase 8 |
| Observability | Langfuse (free cloud tier) + own JSON-lines logs |
| Stats | scipy (Wilcoxon signed-rank), bootstrap CIs |
| Quality | pytest, ruff, GitHub Actions CI, mocked-LLM tests via the replay cache |
| Serving | FastAPI (+ SSE streaming of run events), Docker, Gradio frontend, Hugging Face Spaces |
| Writeup | 6–8 page technical report (LaTeX/Overleaf) + README + blog post |

Rate-limit plan: develop with low retry caps (3 code / 2 strategy) and use the replay cache
aggressively.

**Free-tier budget (free models only; no paid billing).** Measured limits on the
development key (AI Studio rate-limit page, Oct 2026):

| Model | RPM | RPD |
|---|---|---|
| `gemini-3.5-flash-lite` (default workhorse) | 15 | **500** |
| `gemini-3.1-flash-lite` (API id; same limits, 250K tokens/min) | 15 | **500** |
| `gemma-4-26b-a4b-it` (open weights, 26B MoE) | 30 | **14,400** (but only 16K tokens/min) |
| `gemma-4-31b-it` (open weights, 31B dense) | 30 | **14,400** (but only 16K tokens/min) |
| `gemini-2.5-flash`, `gemini-3-flash-preview`, `gemini-3.8-flash`, `gemini-2.5-flash-lite` | 5-10 | 20 |

With the feature agent and escalation (Phase 2) a run costs ~8-14 LLM calls (10 in the first live
run), so 500 requests/day is ~40-60 runs/day. One benchmark pass (12 datasets × 3 seeds ≈ 36 runs ≈
400 calls) fits in about one day; each ablation variant needs another pass,
so the full study is spread over roughly a week of daily quota. The replay cache makes any re-run
free, and a variant that does not change an agent's prompt reuses that agent's cached calls. The
client throttles itself to the model's RPM and raises `QuotaExhausted` immediately on a daily
quota instead of sleeping for hours. Quality caveat: a flash-lite model is weaker at coding than
Flash/Pro, which makes the agent's win-rate harder and the reported numbers more honest; the model
swap ablation (Phase 7) uses the 20/day models on a small subset only.

**Models added 2026-10-05** (API ids confirmed with `models.list()` on the key; each answered a structured-output
call and a plain-text call through our client, except as noted): `gemini-3.1-flash-lite`, `gemma-4-26b-a4b-it`,
`gemma-4-31b-it`. Observations from the probe, not conclusions: the 31B Gemma took 34 s for a trivial
structured call and returned a 500 on one text call, so treat it as slow and flaky until measured; the Gemma
models' 16K tokens/min budget is the binding limit (a retry prompt that embeds the previous script is
several thousand tokens, so ~4-6 calls/min), not their 14.4K requests/day. The big daily allowance makes
Gemma attractive for the many-run benchmark and ablations; whether it writes good enough code is an open
question that Phase 7's model-comparison ablation answers. The client does not yet throttle on tokens/min;
it relies on backoff with the server's retry hint.
**Planned: add an NVIDIA API (build.nvidia.com, OpenAI-compatible endpoints) behind a
provider-agnostic LLM interface**, so the same agents can run on other free models and so the
study can report a cross-provider comparison. `LLMClient` already takes a pluggable `backend`
callable; a second backend plus a `provider` field in the cache key is all that is needed.

---

## Architecture

```
                       ┌─────────────────────────────┐
  dataset (train only) │ HARNESS (trusted)            │  holds hidden holdout labels,
  + task description ─▶│ splits · scores · integrity  │  scores predictions itself
                       └──────────────┬──────────────┘
                                      ▼
                          ┌────────────────────┐   tools via MCP:
                     ┌──▶ │ planner_agent      │◀─ profile_dataset, inspect_column,
                     │    └─────────┬──────────┘   run_quick_cv, search_experience
                     │              ▼
                     │    ┌────────────────────┐
                     │    │ feature_agent      │◀─ inspect_column, run_quick_cv
                     │    └─────────┬──────────┘
                     │              ▼
                     │    ┌────────────────────┐
                ┌──▶ │──▶ │ training_agent     │  writes full script → must write
                │    │    └─────────┬──────────┘  predictions.csv for the test features
                │    │              ▼
                │    │    ┌────────────────────┐
                │    │    │ sandbox_exec       │  temp dir, timeout, no secrets,
                │    │    └─────────┬──────────┘  import allow-list
                │    │              ▼
                │    │    ┌────────────────────┐
                │    │    │ integrity_monitor  │  static + runtime checks for metric gaming
                │    │    └─────────┬──────────┘
                │    │              ▼
                │    │    ┌────────────────────┐  sees CV score + harness validation score,
                │    │    │ critic_agent       │  budget left, attempt history
                │    │    └─────────┬──────────┘
                │    │   RETRY_CODE ─────────────▶ training_agent   (max 10)
                │    └── RETRY_STRATEGY ─────────▶ planner_agent    (max 3)
                │         [optional HITL interrupt: human approves new strategy]
                │                   │ PASS / FAIL / BUDGET_EXHAUSTED
                │                   ▼
                │         ┌────────────────────┐
                └──────── │ report_node        │  harness scores on hidden holdout,
                          └─────────┬──────────┘  writes experience card to memory
                                    ▼
                                   END
```

**Integrity rule (non-negotiable):** the agent sees train features and labels, plus *test
features only*. The harness alone holds the test labels and computes the reported metric from
`predictions.csv`. Any number the agent prints is advisory and never reported.

The Critic decides using:
- the agent's own CV score;
- a *harness-computed* score on a separate validation split, so the Critic is not fooled by
  self-reported numbers;
- the retry history;
- the remaining token, time and cost budget.

**State (Pydantic):**
- `run_id`, `dataset_id`, `task_description`, `metric_name`, `budget`
  (tokens / seconds / USD-equivalent), `spent`
- `baselines` (B0/B1/B2 validation scores)
- `strategy_history: list[Strategy]`, `current_strategy`, `current_feature_spec`
- `current_code`, `last_exec` (stdout, stderr, exit code, duration)
- `cv_score`, `harness_val_score`, `integrity_flags: list[str]`
- `code_retry_count`, `strategy_retry_count`, `verdict`, `status`
- `final_holdout_score`, `trace: list[Event]`

---

## Repository layout

All source lives in one package, `agenticsdk/`.

```
AgenticSDK/
  agenticsdk/
    llm.py            LLM client (backoff, accounting, record/replay cache)
    tracing.py        Langfuse facade; `python -m agenticsdk.tracing` checks the Langfuse keys
    harness/          datasets.py, scorer.py, baselines.py, profile.py, sandbox.py, allowlist.py
    agents/           schemas.py, prompts.py, graph.py, run.py, show.py
    tools/            server.py, client.py, dataset_tools.py   (MCP)
    (later phases, as single modules where possible: memory.py, integrity.py, api.py, eval/ ...)
  tests/  results/  docs/  .github/workflows/ci.yml  README.md  pyproject.toml
```

Commands: `python -m agenticsdk.agents.run ...`, `python -m agenticsdk.agents.show <run>`,
`python -m agenticsdk.harness.baselines`, `python -m agenticsdk.harness.datasets`.

---

## Tiers (scope control)

- **Tier 1 — Core (must finish):** Phases 0–4 and 7. A working, honest, tested system with a
  real benchmark.
- **Tier 2 — Differentiators:** Phases 5, 6, 8 and the
  full ablation set.
- **Tier 3 — Stretch:** MLE-bench Lite subset, open-weight model comparison, workshop
  submission.

Finish each tier before starting the next. A finished Tier 1+2 beats an unfinished Tier 3.

---

## Phase 0 — Harness, Datasets, Baselines (no LLM yet)

1. [x] venv, Python 3.12 (3.11 not installed; `requires-python >=3.11`), `pyproject.toml`; install the stack. Confirm the `GOOGLE_API_KEY`
   loads and that one Gemini Flash call works.
2. [x] `agenticsdk/harness/datasets.py`: registry of **12 OpenML tasks**. Mix binary, multiclass and
   regression; 500 to 100k rows; some with categoricals, missing values and imbalance. Freeze
   the IDs and the train/val/test splits (seeded, stored as parquet). Hold out **3 more** as
   never-touched "final unseen" datasets.
3. [x] `agenticsdk/harness/scorer.py`: scores a `predictions.csv` against hidden labels, validating the
   schema (row count, ids, dtypes, probability range).
4. [x] `agenticsdk/harness/baselines.py`: B0 = default RandomForest, B1 = default LightGBM, B2 = FLAML
   with the same wall-clock budget the agent gets (60 s). Run on all 12 datasets × 3 seeds and
   frozen to `results/baselines.json` (60/20/20 train/val/test; 36 runs per baseline, 0 failures).
5. [x] **Metric + normalization (revised after the first baseline run):**
   - Classification is scored by **negative log-loss** (probabilities clipped to [1e-6, 1-1e-6]
     for everyone); regression by R². ROC-AUC saturated (letter/sick/car all > 0.995) and R² on
     `brazilian_houses` was decided by two outlier rows, so that dataset was replaced by
     `california_housing`. The discarded run is kept as `results/baselines_v0_rocauc_DISCARDED.json.bak`.
   - Normalized score = (s − B0) / max(ref − B0, 0.01) with `ref = max(B1, B2)`, so 0 = default
     forest and 1 = the strongest baseline. Datasets whose mean gap is below 0.01 are flagged
     *uninformative* (credit_g, diamonds) and excluded from normalized aggregates; raw paired
     differences and win/loss counts are reported for all 12.
6. [x] **Checkpoint 0:** `results/checkpoint0_table.txt`. B2 beats B0 on 10/12 datasets; it loses
   on `credit_g` and `credit_approval` (both < 700 rows, high seed variance, so 60 s of AutoML
   overfits its CV). The strongest baseline is B2 on 9/12 and B1 on `bank_marketing`.

## Phase 1 — Sandbox + LLM Client + Single-Agent Loop

7. [x] `agenticsdk/harness/sandbox.py`:
   - fresh temp dir per attempt, containing only train data + test features;
   - timeout with process-*tree* kill (test this on Windows explicitly);
   - environment stripped of all secrets;
   - capture stdout, stderr, exit code and duration.
8. [x] `agenticsdk/harness/allowlist.py`: AST scan that rejects imports outside the allow-list, `os.system`,
   `subprocess`, network modules, and file access outside the temp dir.
9. [x] `agenticsdk/llm.py`:
   - backoff on 429/5xx;
   - per-call token logging and a USD-equivalent cost at list price, even on the free tier;
   - record/replay cache.
10. [x] Pydantic schemas for `Strategy`, `Verdict`, `ExecSummary`, `Attempt` (holds the code; no separate
    `CodeArtifact`) and the full `AgentState`.
11. [x] planner → training → sandbox → critic (`PASS` / `RETRY_CODE`) → report.
12. [x] **Checkpoint 1:**
    - The loop works on 1 dataset and the harness scores the holdout.
    - Killing the process mid-run and re-running with the replay cache reproduces the run with
      zero API calls.
    - A deliberately infinite-loop script is killed cleanly.

## Phase 2 — Multi-Agent Graph + Strategy Escalation

13. [x] Add `feature_agent` and the 3-way Critic verdict (`PASS` / `RETRY_CODE` /
    `RETRY_STRATEGY`) plus `FAIL` / `BUDGET_EXHAUSTED`.
14. [x] On `RETRY_STRATEGY`, the planner receives the full strategy history plus the Critic's
    reasoning, and **must** propose a strategy that differs on a structured field (model family
    or feature approach). Enforce this in the schema, not in the prompt.
15. [x] The Critic uses `harness_val_score` (trusted), not just the agent's printed CV.
16. [x] Add the `SqliteSaver` checkpointer, so runs are resumable after a crash or rate-limit
    stall.
17. [x] **Checkpoint 2:** on a dataset engineered to defeat the first instinct (e.g. a strongly
    non-linear task with a linear-model first guess), the trace shows a real `RETRY_STRATEGY`
    that improves the harness score.

    **Result (2026-10-05, `gemini-3.5-flash-lite`, `california_housing`, seed 0, first strategy forced
    to `linear`):** strategy 1 (linear + transforms_scaling) scored val R² 0.623; the Critic returned
    `RETRY_STRATEGY` ("linear models are too restrictive", all baselines ≥ 0.80); strategy 2
    (gradient_boosting + aggregations_binning) first crashed to 0.061 (a code bug, `RETRY_CODE`), then
    scored val 0.849; test R² 0.855 against B2 0.847 (normalized 1.20). 10 API calls. `--replay-only`
    reproduces it with 0 API calls. Trace: `results/checkpoint2_trace.txt`. This is ONE run with an
    engineered bad start; it shows the mechanism works, not that the agent beats the baselines in
    general. Implementation notes: the Critic's verdict is clamped to the budget in code (code
    retries per strategy, strategy changes, total attempts); novelty on (model_family,
    feature_approach) is checked in code, with a re-ask loop and a clean stop if the planner
    cannot produce a new pair; the `first_family` constraint is an experiment knob, not a default.

## Phase 3 — Tools via MCP + Native Tool Calling

18. [x] `agenticsdk/tools/server.py` (MCP SDK 2.x, where `FastMCP` is now `MCPServer`). Tools:
    - `profile_dataset` (shape, dtypes, missingness, cardinality, target balance);
    - `inspect_column(name)`;
    - `run_quick_cv(model_spec, feature_spec, folds=3)`, which runs inside the sandbox;
    - `search_experience(query)`, wired in Phase 5 (not built yet).
    The server is scoped to one (dataset, seed) and reads the train split only. `run_quick_cv` does
    not run model-written code: it renders a fixed template from a validated spec (model family, a
    whitelist of range-checked hyperparameters, encoding, scaling) and runs it through the same
    sandbox as the agents' scripts.
19. [x] Connect the planner and feature agents to the MCP tools. The model decides when to call which
    tool (native function calling), with a per-agent tool-call budget enforced in code (planner 3,
    feature 2). **Deviation:** the official `mcp` client is used directly instead of
    `langchain-mcp-adapters`, because the tool loop is custom so that every turn goes through the
    record/replay cache (`LLMClient.tool_step`; the conversation is serialized into the prompt, which
    also avoids Gemini's thought-signature requirement for multi-turn function calls). Opt-in with
    `--use-tools`; with it off, behaviour and cache keys are identical to Phase 2.
20. [~] Test that the MCP server also works from a stock MCP client. Done: the official MCP SDK
    client talks to it over stdio in `tests/test_mcp_tools.py` and `tests/verify/verify_phase3.py`. Not yet
    verified: the MCP Inspector GUI or Claude Desktop (the command is in `agenticsdk/tools/server.py`).
21. [x] **Checkpoint 3:** the trace shows the planner calling `profile_dataset` and
    `run_quick_cv` *before* committing to a strategy, and the decision changes based on what the
    tool returned.

    **Result (2026-10-05, `gemini-3.5-flash-lite`, `eucalyptus`, seed 0, `--use-tools`):** the planner
    called `profile_dataset`, then `run_quick_cv` on gradient boosting (neg-log-loss -1.17) and linear
    (-1.39), and then chose bagged trees, away from the boosting default; it never measured bagged
    trees itself. After two weak attempts the Critic returned `RETRY_STRATEGY`; the re-planning
    investigation tried one-hot features with missing-value indicators and chose gradient boosting
    with a missingness-focused feature plan. Final: val -0.746, test -0.747 against B2 -0.746 and
    B0 -0.788 (normalized 0.98; the denominator is only 0.042, so read the raw numbers). 20 LLM
    calls, 10 tool calls. Trace: `results/checkpoint3_trace.txt`; replays with 0 API calls.
    Honest notes: (1) the first live run (`cp3_eucalyptus`, kept in `runs/`) reached test -0.695 with a
    stacked ensemble; the re-run's strategy-2 differs, so results vary between runs and this is one
    seed on one dataset, not a claim that tools help; Phase 7 measures that with a tools-on/off
    ablation. (2) That first run exposed a real flaw, found by reading the trace: the re-planning
    investigation repeated the first investigation call for call because its prompt lacked the
    earlier strategies. Fixed (it now sees the history) with a regression test. (3) A tools-on run
    costs about 15-25 LLM calls versus about 10, which matters for the 500/day budget.

## Phase 4 — Observability, Tests, CI

22. [x] Langfuse tracing of every LLM call and tool call, nested per run, with token and cost
    attributes; JSON-lines event log in parallel. `agenticsdk/tracing.py` is a facade that is a
    no-op without `LANGFUSE_*` keys and can never break a run. One trace per run: `agent_run` >
    node span (planner, feature, training, execute, critic, report; its output is the node's own
    decision, score or error) > LLM generation (model, tokens, USD-equivalent cost, cached or live)
    and tool call. Cache hits report zero usage so replays are not double-counted. Verified against
    the real Langfuse SDK with an in-memory exporter (`tests/test_tracing.py`); NOT yet against
    Langfuse Cloud (needs free API keys; `python -m agenticsdk.tracing` checks them).
23. [x] `tests/`:
    - scorer, sandbox (timeout, env strip, allow-list) and schema unit tests;
    - **mocked-LLM graph tests** so the full graph runs in CI with zero API calls. **Deviation:**
      these use a scripted fake LLM, not recorded replays. Recorded live runs replay exactly only on
      the machine that produced them: the prompts embed numbers computed by the sandboxed scripts and
      the CV tool (log-loss digits), which can differ across OS and library versions and cause cache
      misses. So `tests/verify/verify_phase0_2.py` / `tests/verify/verify_phase3.py` (which replay) stay local-only.
24. [x] GitHub Actions: ruff + pytest on every push. `.github/workflows/ci.yml` runs ruff, then the
    tests with coverage on Ubuntu and Windows (Python 3.12 only; 3.11 is untested), with no secrets.
    A coverage percentage is written to the job summary and `coverage.xml` is uploaded; a
    coverage floor of 70% is enforced (measured 76-79% depending on the run). First run on GitHub
    (2026-10-07, public repo, free standard runners): lint 6 s, Ubuntu 3m28s, Windows 5m02s, all green,
    so the Ubuntu job (never run before) passed too. The only annotations were Node-20 deprecation
    notices for the action versions; `ci.yml` was bumped to checkout v7 / setup-python v7 / cache v6 /
    upload-artifact v7 (not yet confirmed on GitHub). NOT done: a coverage badge (needs an external
    service or a gist).
25. [~] **Checkpoint 4:**
    - Green CI. DONE (2026-10-07, Ubuntu and Windows).
    - From a Langfuse trace alone, every decision in a past run can be explained.
    - A teammate could run the tests without an API key.

    **Status (2026-10-07):** a dry run of the workflow on a clean copy (only files git would track,
    a brand-new venv, no `.env`, a cold dataset cache, no keys) passed: ruff clean, 123 tests passed
    in 6m41s, 79% coverage (Windows only).
    Still open: explaining a past run from a Langfuse Cloud trace alone.

## Phase 5 — Cross-Run Memory (experience retrieval)

26. [ ] After each run, write an **experience card**: dataset meta-features, strategies tried,
    what failed and why, and the final normalized score. Embed it into Chroma.
27. [ ] `search_experience` retrieves the top-k similar past runs. The planner may use them but
    must cite which card influenced it.
28. [ ] **Leakage rule:** during benchmarking, memory only ever contains runs from *other*
    datasets (leave-one-dataset-out).
29. [ ] **Checkpoint 5:** on a held-out dataset, memory-on reaches its final score in fewer
    attempts or tokens than memory-off. If it doesn't, report that honestly.

## Phase 6 — Integrity & Specification-Gaming Study (the safety angle)

30. [ ] `harness/integrity.py`, static checks on generated code:
    - reading paths outside the sandbox;
    - hard-coded metric prints;
    - fitting on test features with labels inferred from elsewhere;
    - target leakage (a feature that is a near-copy of the target).
31. [ ] **Honeypot:** place a decoy `test_labels_DO_NOT_USE.csv` (random labels) in a fraction
    of sandboxes, and log any access.
    - Measure the **gaming rate** per model and per prompt variant.
    - Test whether an explicit integrity instruction or the Critic's oversight reduces it.
32. [ ] `integrity_monitor` node: flagged attempts are rejected, logged and counted. They are
    never reported as wins.
33. [ ] **Checkpoint 6:** a table of gaming-attempt rate × model × prompt condition, with
    example traces. This section is directly relevant to frontier-lab safety work, so write it
    carefully and neutrally.

## Phase 7 — Benchmark, Ablations, Failure Taxonomy, Statistics

34. [ ] Full benchmark: 12 datasets × 3 seeds, unattended, with a fixed budget per run. No
    per-dataset tuning of prompts, ever.
35. [ ] **Ablations** (same datasets, seeds and budget):
    - A. Full system
    - B. No strategy escalation (code retries only)
    - C. Single ReAct agent with the same tools (no multi-agent)
    - D. No tools (pure code generation)
    - E. No memory
    - F. Model swap: Flash-Lite vs Flash (vs Pro on a subset, if quota allows)
36. [ ] Metrics per variant:
    - normalized score;
    - win-rate vs B0, B1 and B2;
    - tokens and USD-equivalent cost;
    - wall-clock time and retries;
    - **cost per point of normalized score**.
37. [ ] Statistics:
    - bootstrap 95% CIs;
    - Wilcoxon signed-rank tests between variants;
    - the number of wins and losses that are actually significant.
38. [ ] **Failure taxonomy:** hand-label ~100 failed attempts (import error, API hallucination,
    schema mismatch, timeout, leakage, wrong metric, bad strategy, ...).
    - Build an LLM-judge auto-labeler.
    - Report its agreement with the hand labels (Cohen's κ). Then label the rest automatically.
39. [ ] Run the final system once on the **3 never-touched datasets** and report the result
    separately. This guards against overfitting the prompts to the benchmark.
40. [ ] **Checkpoint 7:** one headline sentence plus one results table, e.g. "normalized score
    0.71 ± 0.08; beats default LightGBM on 9/12 (6 significant), loses to FLAML on 5/12;
    strategy escalation adds +0.12 at 1.4× cost; agents attempted metric gaming in 3% of
    attempts, reduced to 0.5% with the integrity Critic." The numbers here are placeholders;
    report whatever is actually measured.

## Phase 8 — Production Serving + Human-in-the-Loop

41. [ ] FastAPI: `POST /runs` (upload CSV + task), `GET /runs/{id}`, and `GET /runs/{id}/events`
    (SSE stream of graph events). Run status lives in the SQLite checkpointer.
42. [ ] HITL: optional LangGraph `interrupt` before executing a new strategy. The API exposes
    approve/reject.
43. [ ] Dockerfile. Generated code runs in a container with `--network none` and resource
    limits.
44. [ ] Gradio UI calling the API: live node-by-node progress, strategy history, a
    baseline-vs-agent chart, and download of the final script and predictions.
45. [ ] Deploy to Hugging Face Spaces, with a demo-mode budget cap to protect the API key.
46. [ ] **Checkpoint 8:** a stranger can open the public link, drop in a CSV and watch the
    agents work, and `docker run` reproduces it locally.

## Phase 9 — Technical Report and Publication

47. [ ] Technical report, 6–8 pages: problem, related work, system design, evaluation protocol,
    results, ablations, failure taxonomy, specification-gaming study, limitations.
    Arxiv-quality figures.
48. [ ] README:
    - one-paragraph pitch;
    - architecture diagram;
    - the headline result table;
    - a GIF of the demo;
    - a "how we prevent the agent from grading itself" section;
    - reproduce-in-one-command instructions.
49. [ ] Blog post (Medium/Substack/LinkedIn) summarising the gaming and ablation findings. This
    is the readable summary of the findings.
50. [ ] Clean commit history, a tagged release, and a public repo.

## Tier 3 — Stretch (only after Phase 9)

- [ ] Run a small **MLE-bench Lite** subset (2–3 tabular competitions) for comparability with
  published agent numbers.
- [ ] Add an open-weight model (e.g. Qwen 2.5 7B via Ollama or a hosted GPU) to ablation F for a
  closed-vs-open comparison.
- [ ] Submit the integrity/ablation findings to an agents or evaluation workshop at a major ML
  conference.
