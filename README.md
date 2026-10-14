# AgenticSDK

[![CI](https://github.com/Shubhojit2000/AgenticSDK/actions/workflows/ci.yml/badge.svg)](https://github.com/Shubhojit2000/AgenticSDK/actions/workflows/ci.yml)

A multi-agent system (LangGraph) that builds ML pipelines for tabular data, and an evaluation harness that holds the
validation and test labels itself, so an agent cannot grade its own work. The agents are the thing under test; the
harness, the frozen baselines and the ablation study are the point of the project.

**What it found** (12 datasets x 3 seeds x 5 variants = 180 runs on free Gemini models; full tables in
[`results/checkpoint7.txt`](results/checkpoint7.txt)):

- The full system beats a default random forest on 12/12 datasets and default LightGBM on 11/12, and **ties** a
  60-second FLAML run (6 won, 6 lost). It is not a better AutoML tool than FLAML.
- Letting the agents change strategy after a failure (strategy escalation) is the one component with a significant
  effect (+0.62 normalized points, p = 0.034). Tools, memory and the multi-agent split did **not** show a significant
  benefit with 12 datasets.
- On 3 never-touched datasets: 0.83 normalized score, no failed run. The integrity monitor flagged none of 750 attempts.
- Limits: 3 seeds per dataset, one model family, a wide confidence interval, and one ablation (a larger model) that the
  free quota did not allow. See "Limitations" below.

## Try it

**1. See a real run without any API key** (about 10 seconds):

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell (Git Bash: source venv/Scripts/activate)
# source venv/bin/activate     # Linux / macOS
pip install -e ".[serve]"
python -m agenticsdk.agents.show examples/recorded_run_credit_g
```

It prints the planner's tool calls, each strategy, the scripts the coding agent wrote, the critic's verdicts and the
final held-out score against the three baselines (`examples/recorded_run_credit_g/attempt_*/solution.py` has the code).

**2. Run the live demo yourself** (needs a free Gemini key, no billing; a run takes about 3 to 10 minutes):

```bash
cp .env.example .env             # then put your key in .env: GOOGLE_API_KEY=...   (https://aistudio.google.com/apikey)
python -m agenticsdk.service     # open http://127.0.0.1:7860/ui
```

In the page: click the example row (German credit data), press **Start**. You watch the baselines being fitted, then
each strategy and attempt as it happens, a chart of the agents against the baselines on rows they never saw, and you
can download the final script and its predictions. Tick "Ask me to approve each strategy" to approve or veto what the
planner proposes before it runs. Any CSV works: pick the column to predict and the service holds back validation and
test rows itself. If the key is missing, the page says so at the top.

**3. Run the benchmark datasets from the command line** (downloads 15 OpenML datasets once):

```bash
python -m agenticsdk.harness.datasets
python -m agenticsdk.agents.run --dataset california_housing --first-family linear --run-id demo
python -m agenticsdk.agents.show demo
python -m agenticsdk.agents.run --dataset eucalyptus --use-tools --run-id demo_tools
```

## How the agent is stopped from grading itself

- The harness (`agenticsdk/harness/`) owns the validation and test labels. The agent's script only writes predictions;
  the harness scores them. The agent never reads a label file for the held-out rows.
- Each script runs in a fresh temp directory with a stripped environment, a wall-clock timeout (process-tree kill) and
  an AST allow-list. The allow-list is defense in depth, not a security boundary; `AGENTICSDK_SANDBOX=docker` runs scripts
  in a container with no network (written and unit-tested, never run against a real Docker daemon by the author).
- The Critic sees validation scores and the three baselines' validation scores, never test scores.
- An integrity monitor checks each script statically for gaming (reading a decoy label file, printing a metric as a
  literal, fitting on test features with self-made labels, near-copies of the target). Flagged attempts are rejected or
  recorded and can never be reported as the best result. Heuristics over the AST, not a guarantee.
- Memory search never returns a card from the dataset being solved (leave-one-dataset-out, enforced in code).

## Benchmark results

12 development datasets x 3 seeds, free Gemini models; normalized score: 0 = default random forest, 1 = the better of
default LightGBM and a 60-second FLAML run. Intervals are bootstrapped over datasets.

| variant | normalized score [95% CI] | beats RF / LightGBM / FLAML (datasets of 12) | USD-equivalent per run |
|---|---|---|---|
| A full system (graph + tools + memory) | 0.84 [0.51, 1.14] | 12 / 11 / 6 | 0.0088 |
| B no strategy escalation | 0.01 [-0.66, 0.65] | 8 / 8 / 6 | 0.0063 |
| C single ReAct agent | 0.95 [0.16, 1.71] | 11 / 11 / 7 | 0.0082 |
| D no tools | 1.01 [0.57, 1.47] | 11 / 11 / 7 | 0.0060 |
| E no memory | 0.62 [0.13, 1.05] | 9 / 11 / 7 | 0.0071 |

Against variant E (graph + tools, no memory), only strategy escalation makes a significant difference (+0.62, Wilcoxon
p = 0.034); tools, a single ReAct agent and memory are not distinguishable from it with 12 datasets. On three
never-touched datasets the full system scores 0.83 [0.44, 1.05] with no failed run.

Reproduce a table from the saved runs: `python -m agenticsdk.stats summary`, `python -m agenticsdk.stats baselines`.
Re-running the whole grid costs about 2,700 model calls (about a week of one free key, see `python -m agenticsdk.bench plan`).

## Limitations

- 12 datasets and 3 seeds give wide intervals; per-dataset significance is deliberately not claimed.
- The normalized score is unstable on datasets where the baselines nearly tie (abalone, eucalyptus), so the win counts are
  the safer summary.
- The benchmark used two free models (one per dataset group, so paired comparisons are not confounded); absolute scores and
  times mix the two. The larger-model ablation was not run (its free quota is about 20 calls a day).
- The failure-taxonomy labels used to check the LLM judge were drafted by the assistant that helped build the project, not by
  an independent human (`results/taxonomy/LABELLED_BY.txt`); the agreement figure (kappa 0.82) should be read with that in mind.
- Cross-run memory was evaluated twice (a pilot and the benchmark) and showed no significant benefit.
- No hosted demo: the service is meant to be run locally. `deploy/huggingface/` assembles a Space but it has not been deployed.

## Layout

```
agenticsdk/
  llm.py            LLM client: retries, rate limiting, record/replay cache, token and cost accounting
  memory.py         cross-run memory: experience cards in a Chroma store, retrieval, `python -m agenticsdk.memory`
  tracing.py        optional Langfuse tracing (no-op without keys); `python -m agenticsdk.tracing` checks your keys
  gaming_study.py   the honeypot / prompt-condition study of metric-gaming attempts
  bench.py, stats.py, taxonomy.py   benchmark runner and ablation variants, bootstrap / Wilcoxon statistics, failure taxonomy
  service/          FastAPI app (+SSE, human approval), the per-run worker process, the Gradio UI
  harness/          the trusted side: datasets, splits, scorer, baselines, the sandbox that runs agent code, and the integrity checks
  agents/           the agents: schemas, prompts, the LangGraph graph, the run command, the trace viewer
  tools/            the MCP servers the agents call through native function calling (dataset tools; memory search)
examples/           an example CSV and one recorded run you can read without an API key
tests/              scripted-LLM tests (no API key needed); tests/verify/ has local replay checks
docs/DESIGN.md      design, evaluation protocol, and the measured result at each checkpoint
results/            frozen baselines and the checkpoint results
```

## Reading order (about 2,500 lines of core source)

1. `harness/datasets.py`, `harness/scorer.py`: what is evaluated and how a score is computed. The agent never sees val/test labels.
2. `harness/sandbox.py`, `harness/allowlist.py`: how agent-written scripts run (temp dir, stripped environment, timeout with process-tree kill, AST allow-list).
3. `agents/schemas.py`, `agents/prompts.py`: the data the agents pass around and what they are told.
4. `agents/graph.py`: the whole loop, planner -> feature -> training -> execute -> critic -> report, with budgets enforced in code.
5. `llm.py`: the model client; `agents/run.py`: the command that runs one dataset.
6. `tools/`: the MCP server (`server.py`), its tools (`dataset_tools.py`) and the client the graph uses (`client.py`).

## What is in the box

- **Harness:** 12 development and 3 unseen OpenML datasets, frozen 60/20/20 splits, scoring (log-loss / R^2), and three
  frozen baselines (default RandomForest, default LightGBM, FLAML).
- **Agents:** planner -> feature -> training -> sandboxed execution -> critic, with strategy escalation, budgets enforced in
  code, and resumable runs.
- **Tools:** an MCP server (dataset profile, column inspection, quick cross-validation) that the agents call through native
  function calling (`--use-tools`).
- **Memory (optional):** finished runs become experience cards in a Chroma store; the planner can search cards from *other*
  datasets (`--use-memory`) and must cite the cards it used.
- **Integrity monitor, evaluation tooling, web service:** as described above.
- **Observability:** optional Langfuse tracing; JSON-lines logs are always written.

## Memory (optional)

```bash
python -m agenticsdk.memory ingest demo demo_tools     # turn finished runs into cards (embeds them with Gemini)
python -m agenticsdk.memory list
python -m agenticsdk.agents.run --dataset eucalyptus --use-tools --use-memory --run-id with_memory
```

Cards from the three final-unseen datasets are refused unless you pass `--allow-unseen`.

## Web service details

```bash
DEMO_MODE=1 uvicorn agenticsdk.service.app:app --port 7860   # public-demo caps: 5 MB, 20k rows, 4 scripts, 1 run at a time
```

An upload becomes a custom dataset: the service splits it 60/20/20, keeps the validation and test labels server-side like the
benchmark does, fits the three baselines, then starts the agent graph in its own process. API docs are at `/docs`.
`Dockerfile` builds the service itself; `Dockerfile.sandbox` builds the script sandbox image
(`docker build -f Dockerfile.sandbox -t agenticsdk-sandbox .`); neither has been built by the author.

## Integrity study and benchmark commands

```bash
python -m agenticsdk.agents.run --dataset credit_g --integrity enforce --honeypot      # a decoy label file is in the sandbox
python -m agenticsdk.gaming_study run --datasets credit_g car abalone; python -m agenticsdk.gaming_study report
python -m agenticsdk.bench plan                                                         # what the 216-run grid still needs
python -m agenticsdk.bench run --variants E_no_memory --max-live-calls 450               # resumable; stops before the daily quota
python -m agenticsdk.stats summary                                                      # table of variants, --group unseen for the final datasets
python -m agenticsdk.taxonomy report; python -m agenticsdk.taxonomy kappa               # failure taxonomy and judge agreement
```

## Tests (no API key needed)

```bash
pip install -e ".[dev,serve]"
ruff check .
pytest --cov
```

Every LLM call in the tests is scripted, so the suite needs no key and sends nothing anywhere.
The tests download two small OpenML datasets on first use. `tests/verify/verify_phase0_2.py` and `tests/verify/verify_phase3.py`
replay recorded live runs; those replays depend on numerically identical results, so they are
meant for the machine that recorded them and are not part of CI.

## Tracing (optional)

Put `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (and `LANGFUSE_HOST`) in `.env`, then run
`python -m agenticsdk.tracing` to check them. Each run becomes one trace: `agent_run` -> node spans
(planner, feature, training, execute, critic, report) -> LLM generations (tokens, USD-equivalent cost,
cache hit or live) and tool calls. Without keys nothing is sent.
