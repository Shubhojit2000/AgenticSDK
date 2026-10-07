"""Run the agent graph on one (dataset, seed).

    python -m agenticsdk.agents.run --dataset credit_approval --seed 0 --retries 3
    python -m agenticsdk.agents.run --dataset credit_approval --seed 0 --run-id demo2 --replay-only  # zero API calls
    python -m agenticsdk.agents.run --dataset california_housing --first-family linear --run-id cp2   # engineered bad start
    python -m agenticsdk.agents.run --dataset california_housing --run-id cp2 --resume                # continue after a crash
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agenticsdk.agents.graph import build_graph
from agenticsdk.agents.schemas import AgentState
from agenticsdk.harness.baselines import BUDGET_SECONDS, load_results
from agenticsdk.harness.datasets import REGISTRY, load_split
from agenticsdk.harness.profile import profile_text
from agenticsdk.llm import DEFAULT_MODEL, LLMClient
from agenticsdk.tools.client import MCPToolClient
from agenticsdk.tracing import get_tracer

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = ROOT / "runs"


def initial_state(key: str, seed: int, run_id: str, max_code_retries: int, budget: int,
                  max_strategy_retries: int = 2, max_total_attempts: int = 6,
                  first_family: str | None = None, use_tools: bool = False) -> AgentState:
    spec = REGISTRY[key]
    data = load_split(key, seed)
    res = load_results()
    base_val, base_test = {}, {}
    for b in ("B0", "B1", "B2"):
        r = res.get(f"{key}|{seed}|{b}")
        if r and "error" not in r:
            base_val[b], base_test[b] = r["val"], r["test"]
    return AgentState(
        run_id=run_id, dataset_key=key, seed=seed, kind=spec.kind, metric_name=spec.metric,
        classes=data["meta"]["classes"],
        task_description=f"Build the best predictive model for the tabular dataset '{key}': predict the "
                         f"column 'target' from all other columns.",
        profile=profile_text(data["train"], spec.kind, data["meta"]["classes"]),
        budget_seconds=budget, max_code_retries=max_code_retries, max_strategy_retries=max_strategy_retries,
        max_total_attempts=max_total_attempts, first_family=first_family, use_tools=use_tools,
        baselines_val=base_val, baselines_test=base_test,
    )


def run_dataset(key: str, seed: int = 0, run_id: str | None = None, mode: str = "cache",
                max_code_retries: int = 2, budget: int = BUDGET_SECONDS, model: str = DEFAULT_MODEL,
                max_strategy_retries: int = 2, max_total_attempts: int = 6, first_family: str | None = None,
                resume: bool = False, use_tools: bool = False) -> AgentState:
    """Run (or, with resume=True, continue) one agent run. Progress is checkpointed after every node in
    runs/<run_id>/checkpoints.sqlite, so a crash or quota stall loses at most the node in flight."""
    run_id = run_id or f"{key}_s{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    db = run_dir / "checkpoints.sqlite"
    if db.exists() and not resume:
        raise FileExistsError(f"{run_dir} already has a checkpoint; pass resume=True/--resume or use a new run id")
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = LLMClient(run_dir=run_dir, mode=mode, model=model)
    conn = sqlite3.connect(db, check_same_thread=False)
    mcp = MCPToolClient(key, seed).start() if use_tools else None   # separate server process, train split only
    tracer = get_tracer()
    try:
        with tracer.session(run_id, tags=[key, f"seed{seed}"] + (["tools"] if use_tools else []),
                            metadata={"model": model, "mode": mode}), \
                tracer.observe("agent_run", as_type="agent",
                               input={"dataset": key, "seed": seed, "model": model, "use_tools": use_tools,
                                      "max_code_retries": max_code_retries,
                                      "max_strategy_retries": max_strategy_retries,
                                      "max_total_attempts": max_total_attempts}) as root:
            graph = build_graph(llm, run_dir, checkpointer=SqliteSaver(conn), tools=mcp)
            cfg = {"configurable": {"thread_id": run_id}, "recursion_limit": 100}
            if resume and graph.get_state(cfg).values:
                out = graph.invoke(None, config=cfg)          # continue from the last checkpoint
            else:
                out = graph.invoke(initial_state(key, seed, run_id, max_code_retries, budget,
                                                 max_strategy_retries, max_total_attempts, first_family,
                                                 use_tools), config=cfg)
            final = AgentState.model_validate(out) if isinstance(out, dict) else out
            root.update(output={"status": final.status, "attempts": len(final.attempts),
                                "strategies": len(final.strategy_history),
                                "final_val_score": final.final_val_score,
                                "final_test_score": final.final_test_score,
                                "final_normalized": final.final_normalized, "llm_usage": llm.usage()})
            if final.final_normalized is not None:
                root.score("normalized_score", final.final_normalized)
            url = root.trace_url()
            if url:
                print("Langfuse trace:", url)
    finally:
        conn.close()
        if mcp:
            mcp.close()
        tracer.flush()
    return final


def _summarize(final: AgentState, llm_usage: dict | None = None) -> str:
    base = ", ".join(f"{k}={final.baselines_test[k]:.4f}" for k in sorted(final.baselines_test))
    lines = [f"run_id={final.run_id}  status={final.status}  attempts={len(final.attempts)}  "
             f"strategies={len(final.strategy_history)}",
             f"val/test (harness): {final.final_val_score} / {final.final_test_score}",
             f"baselines (test): {base}", f"normalized score: {final.final_normalized}"]
    for a in final.attempts:
        lines.append(f"  attempt {a.index} (strategy {a.strategy_index}): val={a.val_score} verdict={a.verdict_decision}({a.verdict_source}) "
                     f"{'ERR: ' + a.error[:100].replace(chr(10), ' ') if a.error else ''}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retries", type=int, default=2, help="code retries per strategy")
    ap.add_argument("--strategy-retries", type=int, default=2)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--first-family", help="force the first strategy's model_family (experiment knob)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use-tools", action="store_true", help="planner/feature agents investigate via MCP tools")
    ap.add_argument("--run-id")
    ap.add_argument("--replay-only", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    mode = "replay_only" if a.replay_only else "no_cache" if a.no_cache else "cache"
    final = run_dataset(a.dataset, a.seed, a.run_id, mode, a.retries, model=a.model,
                        max_strategy_retries=a.strategy_retries, max_total_attempts=a.max_attempts,
                        first_family=a.first_family, resume=a.resume, use_tools=a.use_tools)
    print(_summarize(final))
    rep = RUNS_DIR / final.run_id / "report.json"
    if rep.exists():
        print("llm usage:", json.loads(rep.read_text())["llm_usage"])
