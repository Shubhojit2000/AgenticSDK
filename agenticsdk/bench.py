"""Phase 7 benchmark runner: 12 dev datasets x 3 seeds x variants, unattended and resumable.

    python -m agenticsdk.bench plan                                   # what is done, what is left, estimated calls
    python -m agenticsdk.bench run --variants E_no_memory D_no_tools --max-live-calls 450
    python -m agenticsdk.bench cards                                  # build the memory store for A_full from E_no_memory runs
    python -m agenticsdk.bench unseen --variant A_full                # final run on the 3 never-touched datasets

Variants (same datasets, seeds, per-run budget; no per-dataset prompt changes, ever):
    A_full           multi-agent graph + MCP tools + cross-run memory (cards from OTHER datasets' E runs)
    B_no_escalation  tools, but RETRY_STRATEGY is not allowed: code retries only (same 6-attempt budget)
    C_react          one ReAct agent with the same tools, sandbox and attempt budget
    D_no_tools       multi-agent graph, no tools (pure code generation)
    E_no_memory      multi-agent graph + tools, no memory
    F_flash          like E_no_memory on a larger model (default gemini-3.8-flash)

A run is skipped when its report.json exists, so a quota stall or a crash costs nothing on restart.
``--max-live-calls`` stops the loop before the free daily quota is spent (live = non-cached calls today).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agenticsdk.harness.datasets import DEV, SEEDS, UNSEEN
from agenticsdk.llm import DEFAULT_MODEL, quota_day_start

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
BENCH_MEMORY = ROOT / "memory" / "bench"
CALLS_PER_RUN = {"A_full": 24, "B_no_escalation": 20, "C_react": 14, "D_no_tools": 18, "E_no_memory": 22, "F_flash": 22}
FLASH_MODEL = "gemini-3.8-flash"

VARIANTS = {
    "A_full": dict(use_tools=True, use_memory=True),
    "B_no_escalation": dict(use_tools=True, max_strategy_retries=0, max_code_retries=5),
    "C_react": dict(react=True),
    "D_no_tools": dict(),
    "E_no_memory": dict(use_tools=True),
    "F_flash": dict(use_tools=True, model=FLASH_MODEL),
}


def run_id(variant: str, ds: str, seed: int) -> str:
    return f"bench_{variant}_{ds}_s{seed}"


def done(variant: str, ds: str, seed: int) -> bool:
    return (RUNS_DIR / run_id(variant, ds, seed) / "report.json").exists()


def live_calls_today(since: float | None = None, model: str | None = None) -> int:
    """Non-cached LLM calls recorded in any run directory since the free quota last reset (midnight Pacific).
    ``model``: count only that model's calls (each model has its own daily quota)."""
    since = since or quota_day_start()
    n = 0
    for f in RUNS_DIR.glob("*/llm_calls.jsonl"):
        if f.stat().st_mtime < since:
            continue
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                c = json.loads(ln)
                if not c.get("cached") and c.get("ts", 0) >= since and (model is None or c.get("model") == model):
                    n += 1
    return n


def run_one(variant: str, ds: str, seed: int, model: str | None = None) -> None:
    kw = dict(VARIANTS[variant])
    rid = run_id(variant, ds, seed)
    if kw.pop("react", False):
        from agenticsdk.agents.react import run_react
        run_react(ds, seed, rid, **({"model": model} if model else {}))
        return
    from agenticsdk.agents.run import run_dataset
    if model:
        kw["model"] = model
    if kw.get("use_memory"):
        kw["memory_dir"] = BENCH_MEMORY
    kw.setdefault("max_code_retries", 2)
    # a run that was killed part-way (power cut, restart of the unattended benchmark) continues from its checkpoint
    kw["resume"] = (RUNS_DIR / rid / "checkpoints.sqlite").exists()
    run_dataset(ds, seed, rid, **kw)


def grid(variants, datasets, seeds):
    for seed in seeds:
        for ds in datasets:
            for v in variants:
                yield v, ds, seed


def cmd_plan(variants, datasets, seeds) -> None:
    left = {}
    for v, ds, s in grid(variants, datasets, seeds):
        if not done(v, ds, s):
            left[v] = left.get(v, 0) + 1
    total = len(list(grid(variants, datasets, seeds)))
    print(f"{total} runs in the grid; live calls since the quota reset: {live_calls_today()}")
    for v in variants:
        n = left.get(v, 0)
        print(f"  {v:17s} left {n:3d}/{len(datasets) * len(seeds)}   ~{n * CALLS_PER_RUN[v]:5d} calls")
    print(f"  total ~{sum(left.get(v, 0) * CALLS_PER_RUN[v] for v in variants)} calls; free quota is about 500 per model per day")


def cmd_run(variants, datasets, seeds, max_live: int, model: str | None) -> None:
    for v, ds, s in grid(variants, datasets, seeds):
        if done(v, ds, s):
            continue
        used = live_calls_today(model=(model if v != "F_flash" else (model or FLASH_MODEL)) or DEFAULT_MODEL)
        if used + CALLS_PER_RUN[v] > max_live:
            print(f"stopping: {used} live calls since the quota reset, next run needs ~{CALLS_PER_RUN[v]}, limit {max_live}", flush=True)
            return
        print(f"=== {time.strftime('%H:%M:%S')} {run_id(v, ds, s)}", flush=True)
        try:
            run_one(v, ds, s, model if v != "F_flash" else (model or FLASH_MODEL))
        except Exception as e:  # noqa: BLE001 - one bad run must not end an unattended benchmark; it is retried next time
            print("FAILED", run_id(v, ds, s), type(e).__name__, str(e)[:200], flush=True)
            if "QuotaExhausted" in type(e).__name__:
                return


def cmd_cards(datasets, seeds, embedder: str = "gemini") -> None:
    """Ingest the cards of the finished E_no_memory runs (DEV datasets only) into the store A_full searches."""
    from agenticsdk.memory import ExperienceStore, build_card, get_embedder
    store = ExperienceStore(BENCH_MEMORY, get_embedder(embedder))
    n = 0
    for ds in datasets:
        for s in seeds:
            d = RUNS_DIR / run_id("E_no_memory", ds, s)
            if (d / "report.json").exists():
                store.add(build_card(d))
                n += 1
    print(f"ingested {n} cards; store now holds {len(store)} ({BENCH_MEMORY})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "cards", "unseen"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--variant", default="A_full", choices=list(VARIANTS))
    ap.add_argument("--datasets", nargs="+", default=[d.key for d in DEV])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--max-live-calls", type=int, default=450)
    ap.add_argument("--model")
    ap.add_argument("--embedder", default="gemini", choices=["gemini", "hash"])
    a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan(a.variants, a.datasets, a.seeds)
    elif a.cmd == "run":
        cmd_run(a.variants, a.datasets, a.seeds, a.max_live_calls, a.model)
    elif a.cmd == "cards":
        cmd_cards(a.datasets, a.seeds, a.embedder)
    else:   # the final-unseen evaluation: one variant, once, reported separately
        cmd_run([a.variant], [d.key for d in UNSEEN], a.seeds, a.max_live_calls, a.model)


if __name__ == "__main__":
    main()
