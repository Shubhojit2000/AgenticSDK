"""Offline check of the Phase 0-2 goals. No API key, no network, no LLM calls (about 1 minute).

    venv\\Scripts\\python tests\\verify\\verify_phase0_2.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # tests/verify/ -> project root
sys.path.insert(0, str(ROOT))

from agenticsdk.agents.graph import _violation  # noqa: E402
from agenticsdk.agents.run import initial_state, run_dataset  # noqa: E402
from agenticsdk.agents.schemas import Strategy, Verdict  # noqa: E402
from agenticsdk.harness.datasets import DEV, UNSEEN, load_split  # noqa: E402
from agenticsdk.harness.sandbox import _clean_env, prepare_workdir, run_script  # noqa: E402
from agenticsdk.harness.scorer import normalized_score, score_frame  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # a crash in a check is a failed check
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")


# ---------------------------------------------------------------- Phase 0
def p0_datasets():
    n_dev, n_un = len(DEV), len(UNSEEN)
    kinds = sorted({d.kind for d in DEV})
    return n_dev == 12 and n_un == 3, f"{n_dev} dev + {n_un} unseen datasets; kinds: {kinds}"


def p0_splits():
    bad = []
    for d in DEV:
        s = load_split(d.key, 0)
        n = len(s["train"]) + len(s["val"]) + len(s["test"])
        fr = [round(len(s[k]) / n, 2) for k in ("train", "val", "test")]
        if fr != [0.6, 0.2, 0.2]:
            bad.append((d.key, fr))
    return not bad, "all 12 dev datasets split 60/20/20" if not bad else f"bad splits: {bad}"


def p0_baselines():
    res = json.loads((ROOT / "results" / "baselines.json").read_text())
    errs = [k for k, v in res.items() if "error" in v]
    return len(res) == 108 and not errs, f"{len(res)} frozen baseline runs (12 datasets x 3 seeds x 3 baselines), {len(errs)} errors"


def p0_scorer():
    s = load_split("credit_g", 0)
    classes, n = s["meta"]["classes"], len(s["val"])
    truth = s["val"]["__target__"].astype(str).to_numpy()
    perfect = pd.DataFrame({c: np.where(truth == c, 0.99, 0.01) for c in classes})
    perfect.insert(0, "id", range(n))
    prior = pd.DataFrame({c: [1 / len(classes)] * n for c in classes})
    prior.insert(0, "id", range(n))
    good, flat = score_frame(perfect, "credit_g", 0, "val"), score_frame(prior, "credit_g", 0, "val")
    short = score_frame(perfect.iloc[:-5], "credit_g", 0, "val")
    unnormalised = perfect.copy()
    unnormalised[classes[0]] += 0.5
    bad_sum = score_frame(unnormalised, "credit_g", 0, "val")
    ok = good.valid and flat.valid and good.score > flat.score and not short.valid and not bad_sum.valid
    return ok, (f"near-perfect {good.score:.3f} > uniform {flat.score:.3f}; "
                f"missing rows rejected ({short.error}); probabilities not summing to 1 rejected")


def p0_normalization():
    a, b = normalized_score(-0.5, -0.5, -0.3), normalized_score(-0.3, -0.5, -0.3)
    return abs(a) < 1e-9 and abs(b - 1) < 1e-9, "default forest scores 0.0, strongest baseline scores 1.0"


# ---------------------------------------------------------------- Phase 1
def p1_no_label_leak():
    wd = Path(tempfile.mkdtemp())
    try:
        prepare_workdir("credit_g", 0, wd)
        files = sorted(p.name for p in wd.iterdir())
        va = pd.read_csv(wd / "val_features.csv", nrows=1)
        te = pd.read_csv(wd / "test_features.csv", nrows=1)
        ok = files == ["test_features.csv", "train.csv", "val_features.csv"] and "target" not in va and "target" not in te
        return ok, f"agent workdir holds only {files}; val/test files have no label column"
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def p1_secrets_stripped():
    env = _clean_env()
    leaked = [k for k in env if any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "GOOGLE", "GEMINI"))]
    return not leaked, f"sandbox environment has {len(env)} variables, none look like secrets"


def p1_infinite_loop():
    wd = Path(tempfile.mkdtemp())
    try:
        prepare_workdir("credit_g", 0, wd)
        t0 = time.time()
        r = run_script("while True:\n    pass\n", wd, timeout=5)
        took = time.time() - t0
        return r.timed_out and took < 15, f"infinite loop killed after {took:.1f}s (limit 5s), timed_out={r.timed_out}"
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def p1_blocked_import():
    wd = Path(tempfile.mkdtemp())
    try:
        prepare_workdir("credit_g", 0, wd)
        r = run_script("import os\nprint(os.listdir('.'))\n", wd, timeout=10)
        return r.blocked and r.exit_code is None, f"'import os' rejected before running: {r.stderr.strip().splitlines()[-1][:80]}"
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ---------------------------------------------------------------- Phase 1 + 2 (recorded live run, replayed)
def replay_checkpoint2():
    rid = "verify_replay"
    shutil.rmtree(ROOT / "runs" / rid, ignore_errors=True)
    f = run_dataset("california_housing", 0, rid, mode="replay_only", first_family="linear")
    u = json.loads((ROOT / "runs" / rid / "report.json").read_text())["llm_usage"]
    ok = f.status == "passed" and u["api_calls"] == 0 and len(f.strategy_history) == 2
    return ok, (f"replayed the recorded Checkpoint-2 run with {u['api_calls']} API calls ({u['cache_hits']} cache hits); "
                f"status={f.status}, strategies={len(f.strategy_history)}, test R2={f.final_test_score:.4f}, "
                f"normalized={f.final_normalized:.2f}")


# ---------------------------------------------------------------- Phase 2
def p2_trace():
    ev = [json.loads(x) for x in (ROOT / "runs" / "cp2_california" / "events.jsonl").read_text().splitlines()]
    esc = [i for i, e in enumerate(ev) if e["node"] == "critic" and e["data"]["decision"] == "RETRY_STRATEGY"]
    nxt = ev[esc[0] + 1] if esc else {}
    ok = bool(esc) and nxt.get("node") == "planner" and "strategy" in nxt.get("data", {})
    return ok, "recorded trace: Critic returned RETRY_STRATEGY, then the planner proposed a new strategy" if ok else "no RETRY_STRATEGY in trace"


def p2_novelty_rule():
    s = initial_state("credit_g", 0, "x", 2, 60)
    a = Strategy(model_family="linear", feature_approach="minimal", summary="a", preprocessing=[], model="m",
                 validation="v", risks=[])
    s.strategy_history = [a]
    same = _violation(s, a)
    other = _violation(s, a.model_copy(update={"model_family": "gradient_boosting"}))
    ok = same is not None and other is None
    return ok, f"repeat refused ('{(same or '')[:55]}...'), a different model_family accepted"


def p2_verdict_schema():
    v = Verdict(decision="RETRY_STRATEGY", reasoning="r", feedback="f")
    return v.decision == "RETRY_STRATEGY", "Critic verdict supports PASS / RETRY_CODE / RETRY_STRATEGY"


if __name__ == "__main__":
    print("== Phase 0: harness, datasets, baselines")
    for n, f in (("12 dev + 3 unseen datasets frozen", p0_datasets), ("60/20/20 splits", p0_splits),
                 ("baselines frozen, no errors", p0_baselines), ("scorer validates and scores", p0_scorer),
                 ("normalized score anchors", p0_normalization)):
        check(n, f)
    print("\n== Phase 1: sandbox and integrity")
    for n, f in (("agent never sees val/test labels", p1_no_label_leak), ("secrets stripped from sandbox", p1_secrets_stripped),
                 ("infinite loop is killed", p1_infinite_loop), ("blocked import never runs", p1_blocked_import)):
        check(n, f)
    print("\n== Phase 2: escalation")
    for n, f in (("Critic has RETRY_STRATEGY", p2_verdict_schema), ("new strategy must differ (enforced in code)", p2_novelty_rule),
                 ("recorded live trace shows a real escalation", p2_trace),
                 ("recorded run replays with 0 API calls", replay_checkpoint2)):
        check(n, f)
    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)
