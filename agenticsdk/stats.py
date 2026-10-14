"""Phase 7 statistics: turn benchmark run reports into tables, confidence intervals and paired tests.

    python -m agenticsdk.stats summary                      # one row per variant
    python -m agenticsdk.stats compare A_full E_no_memory   # paired comparison of two variants
    python -m agenticsdk.stats baselines                    # datasets won/lost against B0/B1/B2 with sign-test p-values

Unit of analysis: one (dataset, seed) run. Variants are compared on the SAME (dataset, seed) pairs only.
Normalized score: 0 = default random forest (B0), 1 = the stronger of B1/B2, on the harness test split.

Honesty notes encoded here:
* bootstrap CIs resample whole datasets (clusters), not individual seeds, because the 3 seeds of one
  dataset are not independent samples of "a dataset";
* Wilcoxon signed-rank tests run on per-dataset mean differences (n = number of datasets), since with 3 seeds a
  per-dataset test can never reach p < 0.25 and a "significant per-dataset win" would be an artefact;
* a run that failed to produce a valid score counts as the worst case (normalized score taken as the minimum
  observed in that comparison), never silently dropped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
VARIANT_ORDER = ["A_full", "B_no_escalation", "C_react", "D_no_tools", "E_no_memory", "F_flash"]


def load_runs(prefix: str = "bench_", runs_dir: Path = RUNS_DIR, group: str | None = "dev") -> pd.DataFrame:
    """Benchmark runs as one row each. ``group``: "dev" (the 12 development datasets, default), "unseen" (the 3 final
    never-touched datasets, reported separately) or None (everything)."""
    from agenticsdk.harness.datasets import REGISTRY
    rows = []
    for rep in sorted(runs_dir.glob(f"{prefix}*/report.json")):
        d = json.loads(rep.read_text(encoding="utf-8"))
        name = rep.parent.name[len(prefix):]
        variant, rest = next(((v, name[len(v) + 1:]) for v in VARIANT_ORDER if name.startswith(v + "_")), (None, None))
        if variant is None:
            continue
        dataset, _, seed = rest.rpartition("_s")
        if group is not None and (REGISTRY[dataset].group if dataset in REGISTRY else "dev") != group:   # custom = dev
            continue
        u = d.get("llm_usage", {})
        bt = d.get("baselines_test") or {}
        test = d.get("final_test_score")
        rows.append({
            "variant": variant, "dataset": dataset, "seed": int(seed), "status": d["status"],
            "valid": test is not None, "test": test, "norm": d.get("final_normalized"),
            "attempts": len(d.get("attempts", [])), "strategies": len(d.get("strategy_history", [])),
            "calls": u.get("calls", 0), "tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
            "cost": (u.get("cost_usd_live", 0.0) or 0.0) + (u.get("cost_usd_replayed", 0.0) or 0.0),
            "retries": u.get("retries", 0),
            "beats_B0": (test > bt["B0"]) if test is not None and "B0" in bt else False,
            "beats_B1": (test > bt["B1"]) if test is not None and "B1" in bt else False,
            "beats_B2": (test > bt["B2"]) if test is not None and "B2" in bt else False,
            "seconds": _wall_seconds(rep.parent),
        })
    return pd.DataFrame(rows)


def _wall_seconds(run_dir: Path) -> float:
    """Wall-clock of a run: LLM latency plus script execution time recorded in the run's own logs."""
    t = 0.0
    f = run_dir / "llm_calls.jsonl"
    if f.exists():
        t += sum(json.loads(ln).get("latency_s", 0.0) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
    ev = run_dir / "events.jsonl"
    if ev.exists():
        for ln in ev.read_text(encoding="utf-8").splitlines():
            e = json.loads(ln)
            if e.get("node") == "execute":
                t += e.get("data", {}).get("duration", 0.0) or 0.0
    return round(t, 1)


def _filled(df: pd.DataFrame) -> pd.DataFrame:
    """Failed runs (no valid score) get the worst normalized score seen, so they hurt instead of vanishing."""
    out = df.copy()
    worst = out["norm"].min() if out["norm"].notna().any() else 0.0
    out["norm"] = out["norm"].fillna(worst)
    return out


def bootstrap_ci(df: pd.DataFrame, col: str = "norm", n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile CI, resampling datasets (clusters) with replacement."""
    per = df.groupby("dataset")[col].mean().to_numpy()
    if len(per) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n)]
    return float(per.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in [x for x in VARIANT_ORDER if x in set(df["variant"])]:
        g = _filled(df[df["variant"] == v])
        m, lo, hi = bootstrap_ci(g)
        mean_norm = g["norm"].mean()
        rows.append({
            "variant": v, "runs": len(g), "failed": int((~g["valid"]).sum()),
            "norm_mean": m, "ci_lo": lo, "ci_hi": hi,
            "win_B0": g["beats_B0"].mean(), "win_B1": g["beats_B1"].mean(), "win_B2": g["beats_B2"].mean(),
            "attempts": g["attempts"].mean(), "tokens": g["tokens"].mean(), "cost_usd": g["cost"].mean(),
            "seconds": g["seconds"].mean(), "retries": g["retries"].mean(),
            "cost_per_norm_point": (g["cost"].mean() / mean_norm) if mean_norm > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, a: str, b: str) -> dict:
    """Paired comparison of variant b against variant a on the (dataset, seed) pairs both completed."""
    x = _filled(df[df["variant"].isin([a, b])])
    piv = x.pivot_table(index=["dataset", "seed"], columns="variant", values="norm").dropna()
    if piv.empty:
        return {"pairs": 0}
    d = piv[b] - piv[a]
    per_ds = d.groupby("dataset").mean()
    out = {"pairs": len(d), "datasets": len(per_ds), "mean_diff": float(d.mean()),
           "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()), "ties": int((d.abs() <= 1e-9).sum())}
    lo_hi = bootstrap_ci(pd.DataFrame({"dataset": per_ds.index, "norm": per_ds.to_numpy()}))
    out["ci"] = (lo_hi[1], lo_hi[2])
    out["datasets_better"] = int((per_ds > 1e-9).sum())
    out["datasets_worse"] = int((per_ds < -1e-9).sum())
    if len(per_ds) >= 6 and (per_ds.abs() > 1e-9).any():
        out["wilcoxon_p"] = float(sps.wilcoxon(per_ds.to_numpy(), zero_method="wilcox").pvalue)
    else:
        out["wilcoxon_p"] = None
    return out


def vs_baseline(df: pd.DataFrame, variant: str, baseline: str) -> dict:
    """Is a variant better than a frozen baseline? A dataset is a win when the variant beats the baseline on more than
    half of its seeds, a loss when on fewer than half (3 seeds: never a tie; with other seed counts an exact half is
    "mixed" and ignored). ``sign_p`` is the exact two-sided binomial (sign) test of wins against losses over datasets,
    so it needs no common unit between log-loss and R^2; with 12 datasets 10 wins to 2 losses gives p = 0.039."""
    g = df[df["variant"] == variant]
    col = f"beats_{baseline}"
    per = g.groupby("dataset")[col].mean()
    wins, losses = int((per > 0.5).sum()), int((per < 0.5).sum())
    p = float(sps.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else None
    return {"datasets": len(per), "win_datasets": wins, "lose_datasets": losses, "mixed": int((per == 0.5).sum()),
            "sign_p": p}


def render_vs_baselines(df: pd.DataFrame) -> str:
    """Per variant and frozen baseline: datasets won / lost (mean over seeds) and the sign-test p-value."""
    lines = ["Datasets won / lost against the frozen baselines (a dataset is won when the variant beats the baseline on "
             "more than half of its seeds); p = exact two-sided sign test over datasets", ""]
    lines.append(f"{'variant':17s}" + "".join(f"{b:>20s}" for b in ("B0 default RF", "B1 default LightGBM", "B2 FLAML")))
    for v in sorted(set(df["variant"])):
        row = f"{v:17s}"
        for b in ("B0", "B1", "B2"):
            r = vs_baseline(df, v, b)
            ptxt = "n/a" if r["sign_p"] is None else f"{r['sign_p']:.3f}"
            row += f"{r['win_datasets']:>3d}W/{r['lose_datasets']:>2d}L p={ptxt:>5s}".rjust(20)
        lines.append(row)
    return "\n".join(lines)


def render(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no benchmark runs yet)"
    s = summary(df)
    lines = ["Benchmark summary (normalized test score: 0 = default random forest, 1 = best of default LightGBM / FLAML)",
             f"{'variant':17s}{'runs':>5s}{'fail':>5s}{'norm [95% CI over datasets]':>31s}"
             f"{'winB0':>7s}{'winB1':>7s}{'winB2':>7s}{'attempts':>9s}{'tokens':>9s}{'$/run':>8s}{'sec':>6s}{'$/pt':>8s}"]
    for r in s.itertuples():
        lines.append(f"{r.variant:17s}{r.runs:5d}{r.failed:5d}   {r.norm_mean:7.3f} [{r.ci_lo:6.3f}, {r.ci_hi:6.3f}]"
                     f"{r.win_B0:7.0%}{r.win_B1:7.0%}{r.win_B2:7.0%}{r.attempts:9.1f}{r.tokens:9.0f}"
                     f"{r.cost_usd:8.4f}{r.seconds:6.0f}{r.cost_per_norm_point:8.4f}")
    base = "A_full" if "A_full" in set(df["variant"]) else s["variant"].iloc[0]
    others = [v for v in s["variant"] if v != base]
    if others:
        lines += ["", f"Paired comparisons against {base} (same dataset and seed; Wilcoxon over per-dataset means)"]
        for v in others:
            c = compare(df, base, v)
            if not c["pairs"]:
                continue
            p = "n/a" if c["wilcoxon_p"] is None else f"{c['wilcoxon_p']:.3f}"
            lines.append(f"  {v:17s}{c['mean_diff']:+8.3f} (CI {c['ci'][0]:+.3f}..{c['ci'][1]:+.3f})  "
                         f"{c['pairs']} pairs: {c['wins']}W/{c['losses']}L/{c['ties']}T   "
                         f"datasets {c['datasets_better']} better / {c['datasets_worse']} worse   Wilcoxon p={p}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["summary", "compare", "baselines"])
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--prefix", default="bench_")
    ap.add_argument("--group", default="dev", choices=["dev", "unseen", "all"], help="which datasets (default: the 12 dev datasets)")
    a = ap.parse_args()
    df = load_runs(a.prefix, group=None if a.group == "all" else a.group)
    if a.cmd == "summary":
        print(render(df))
    elif a.cmd == "baselines":
        print(render_vs_baselines(df))
    else:
        print(json.dumps(compare(df, *a.variants[:2]), indent=1, default=str))


if __name__ == "__main__":
    main()
