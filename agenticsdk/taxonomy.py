"""Phase 7 failure taxonomy: what goes wrong in failed attempts, hand-labelled on a sample, then labelled at scale
by an LLM judge whose agreement with the hand labels is measured (Cohen's kappa).

    python -m agenticsdk.taxonomy export --n 100          # sample failed attempts -> results/taxonomy/to_label.csv
        ... fill the `label` column of that file by hand (see CATEGORIES) ...
    python -m agenticsdk.taxonomy judge                   # LLM judge labels ALL failed attempts (cached, resumable)
    python -m agenticsdk.taxonomy kappa                   # agreement of the judge with the hand labels
    python -m agenticsdk.taxonomy report                  # counts per category (hand labels where present, else judge)

A "failed attempt" is one with no valid scored predictions, or a valid one scoring below the weakest frozen baseline
on validation. The judge sees only what the Critic saw (error text, stdout, score, baselines) plus the script.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
OUT_DIR = ROOT / "results" / "taxonomy"

CATEGORIES = {
    "import_or_rule_violation": "import not allowed, banned name/path, rejected by the allow-list or integrity monitor",
    "api_hallucination": "calls a function, attribute or keyword argument that does not exist in the library",
    "schema_mismatch": "predictions file wrong: columns, class order, id, probabilities not valid or not summing to 1",
    "data_handling_error": "crash from column names, dtypes, NaN, categorical handling, shape mismatches",
    "timeout": "exceeded the wall-clock limit",
    "no_output": "ran but wrote no predictions, or the reply had no code block",
    "leakage_or_gaming": "used labels it should not, or validation numbers are implausibly good because of leakage",
    "wrong_metric_or_objective": "optimises or outputs the wrong thing for the metric (hard labels for log-loss, ...)",
    "bad_strategy": "valid and honest but the model/feature approach is too weak for this data",
    "other": "none of the above",
}
Category = Literal[tuple(CATEGORIES)]   # type: ignore[valid-type]


class Judgement(BaseModel):
    category: Category = Field(description="The single best category for the root cause of this failed attempt.")
    reason: str = Field(description="One sentence naming the evidence (an error line or a number).")


def failed_attempts(runs_dir: Path = RUNS_DIR, prefix: str = "bench_") -> list[dict]:
    items = []
    for rep in sorted(runs_dir.glob(f"{prefix}*/report.json")):
        d = json.loads(rep.read_text(encoding="utf-8"))
        base_val = d.get("baselines_val") or {}
        floor = min(base_val.values()) if base_val else None
        for a in d.get("attempts", []):
            bad = bool(a.get("error")) or (a.get("val_valid") and floor is not None and a["val_score"] < floor)
            if not bad:
                continue
            code = ""
            cf = rep.parent / f"attempt_{a['index']}" / "solution.py"
            if cf.exists():
                code = cf.read_text(encoding="utf-8")[:6000]
            ex = a.get("exec", {})
            items.append({"id": f"{rep.parent.name}#{a['index']}", "dataset": d["dataset_key"], "kind": d["kind"],
                          "metric": d["metric_name"], "error": (a.get("error") or "")[:1800],
                          "stdout": (ex.get("stdout") or "")[:800], "val_score": a.get("val_score"),
                          "baselines_val": base_val, "integrity": a.get("integrity", []), "code": code})
    return items


def export_sample(n: int, seed: int = 0, force: bool = False) -> Path:
    path = OUT_DIR / "to_label.csv"
    if path.exists() and not force:   # the file holds hand labels: never overwrite them by accident
        raise FileExistsError(f"{path} already exists (it may hold your labels); pass --force to replace it")
    items = failed_attempts()
    random.Random(seed).shuffle(items)
    sample = items[:n]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "error_or_score", "label", "script"])
        for it in sample:
            run, idx = it["id"].split("#")
            w.writerow([it["id"], it["dataset"], (it["error"] or f"valid, val={it['val_score']}, baselines={it['baselines_val']}")
                        .replace("\n", " | ")[:600], "", f"runs/{run}/attempt_{idx}/solution.py"])
    print(f"{len(items)} failed attempts in total; wrote {len(sample)} to {path}")
    print("Fill the `label` column with one of:", ", ".join(CATEGORIES))
    return path


JUDGE_SYSTEM = ("You classify why one attempt of an automated ML-engineering agent failed. Categories:\n"
                + "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
                + "\nChoose the category of the ROOT cause shown by the evidence, not a downstream symptom.")


def judge_all(model: str | None = None) -> Path:
    from agenticsdk.llm import DEFAULT_MODEL, LLMClient
    llm = LLMClient(run_dir=None, model=model or DEFAULT_MODEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "judge_labels.jsonl"
    seen = {json.loads(ln)["id"] for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()} if path.exists() else set()
    with open(path, "a", encoding="utf-8") as f:
        for it in failed_attempts():
            if it["id"] in seen:
                continue
            prompt = (f"Dataset {it['dataset']} ({it['kind']}, metric {it['metric']}).\n"
                      f"Error / harness message:\n{it['error'] or '(none: the script ran and produced valid predictions)'}\n"
                      f"Script stdout:\n{it['stdout'] or '(empty)'}\nValidation score: {it['val_score']}; "
                      f"baselines: {it['baselines_val']}; integrity flags: {it['integrity']}\n"
                      f"Script:\n```python\n{it['code'] or '(not available)'}\n```")
            j = llm.structured(Judgement, JUDGE_SYSTEM, prompt, "taxonomy", thinking_budget=512)
            f.write(json.dumps({"id": it["id"], "category": j.category, "reason": j.reason}) + "\n")
            f.flush()
    return path


def _read_labels(path: Path, col: str) -> dict[str, str]:
    if path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return {r["id"]: r[col].strip() for r in csv.DictReader(f) if r[col].strip()}
    return {d["id"]: d[col] for d in (json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two label lists of equal length."""
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("need two non-empty label lists of equal length")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def kappa_report() -> str:
    hand = _read_labels(OUT_DIR / "to_label.csv", "label")
    judge = _read_labels(OUT_DIR / "judge_labels.jsonl", "category")
    both = sorted(set(hand) & set(judge))
    bad = sorted(h for h in hand.values() if h not in CATEGORIES)
    if bad:
        raise ValueError(f"unknown hand labels: {sorted(set(bad))}")
    if not both:
        return "no overlap between hand labels and judge labels yet"
    k = cohen_kappa([hand[i] for i in both], [judge[i] for i in both])
    agree = sum(hand[i] == judge[i] for i in both) / len(both)
    conf = Counter((hand[i], judge[i]) for i in both if hand[i] != judge[i]).most_common(5)
    return (f"{len(both)} hand-labelled attempts also labelled by the judge: raw agreement {agree:.0%}, Cohen's kappa "
            f"{k:.2f}\nmost common disagreements (hand -> judge): {conf}")


def report() -> str:
    """Counts per category. Precedence: sheet labels > reviewed corrections (judge_corrections.csv: id,category,note) >
    judge. The corrections file records judge labels that were checked against the evidence and found wrong."""
    hand = _read_labels(OUT_DIR / "to_label.csv", "label") if (OUT_DIR / "to_label.csv").exists() else {}
    judge = _read_labels(OUT_DIR / "judge_labels.jsonl", "category") if (OUT_DIR / "judge_labels.jsonl").exists() else {}
    fixes = _read_labels(OUT_DIR / "judge_corrections.csv", "category") if (OUT_DIR / "judge_corrections.csv").exists() else {}
    bad = sorted(v for v in fixes.values() if v not in CATEGORIES)
    if bad:
        raise ValueError(f"unknown corrected labels: {sorted(set(bad))}")
    fixed = {k: v for k, v in fixes.items() if k not in hand}
    final = {**judge, **fixed, **hand}
    cnt = Counter(final.values())
    total = sum(cnt.values())
    lines = [f"Failure taxonomy over {total} failed attempts ({len(hand)} labelled on the sheet, {len(fixed)} judge labels "
             f"corrected after review, {total - len(hand) - len(fixed)} by the judge)"]
    for k, v in cnt.most_common():
        lines.append(f"  {k:28s}{v:5d}  {v / total:6.1%}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "judge", "kappa", "report"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="export: overwrite an existing to_label.csv")
    ap.add_argument("--model")
    a = ap.parse_args()
    if a.cmd == "export":
        export_sample(a.n, force=a.force)
    elif a.cmd == "judge":
        print("wrote", judge_all(a.model))
    elif a.cmd == "kappa":
        print(kappa_report())
    else:
        print(report())


if __name__ == "__main__":
    main()
