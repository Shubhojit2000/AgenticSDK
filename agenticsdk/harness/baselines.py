"""Frozen baselines the agent system is compared against.

  B0 : default RandomForest (+ minimal imputation / ordinal encoding)  -> the floor
  B1 : default LightGBM (native categorical handling)                   -> a strong, cheap model
  B2 : FLAML AutoML under the SAME wall-clock budget the agent gets      -> a strong AutoML bar

All predictions go through ``scorer.score_frame`` so baselines are validated exactly like agent
output. Results are appended to ``results/baselines.json`` and are resumable (finished
(dataset, seed, baseline) triples are skipped).

Run:  python -m agenticsdk.harness.baselines            # dev datasets
      python -m agenticsdk.harness.baselines --groups unseen   # only for the final evaluation
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from agenticsdk.harness.datasets import REGISTRY, SEEDS, TARGET, cat_columns, load_split
from agenticsdk.harness.scorer import NORM_MIN_GAP, score_frame

warnings.filterwarnings("ignore")

BUDGET_SECONDS = 60  # wall-clock budget shared by B2 and (later) the agent system
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "baselines.json"


def _xy(split_df: pd.DataFrame):
    return split_df.drop(columns=TARGET), split_df[TARGET]


def _prep_rf(train_X: pd.DataFrame, *others: pd.DataFrame):
    """Median-impute numerics, ordinal-encode categoricals (unknown -> -1)."""
    cats = cat_columns(train_X)
    nums = [c for c in train_X.columns if c not in cats]
    medians = train_X[nums].median().fillna(0.0)
    mappings = {c: {v: i for i, v in enumerate(sorted(train_X[c].dropna().astype(str).unique()))} for c in cats}

    def tf(X: pd.DataFrame) -> np.ndarray:
        out = X[nums].astype(float).fillna(medians)
        for c in cats:
            out[c] = X[c].astype(object).map(lambda v, m=mappings[c]: m.get(str(v), -1) if v is not None and v == v else -1)
        return out[nums + cats].to_numpy(dtype=float)

    return [tf(train_X), *[tf(o) for o in others]]


def _prep_cat(train_X: pd.DataFrame, *others: pd.DataFrame):
    """Convert categoricals to pandas ``category`` with train-fixed categories (LightGBM / FLAML)."""
    cats = cat_columns(train_X)
    levels = {c: sorted(train_X[c].dropna().astype(str).unique()) for c in cats}

    def tf(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in cats:
            X[c] = pd.Categorical(X[c].astype(object).map(lambda v: str(v) if v is not None and v == v else np.nan), categories=levels[c])
        return X

    return [tf(train_X), *[tf(o) for o in others]]


def _to_frame(kind: str, proba_or_pred: np.ndarray, classes: list[str] | None, model_classes=None) -> pd.DataFrame:
    n = len(proba_or_pred)
    if kind == "regression":
        return pd.DataFrame({"id": np.arange(n), "prediction": np.asarray(proba_or_pred, dtype=float)})
    mc = [str(c) for c in model_classes]
    df = pd.DataFrame(proba_or_pred, columns=mc)[classes]  # align to the harness class order
    df.insert(0, "id", np.arange(n))
    return df


def fit_predict(baseline: str, kind: str, classes, train, val, test, seed: int, budget: int):
    """Return (val_frame, test_frame, fit_seconds, info)."""
    Xtr, ytr = _xy(train)
    Xva, _ = _xy(val)
    Xte, _ = _xy(test)
    t0 = time.time()
    info: dict = {}
    if baseline == "B0":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        a, b, c = _prep_rf(Xtr, Xva, Xte)
        model = (RandomForestRegressor if kind == "regression" else RandomForestClassifier)(random_state=seed, n_jobs=-1)
        model.fit(a, ytr.astype(float) if kind == "regression" else ytr)
        run = (lambda X: model.predict(X)) if kind == "regression" else (lambda X: model.predict_proba(X))
        mc = None if kind == "regression" else model.classes_
        inputs = (b, c)
    elif baseline == "B1":
        import lightgbm as lgb

        a, b, c = _prep_cat(Xtr, Xva, Xte)
        model = (lgb.LGBMRegressor if kind == "regression" else lgb.LGBMClassifier)(random_state=seed, verbose=-1)
        model.fit(a, ytr.astype(float) if kind == "regression" else ytr)
        run = (lambda X: model.predict(X)) if kind == "regression" else (lambda X: model.predict_proba(X))
        mc = None if kind == "regression" else model.classes_
        inputs = (b, c)
    elif baseline == "B2":
        from flaml import AutoML

        a, b, c = _prep_cat(Xtr, Xva, Xte)
        metric = {"binary": "log_loss", "multiclass": "log_loss", "regression": "r2"}[kind]
        model = AutoML()
        model.fit(
            a, ytr.astype(float) if kind == "regression" else ytr,
            task="regression" if kind == "regression" else "classification",
            metric=metric, time_budget=budget, seed=seed, verbose=0, n_jobs=-1,
        )
        run = (lambda X: model.predict(X)) if kind == "regression" else (lambda X: model.predict_proba(X))
        mc = None if kind == "regression" else model.classes_
        inputs = (b, c)
        info = {"best_estimator": model.best_estimator, "best_config": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in (model.best_config or {}).items()}}
    else:
        raise ValueError(baseline)
    secs = time.time() - t0
    vf = _to_frame(kind, run(inputs[0]), classes, mc)
    tf_ = _to_frame(kind, run(inputs[1]), classes, mc)
    return vf, tf_, secs, info


def load_results() -> dict:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def run_all(groups=("dev",), seeds=SEEDS, baselines=("B0", "B1", "B2"), budget=BUDGET_SECONDS, only=None):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    res = load_results()
    for spec in REGISTRY.values():
        if spec.group not in groups or (only and spec.key not in only):
            continue
        for seed in seeds:
            data = load_split(spec.key, seed)
            classes = data["meta"]["classes"]
            for b in baselines:
                k = f"{spec.key}|{seed}|{b}"
                if k in res:
                    continue
                try:
                    vf, tf_, secs, info = fit_predict(b, spec.kind, classes, data["train"], data["val"], data["test"], seed, budget)
                    sv, st = score_frame(vf, spec.key, seed, "val"), score_frame(tf_, spec.key, seed, "test")
                    if not (sv.valid and st.valid):
                        raise RuntimeError(f"invalid baseline predictions: {sv.error or st.error}")
                    res[k] = {"dataset": spec.key, "group": spec.group, "seed": seed, "baseline": b,
                              "metric": spec.metric, "val": sv.score, "test": st.score,
                              "fit_seconds": round(secs, 2), "budget": budget if b == "B2" else None, **info}
                    print(f"{k:34s} val={sv.score:.4f} test={st.score:.4f} ({secs:.0f}s)", flush=True)
                except Exception as e:  # keep going, record the failure
                    res[k] = {"dataset": spec.key, "seed": seed, "baseline": b, "error": f"{type(e).__name__}: {e}"}
                    print(f"{k:34s} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
                RESULTS.write_text(json.dumps(res, indent=1))
    return res


def table(res: dict | None = None, groups=("dev",)) -> pd.DataFrame:
    """Checkpoint-0 table: test score mean +- std over seeds.

    ``ref`` = the stronger of B1/B2 per (dataset, seed); ``gap`` = mean(ref - B0). Datasets with
    gap < NORM_MIN_GAP are flagged ``informative=False`` (nothing to win over the default forest).
    """
    res = res or load_results()
    df = pd.DataFrame([v for v in res.values() if "error" not in v and v["group"] in groups])
    rows = []
    for d, g in df.groupby("dataset"):
        row = {"dataset": d, "metric": g.metric.iloc[0]}
        piv = g.pivot(index="seed", columns="baseline", values="test")
        for b in ("B0", "B1", "B2"):
            row[b] = f"{piv[b].mean():.4f} ± {piv[b].std(ddof=0):.4f}" if b in piv else "-"
        if {"B0", "B1", "B2"} <= set(piv.columns):
            ref = piv[["B1", "B2"]].max(axis=1)
            row["gap"] = round(float((ref - piv["B0"]).mean()), 4)
            row["informative"] = bool(row["gap"] >= NORM_MIN_GAP)
            row["best"] = "B1" if piv["B1"].mean() >= piv["B2"].mean() else "B2"
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["dev"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--budget", type=int, default=BUDGET_SECONDS)
    ap.add_argument("--table", action="store_true", help="print the table and exit")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    if not a.table:
        run_all(tuple(a.groups), only=a.only, budget=a.budget)
    print(table(groups=tuple(a.groups)).to_string(index=False))
