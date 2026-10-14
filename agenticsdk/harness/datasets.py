"""Frozen dataset registry + seeded train/val/test splits.

The benchmark uses 12 development datasets and 3 "final unseen" datasets that are never touched
during development. Splits are 60/20/20 (train/val/test), stratified for classification, seeded,
and written to ``data/splits`` as parquet so every later run sees byte-identical data.

What each party may see:
  * agent    : train (features + target), val features, test features
  * harness  : val target and test target (never shown to the agent)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"

SEEDS = (0, 1, 2)
TARGET = "__target__"
VAL_FRAC = 0.20
TEST_FRAC = 0.20


@dataclass(frozen=True)
class DatasetSpec:
    key: str           # short stable name used in file paths and results
    task_id: int       # OpenML task id (frozen)
    kind: str          # "binary" | "multiclass" | "regression"
    group: str         # "dev" | "unseen"
    note: str          # why this dataset is in the benchmark

    @property
    def metric(self) -> str:
        return {"binary": "neg_log_loss", "multiclass": "neg_log_loss", "regression": "r2"}[self.kind]


# 12 dev datasets: 5 binary, 3 multiclass, 4 regression; 690 to 53,940 rows.
DEV = [
    DatasetSpec("credit_g", 31, "binary", "dev", "small, mixed categorical/numeric"),
    DatasetSpec("credit_approval", 29, "binary", "dev", "small, missing values, many categoricals"),
    DatasetSpec("sick", 3021, "binary", "dev", "heavy missingness, strong class imbalance"),
    DatasetSpec("churn", 167141, "binary", "dev", "imbalanced, mixed types"),
    DatasetSpec("bank_marketing", 14965, "binary", "dev", "45k rows, imbalanced, categoricals"),
    DatasetSpec("car", 146821, "multiclass", "dev", "all-categorical, 4 classes"),
    DatasetSpec("eucalyptus", 2079, "multiclass", "dev", "small, missing values, 5 classes"),
    DatasetSpec("letter", 6, "multiclass", "dev", "20k rows, 26 classes"),
    DatasetSpec("abalone", 361234, "regression", "dev", "classic small regression with one categorical"),
    DatasetSpec("moneyball", 361616, "regression", "dev", "missing values, categoricals"),
    DatasetSpec("california_housing", 361255, "regression", "dev", "20k rows, well-behaved target, geographic features"),
    DatasetSpec("diamonds", 361257, "regression", "dev", "54k rows, categoricals"),
]

# 3 final-unseen datasets: touched only for the final evaluation run.
UNSEEN = [
    DatasetSpec("adult", 7592, "binary", "unseen", "48k rows, missing values, categoricals"),
    DatasetSpec("segment", 146822, "multiclass", "unseen", "7 classes, numeric"),
    DatasetSpec("health_insurance", 361269, "regression", "unseen", "22k rows, categoricals"),
]

REGISTRY = {d.key: d for d in DEV + UNSEEN}

# Datasets uploaded through the service (Phase 8). Persisted on disk, so tool-server subprocesses see them too.
CUSTOM_DIR = ROOT / "data" / "custom"
MAX_CLASSES = 30
MIN_PER_CLASS = 6    # 60/20/20 stratified splits need at least 3 rows of a class in the smallest part


def load_custom_specs() -> None:
    for p in sorted(CUSTOM_DIR.glob("*.json")):
        if p.name.endswith(".baselines.json"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        REGISTRY[d["key"]] = DatasetSpec(d["key"], 0, d["kind"], "custom", d.get("note", "uploaded"))


load_custom_specs()


def cat_columns(X: pd.DataFrame) -> list[str]:
    """Non-numeric feature columns (pandas 3 stores strings as ``str``, not ``object``)."""
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]


def _sanitize_columns(cols) -> list[str]:
    """LightGBM rejects some characters in feature names; keep names simple and unique."""
    out, seen = [], set()
    for i, c in enumerate(cols):
        name = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_") or f"col{i}"
        if name in seen:
            name = f"{name}_{i}"
        seen.add(name)
        out.append(name)
    return out


class UploadError(ValueError):
    """The uploaded table cannot be turned into a task; the message is shown to the user."""


def register_custom(key: str, df: pd.DataFrame, target: str, kind: str = "auto", max_rows: int | None = None) -> DatasetSpec:
    """Validate an uploaded table, store it as a frozen raw dataset and register it. Splits are then made lazily by
    ``make_splits`` exactly as for the benchmark datasets (so the harness, not the agent, holds the val/test labels)."""
    if not re.fullmatch(r"custom_[0-9A-Za-z_]{1,60}", key):
        raise UploadError("internal error: bad dataset key")
    if target not in df.columns:
        raise UploadError(f"target column '{target}' is not in the file; columns are {list(df.columns)[:30]}")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        raise UploadError(f"the file has {len(df)} rows; this deployment accepts at most {max_rows}")
    if len(df) < 100:
        raise UploadError(f"only {len(df)} usable rows; at least 100 are needed for a train/validation/test split")
    if df.shape[1] < 2:
        raise UploadError("the file needs at least one feature column besides the target")
    y = df[target]
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    if kind == "auto":
        kind = "regression" if numeric and y.nunique() > 20 else ("binary" if y.nunique() == 2 else "multiclass")
    if kind not in ("binary", "multiclass", "regression"):
        raise UploadError(f"unknown task type '{kind}'")
    if kind == "regression" and not numeric:
        raise UploadError("regression needs a numeric target column")
    if kind != "regression":
        counts = y.astype(str).value_counts()
        if len(counts) > MAX_CLASSES:
            raise UploadError(f"{len(counts)} classes; at most {MAX_CLASSES} are supported")
        if len(counts) < 2:
            raise UploadError("the target has a single value")
        if counts.min() < MIN_PER_CLASS:
            raise UploadError(f"class '{counts.idxmin()}' has only {counts.min()} rows; every class needs at least "
                              f"{MIN_PER_CLASS} for a stratified split")
        if kind == "binary" and len(counts) != 2:
            raise UploadError(f"binary needs exactly 2 classes, found {len(counts)}")
    X = df.drop(columns=[target]).copy()
    X.columns = _sanitize_columns(X.columns)
    out = X.assign(**{TARGET: y.astype(str).values if kind != "regression" else y.astype(float).values})
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RAW_DIR / f"{key}.parquet")
    (CUSTOM_DIR / f"{key}.json").write_text(json.dumps({"key": key, "kind": kind, "rows": len(out),
                                                       "features": X.shape[1]}), encoding="utf-8")
    spec = DatasetSpec(key, 0, kind, "custom", "uploaded")
    REGISTRY[key] = spec
    return spec


def load_raw(spec: DatasetSpec) -> pd.DataFrame:
    """Download (once) and cache the full dataset with the target in column ``__target__``."""
    path = RAW_DIR / f"{spec.key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import openml

    task = openml.tasks.get_task(spec.task_id, download_splits=False)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=task.target_name)
    X = X.copy()
    X.columns = _sanitize_columns(X.columns)
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype(object).where(X[c].notna(), None)
    df = X.assign(**{TARGET: y.values if hasattr(y, "values") else y})
    df = df[df[TARGET].notna()].reset_index(drop=True)
    if spec.kind != "regression":
        df[TARGET] = df[TARGET].astype(str)
    else:
        df[TARGET] = df[TARGET].astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _split_dir(key: str, seed: int) -> Path:
    return SPLIT_DIR / key / f"seed{seed}"


def make_splits(spec: DatasetSpec, seed: int, overwrite: bool = False) -> Path:
    d = _split_dir(spec.key, seed)
    if d.exists() and not overwrite:
        return d
    df = load_raw(spec)
    strat = df[TARGET] if spec.kind != "regression" else None
    trainval, test = train_test_split(df, test_size=TEST_FRAC, random_state=seed, stratify=strat)
    strat_tv = trainval[TARGET] if spec.kind != "regression" else None
    train, val = train_test_split(
        trainval, test_size=VAL_FRAC / (1 - TEST_FRAC), random_state=seed, stratify=strat_tv
    )
    d.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.reset_index(drop=True).to_parquet(d / f"{name}.parquet")
    meta = {
        "key": spec.key, "task_id": spec.task_id, "kind": spec.kind, "metric": spec.metric,
        "seed": seed, "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "classes": sorted(df[TARGET].unique().tolist()) if spec.kind != "regression" else None,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d


def load_split(key: str, seed: int) -> dict:
    """Everything for one (dataset, seed). Harness-side: includes val/test targets."""
    spec = REGISTRY[key]
    d = make_splits(spec, seed)
    return {
        "spec": spec,
        "meta": json.loads((d / "meta.json").read_text()),
        "train": pd.read_parquet(d / "train.parquet"),
        "val": pd.read_parquet(d / "val.parquet"),
        "test": pd.read_parquet(d / "test.parquet"),
    }


def freeze_all(groups=("dev", "unseen")) -> pd.DataFrame:
    rows = []
    for spec in REGISTRY.values():
        if spec.group not in groups:
            continue
        for seed in SEEDS:
            d = make_splits(spec, seed)
            m = json.loads((d / "meta.json").read_text())
            df = pd.read_parquet(d / "train.parquet")
            rows.append({
                "key": spec.key, "group": spec.group, "kind": spec.kind, "metric": spec.metric,
                "seed": seed, "n_train": m["n_train"], "n_val": m["n_val"], "n_test": m["n_test"],
                "n_features": df.shape[1] - 1,
                "n_cat": len(cat_columns(df.drop(columns=TARGET))),
                "missing_frac": round(float(df.drop(columns=TARGET).isna().mean().mean()), 4),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download and freeze the benchmark datasets and their splits.")
    ap.add_argument("--only", nargs="+", choices=sorted(REGISTRY), help="freeze just these datasets (used by CI)")
    args = ap.parse_args()
    if args.only:
        for key in args.only:
            for seed in SEEDS:
                make_splits(REGISTRY[key], seed)
            print("frozen:", key)
    else:
        pd.set_option("display.width", 200)
        summary = freeze_all()
        print(summary[summary.seed == 0].drop(columns="seed").to_string(index=False))
