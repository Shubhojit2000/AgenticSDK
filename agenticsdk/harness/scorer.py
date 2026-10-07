"""Harness-side scoring. The agent never sees hidden labels or computes the reported metric.

Prediction file formats (CSV, header required):
  * classification : ``id`` + one probability column per class, named exactly as the class labels
                     in ``meta["classes"]``; each row must sum to 1 (tolerance 1e-3)
  * regression     : ``id`` + ``prediction``
``id`` is the 0-based row index of the evaluated split; every id must appear exactly once.

Metrics (all higher-is-better):
  * binary / multiclass : negative log-loss (a proper scoring rule; it does not saturate the way
                          ROC-AUC does on easy datasets, and it rewards calibrated probabilities).
                          Probabilities are clipped to [PROB_CLIP, 1 - PROB_CLIP] for everyone.
  * regression          : R^2
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, r2_score

from agenticsdk.harness.datasets import REGISTRY, TARGET, load_split

PROB_SUM_TOL = 1e-3
PROB_CLIP = 1e-6

# Floor on the (reference - B0) gap used by the normalized score, so a dataset where the strongest
# baseline barely beats the default forest cannot blow the ratio up. Datasets whose mean gap is
# below this floor are "uninformative" and are excluded from normalized aggregates.
NORM_MIN_GAP = 0.01


class InvalidPredictions(ValueError):
    """Raised with a message the Critic can show to the agent."""


@dataclass
class ScoreResult:
    valid: bool
    score: float | None
    metric: str
    error: str | None = None


def metric_value(kind: str, y_true: pd.Series, pred: np.ndarray, classes: list[str] | None) -> float:
    if kind == "regression":
        return float(r2_score(y_true.astype(float), pred))
    p = np.clip(pred, PROB_CLIP, 1 - PROB_CLIP)
    p = p / p.sum(axis=1, keepdims=True)
    return -float(log_loss(y_true.astype(str).values, p, labels=classes))


def validate(pred: pd.DataFrame, n_rows: int, kind: str, classes: list[str] | None) -> np.ndarray:
    """Return the prediction matrix ordered by id, or raise InvalidPredictions."""
    want = ["id", "prediction"] if kind == "regression" else ["id", *classes]
    missing = [c for c in want if c not in pred.columns]
    extra = [c for c in pred.columns if c not in want]
    if missing or extra:
        raise InvalidPredictions(f"columns must be exactly {want}; missing={missing} extra={extra}")
    if len(pred) != n_rows:
        raise InvalidPredictions(f"expected {n_rows} rows, got {len(pred)}")
    if not pd.api.types.is_integer_dtype(pred["id"]) or sorted(pred["id"].tolist()) != list(range(n_rows)):
        raise InvalidPredictions(f"id must contain each integer 0..{n_rows - 1} exactly once")
    vals = pred.drop(columns="id")
    if not all(pd.api.types.is_numeric_dtype(vals[c]) for c in vals.columns):
        raise InvalidPredictions("prediction columns must be numeric")
    arr = vals.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise InvalidPredictions("predictions contain NaN or infinite values")
    if kind != "regression":
        if (arr < 0).any() or (arr > 1).any():
            raise InvalidPredictions("probabilities must lie in [0, 1]")
        if np.abs(arr.sum(axis=1) - 1.0).max() > PROB_SUM_TOL:
            raise InvalidPredictions("each row of class probabilities must sum to 1")
    order = np.argsort(pred["id"].to_numpy())
    return arr[order]


def score_frame(pred: pd.DataFrame, key: str, seed: int, split: str) -> ScoreResult:
    """Score a predictions DataFrame against the hidden labels of ``split`` ('val' or 'test')."""
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    spec = REGISTRY[key]
    data = load_split(key, seed)
    truth = data[split][TARGET]
    classes = data["meta"]["classes"]
    try:
        arr = validate(pred, len(truth), spec.kind, classes)
        if spec.kind == "regression":
            arr = arr[:, 0]
        return ScoreResult(True, metric_value(spec.kind, truth, arr, classes), spec.metric)
    except InvalidPredictions as e:
        return ScoreResult(False, None, spec.metric, str(e))


def score_file(path: str | Path, key: str, seed: int, split: str) -> ScoreResult:
    try:
        pred = pd.read_csv(path)
    except Exception as e:  # unreadable / empty file
        return ScoreResult(False, None, REGISTRY[key].metric, f"cannot read predictions: {e}")
    # class labels such as "1" are read back as ints by read_csv; compare names as strings
    pred.columns = [str(c) for c in pred.columns]
    return score_frame(pred, key, seed, split)


def normalized_score(score: float, b0: float, ref: float) -> float:
    """0 = default RandomForest (B0), 1 = strongest baseline ``ref = max(B1, B2)``."""
    return (score - b0) / max(ref - b0, NORM_MIN_GAP)
