"""Scorer integrity tests: the agent must not be able to get a score from malformed or leaky output."""
import numpy as np
import pandas as pd
import pytest

from agenticsdk.harness.datasets import TARGET, load_split
from agenticsdk.harness.scorer import normalized_score, score_frame


def _perfect_binary(key="credit_g", seed=0, split="val"):
    data = load_split(key, seed)
    truth, classes = data[split][TARGET], data["meta"]["classes"]
    df = pd.DataFrame({c: (truth == c).astype(float) for c in classes})
    df.insert(0, "id", np.arange(len(truth)))
    return df


def test_perfect_predictions_score_near_zero_logloss():
    r = score_frame(_perfect_binary(), "credit_g", 0, "val")
    assert r.valid and r.score == pytest.approx(0.0, abs=1e-4)


def test_uniform_predictions_score_log2():
    df = _perfect_binary()
    classes = [c for c in df.columns if c != "id"]
    df[classes[0]] = df[classes[1]] = 0.5
    r = score_frame(df, "credit_g", 0, "val")
    assert r.valid and r.score == pytest.approx(-np.log(2), abs=1e-6)


def test_confidently_wrong_is_penalised_but_bounded():
    df = _perfect_binary()
    classes = [c for c in df.columns if c != "id"]
    df[classes[0]], df[classes[1]] = df[classes[1]], df[classes[0]]  # every prediction inverted
    r = score_frame(df, "credit_g", 0, "val")
    assert r.valid and r.score == pytest.approx(np.log(1e-6), abs=0.01)  # = -13.8: clipped, not -inf


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.iloc[:-1], "rows"),
    (lambda d: d.rename(columns={"id": "row"}), "columns"),
    (lambda d: d.assign(extra=0.0), "columns"),
    (lambda d: d.assign(id=0), "id"),
    (lambda d: d.assign(**{d.columns[1]: np.nan}), "NaN"),
    (lambda d: d.assign(**{d.columns[1]: d[d.columns[1]] * 2}), "[0, 1]"),
    (lambda d: d.assign(**{d.columns[1]: d[d.columns[1]] * 0.5}), "sum to 1"),
])
def test_malformed_predictions_are_rejected(mutate, msg):
    r = score_frame(mutate(_perfect_binary()), "credit_g", 0, "val")
    assert not r.valid and r.score is None and msg in r.error


def test_row_order_does_not_matter():
    df = _perfect_binary()
    shuffled = df.sample(frac=1.0, random_state=1)
    assert score_frame(shuffled, "credit_g", 0, "val").score == pytest.approx(score_frame(df, "credit_g", 0, "val").score)


def test_regression_r2():
    data = load_split("abalone", 0)
    truth = data["val"][TARGET]
    df = pd.DataFrame({"id": np.arange(len(truth)), "prediction": truth.values})
    assert score_frame(df, "abalone", 0, "val").score == pytest.approx(1.0)
    df["prediction"] = truth.mean()
    assert score_frame(df, "abalone", 0, "val").score == pytest.approx(0.0, abs=1e-9)


def test_val_and_test_targets_differ():
    # predictions perfect for val must NOT score perfectly on test
    r = score_frame(_perfect_binary(split="val"), "credit_g", 0, "test")
    assert r.valid and r.score < -1.0


def test_normalized_score_anchors():
    assert normalized_score(-0.50, -0.50, -0.30) == pytest.approx(0.0)
    assert normalized_score(-0.30, -0.50, -0.30) == pytest.approx(1.0)
    assert normalized_score(-0.20, -0.50, -0.30) == pytest.approx(1.5)
    # a tiny reference-B0 gap is floored so the ratio cannot explode
    assert abs(normalized_score(0.001, 0.0, 0.0001)) < 1
