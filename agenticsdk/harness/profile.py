"""Deterministic, compact dataset profile that goes into agent prompts (no timings, no paths)."""
from __future__ import annotations

import pandas as pd

from agenticsdk.harness.datasets import TARGET, cat_columns

MAX_COLUMNS_DETAILED = 40


def profile_text(train: pd.DataFrame, kind: str, classes: list[str] | None) -> str:
    X, y = train.drop(columns=TARGET), train[TARGET]
    cats = set(cat_columns(X))
    lines = [f"rows (train): {len(train)}", f"feature columns: {X.shape[1]} "
             f"({len(cats)} categorical, {X.shape[1] - len(cats)} numeric)"]
    if kind == "regression":
        d = y.astype(float).describe()
        lines.append(f"target: numeric; mean={d['mean']:.4g} std={d['std']:.4g} min={d['min']:.4g} "
                     f"median={d['50%']:.4g} max={d['max']:.4g}")
    else:
        vc = y.value_counts(normalize=True)
        lines.append(f"target: {len(classes)} classes; class share: "
                     + ", ".join(f"{c}={vc.get(c, 0):.3f}" for c in classes[:30]))
    miss = X.isna().mean()
    lines.append(f"columns with missing values: {int((miss > 0).sum())}"
                 + (f" (worst: {miss.idxmax()} {miss.max():.1%})" if (miss > 0).any() else ""))
    lines.append("columns:")
    for c in list(X.columns)[:MAX_COLUMNS_DETAILED]:
        col = X[c]
        if c in cats:
            top = col.value_counts().head(5)
            ex = ", ".join(f"{k}({v})" for k, v in top.items())
            lines.append(f"  - {c}: categorical, {col.nunique()} unique, {col.isna().mean():.1%} missing; top: {ex}")
        else:
            lines.append(f"  - {c}: numeric, {col.nunique()} unique, {col.isna().mean():.1%} missing; "
                         f"min={col.min():.4g} median={col.median():.4g} max={col.max():.4g}")
    if X.shape[1] > MAX_COLUMNS_DETAILED:
        lines.append(f"  ... and {X.shape[1] - MAX_COLUMNS_DETAILED} more columns")
    return "\n".join(lines)
