"""Tool logic behind the MCP server (plain Python, testable without MCP).

Integrity: every tool reads the TRAIN split only (train.parquet + meta.json). Validation and test
rows are never opened here, so no tool can leak hidden labels. `run_quick_cv` does not run model-
written code: it renders a fixed template from a validated spec and runs it through the same sandbox
(timeout, process-tree kill, stripped environment, allow-list) as the agents' scripts.
"""
from __future__ import annotations

import difflib
import json
import tempfile
from pathlib import Path

import pandas as pd

from agenticsdk.harness.datasets import REGISTRY, TARGET, cat_columns, make_splits
from agenticsdk.harness.profile import profile_text
from agenticsdk.harness.sandbox import AGENT_TARGET, TRAIN_FILE, run_script

CV_TIMEOUT_S = 45
CV_MAX_ROWS = 6000
HIGH_CARDINALITY = 50

# model family -> {parameter: (low, high, is_int)}; anything else is rejected, not passed through.
PARAM_SPEC: dict[str, dict[str, tuple[float, float, bool]]] = {
    "linear": {"C": (1e-3, 100.0, False), "alpha": (1e-3, 1000.0, False)},
    "bagged_trees": {"n_estimators": (10, 300, True), "max_depth": (2, 30, True), "min_samples_leaf": (1, 50, True)},
    "gradient_boosting": {"n_estimators": (20, 500, True), "learning_rate": (0.01, 0.5, False),
                          "num_leaves": (4, 128, True), "min_child_samples": (2, 100, True)},
    "knn": {"n_neighbors": (1, 100, True)},
    "mlp": {"hidden": (8, 256, True), "alpha": (1e-6, 1.0, False), "max_iter": (20, 300, True)},
}
FAMILY_ALIASES = {"knn_svm": "knn"}
ENCODINGS = ("ordinal", "onehot")

_MODELS = {  # (classifier, regressor) constructor source; parameters are spliced in as validated literals
    "linear": ("LogisticRegression(C=%(C)r, max_iter=500)", "Ridge(alpha=%(alpha)r)"),
    "bagged_trees": ("RandomForestClassifier(n_estimators=%(n_estimators)r, max_depth=%(max_depth)r, "
                     "min_samples_leaf=%(min_samples_leaf)r, n_jobs=4, random_state=0)",
                     "RandomForestRegressor(n_estimators=%(n_estimators)r, max_depth=%(max_depth)r, "
                     "min_samples_leaf=%(min_samples_leaf)r, n_jobs=4, random_state=0)"),
    "gradient_boosting": ("LGBMClassifier(n_estimators=%(n_estimators)r, learning_rate=%(learning_rate)r, "
                          "num_leaves=%(num_leaves)r, min_child_samples=%(min_child_samples)r, "
                          "verbose=-1, n_jobs=4, random_state=0)",
                          "LGBMRegressor(n_estimators=%(n_estimators)r, learning_rate=%(learning_rate)r, "
                          "num_leaves=%(num_leaves)r, min_child_samples=%(min_child_samples)r, "
                          "verbose=-1, n_jobs=4, random_state=0)"),
    "knn": ("KNeighborsClassifier(n_neighbors=%(n_neighbors)r)", "KNeighborsRegressor(n_neighbors=%(n_neighbors)r)"),
    "mlp": ("MLPClassifier(hidden_layer_sizes=(%(hidden)r,), alpha=%(alpha)r, max_iter=%(max_iter)r, "
            "early_stopping=True, random_state=0)",
            "MLPRegressor(hidden_layer_sizes=(%(hidden)r,), alpha=%(alpha)r, max_iter=%(max_iter)r, "
            "early_stopping=True, random_state=0)"),
}
_DEFAULTS = {"C": 1.0, "alpha": 1.0, "n_estimators": 100, "max_depth": None, "min_samples_leaf": 1,
             "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20, "n_neighbors": 15,
             "hidden": 64, "max_iter": 100}
_MLP_ALPHA = 1e-4

_TEMPLATE = '''
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

CLF = __CLF__
df = pd.read_csv("train.csv")
y = df.pop("target")
if len(df) > __MAX_ROWS__:
    idx = df.sample(n=__MAX_ROWS__, random_state=0).index
    df, y = df.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
num_cols = list(df.select_dtypes(include="number").columns)
cat_cols = [c for c in df.columns if c not in num_cols]
for c in cat_cols:
    df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
if CLF:
    y = y.astype(str)


def make_prep():
    num = [("imp", SimpleImputer(strategy="median", add_indicator=__MISSING__))]
    if __SCALE__:
        num.append(("sc", StandardScaler()))
    if "__ENCODING__" == "onehot":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat = [("imp", SimpleImputer(strategy="most_frequent")), ("enc", enc)]
    return ColumnTransformer([("num", Pipeline(num), num_cols), ("cat", Pipeline(cat), cat_cols)])


def make_model():
    return __MODEL__


splitter = (StratifiedKFold if CLF else KFold)(n_splits=__FOLDS__, shuffle=True, random_state=0)
scores = []
for tr, va in splitter.split(df, y):
    pipe = Pipeline([("prep", make_prep()), ("model", make_model())]).fit(df.iloc[tr], y.iloc[tr])
    if CLF:
        p = np.clip(pipe.predict_proba(df.iloc[va]), 1e-6, 1 - 1e-6)
        p = p / p.sum(axis=1, keepdims=True)
        scores.append(-log_loss(y.iloc[va], p, labels=list(pipe.classes_)))
    else:
        scores.append(r2_score(y.iloc[va], pipe.predict(df.iloc[va])))
print(json.dumps({"mean": float(np.mean(scores)), "std": float(np.std(scores)),
                  "folds": [round(float(s), 4) for s in scores], "rows": int(len(df))}))
'''


class ToolError(ValueError):
    """A bad tool request; the message goes back to the model so it can correct itself."""


def _fmt(x: float) -> str:
    return f"{x:.4g}"


class TaskTools:
    """The three tools for one (dataset, seed). Reads the train split only."""

    def __init__(self, dataset: str, seed: int = 0):
        if dataset not in REGISTRY:
            raise ToolError(f"unknown dataset '{dataset}'")
        self.dataset, self.seed = dataset, seed
        d = make_splits(REGISTRY[dataset], seed)
        self.meta = json.loads((d / "meta.json").read_text())
        self.train: pd.DataFrame = pd.read_parquet(d / "train.parquet")   # never val.parquet / test.parquet
        self.kind: str = self.meta["kind"]
        self.classes: list[str] | None = self.meta["classes"]
        self.X = self.train.drop(columns=TARGET)
        self.y = self.train[TARGET]
        self.cats = set(cat_columns(self.X))

    # ---- tool 1 ----------------------------------------------------------------------------
    def profile_dataset(self) -> str:
        lines = [profile_text(self.train, self.kind, self.classes), "", "checks:"]
        lines.append(f"  duplicate feature rows: {int(self.X.duplicated().sum())}")
        const = [c for c in self.X.columns if self.X[c].nunique(dropna=False) <= 1]
        lines.append(f"  constant columns: {const or 'none'}")
        hi = [f"{c}({self.X[c].nunique()})" for c in self.cats if self.X[c].nunique() > HIGH_CARDINALITY]
        lines.append(f"  high-cardinality categoricals (>{HIGH_CARDINALITY}): {hi or 'none'}")
        rel = self._univariate_relations()
        lines.append("  strongest numeric relationships with the target (|Spearman|): "
                     + (", ".join(f"{k}={v:.2f}" for k, v in rel[:8]) if rel else "n/a for this task type"))
        return "\n".join(lines)

    def _numeric_target(self) -> pd.Series | None:
        if self.kind == "regression":
            return self.y.astype(float)
        if self.kind == "binary" and self.classes:
            return (self.y.astype(str) == self.classes[1]).astype(float)
        return None

    def _univariate_relations(self) -> list[tuple[str, float]]:
        t = self._numeric_target()
        if t is None:
            return []
        out = []
        for c in self.X.columns:
            if c in self.cats or self.X[c].nunique() < 2:
                continue
            r = self.X[c].astype(float).corr(t, method="spearman")
            if pd.notna(r):
                out.append((c, abs(float(r))))
        return sorted(out, key=lambda kv: (-kv[1], kv[0]))

    # ---- tool 2 ----------------------------------------------------------------------------
    def inspect_column(self, name: str) -> str:
        if name not in self.X.columns:
            near = difflib.get_close_matches(name, list(self.X.columns), n=5)
            raise ToolError(f"no column named '{name}'." + (f" Did you mean: {', '.join(near)}?" if near else
                            f" Columns: {', '.join(map(str, list(self.X.columns)[:40]))}"))
        col = self.X[name]
        lines = [f"column '{name}': {'categorical' if name in self.cats else 'numeric'}, "
                 f"{col.nunique()} unique, {col.isna().mean():.1%} missing"]
        t = self._numeric_target()
        if name in self.cats:
            vc = col.value_counts().head(10)
            for v, n in vc.items():
                extra = ""
                if t is not None:
                    extra = f", target {'mean' if self.kind == 'regression' else 'positive rate'} {_fmt(t[col == v].mean())}"
                lines.append(f"  {v!r}: {n} rows ({n / len(col):.1%}){extra}")
            if col.nunique() > 10:
                lines.append(f"  ... {col.nunique() - 10} more values")
            return "\n".join(lines)
        x = col.astype(float)
        q = x.quantile([0, .25, .5, .75, 1.0])
        lines.append(f"  min {_fmt(q[0])}, q1 {_fmt(q[.25])}, median {_fmt(q[.5])}, q3 {_fmt(q[.75])}, max {_fmt(q[1.0])}; "
                     f"mean {_fmt(x.mean())}, std {_fmt(x.std())}, skew {_fmt(x.skew())}")
        iqr = q[.75] - q[.25]
        out = float(((x < q[.25] - 1.5 * iqr) | (x > q[.75] + 1.5 * iqr)).mean()) if iqr > 0 else 0.0
        lines.append(f"  zeros {(x == 0).mean():.1%}, negatives {(x < 0).mean():.1%}, IQR outliers {out:.1%}")
        if t is not None and x.nunique() > 2:
            bins = pd.qcut(x, q=5, duplicates="drop")
            g = t.groupby(bins, observed=True).mean()
            lines.append(f"  target {'mean' if self.kind == 'regression' else 'positive rate'} by quintile of this column: "
                         + ", ".join(_fmt(v) for v in g.values))
        elif self.classes and self.kind == "multiclass" and len(self.classes) <= 10:
            g = x.groupby(self.y).mean()
            lines.append("  mean of this column per class: " + ", ".join(f"{k}={_fmt(v)}" for k, v in g.items()))
        return "\n".join(lines)

    # ---- tool 3 ----------------------------------------------------------------------------
    def run_quick_cv(self, model_family: str, params: dict | None = None, encoding: str = "ordinal",
                     scale: bool = False, missing_indicators: bool = False, folds: int = 3) -> str:
        fam = FAMILY_ALIASES.get(model_family, model_family)
        if fam not in PARAM_SPEC:
            raise ToolError(f"model_family '{model_family}' is not supported by quick CV. "
                            f"Choose one of: {', '.join(sorted(PARAM_SPEC))}")
        if encoding not in ENCODINGS:
            raise ToolError(f"encoding must be one of {ENCODINGS}")
        if not isinstance(folds, int) or not 2 <= folds <= 5:
            raise ToolError("folds must be an integer from 2 to 5")
        values = dict(_DEFAULTS, alpha=_MLP_ALPHA if fam == "mlp" else _DEFAULTS["alpha"])
        for k, v in (params or {}).items():
            if k not in PARAM_SPEC[fam]:
                raise ToolError(f"parameter '{k}' is not allowed for {fam}. Allowed: "
                                + ", ".join(f"{p} in [{lo:g}, {hi:g}]" for p, (lo, hi, _) in PARAM_SPEC[fam].items()))
            lo, hi, is_int = PARAM_SPEC[fam][k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ToolError(f"parameter '{k}' must be a number in [{lo:g}, {hi:g}], got {v!r}")
            values[k] = int(v) if is_int else float(v)
        clf = self.kind != "regression"
        code = (_TEMPLATE.replace("__CLF__", repr(clf)).replace("__MAX_ROWS__", str(CV_MAX_ROWS))
                .replace("__MISSING__", repr(bool(missing_indicators))).replace("__SCALE__", repr(bool(scale)))
                .replace("__ENCODING__", encoding).replace("__FOLDS__", str(folds))
                .replace("__MODEL__", _MODELS[fam][0 if clf else 1] % values))
        with tempfile.TemporaryDirectory(prefix="quickcv_") as tmp:
            wd = Path(tmp)
            self.train.rename(columns={TARGET: AGENT_TARGET}).to_csv(wd / TRAIN_FILE, index=False)  # train only
            r = run_script(code, wd, timeout=CV_TIMEOUT_S)
        spec = (f"model={fam}, params={ {k: values[k] for k in PARAM_SPEC[fam]} }, features: encoding={encoding}, "
                f"scale={scale}, missing_indicators={missing_indicators}")
        if r.timed_out:
            raise ToolError(f"quick CV timed out after {CV_TIMEOUT_S}s ({spec}). Use a smaller or simpler model.")
        if not r.ok:
            raise ToolError(f"quick CV failed ({spec}): {r.stderr.strip()[-400:] or r.stdout.strip()[-400:]}")
        res = json.loads(r.stdout.strip().splitlines()[-1])
        sub = f", subsampled from {len(self.train)}" if res["rows"] < len(self.train) else ""
        return (f"quick CV, {folds}-fold on {res['rows']} train rows{sub}: {REGISTRY[self.dataset].metric} "
                f"mean {res['mean']:.4f} std {res['std']:.4f} (folds {res['folds']}); {spec}")
