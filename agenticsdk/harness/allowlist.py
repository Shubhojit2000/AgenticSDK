"""Static (AST) screening of agent-written scripts before they are executed.

This is defense in depth against *accidental* misbehaviour and obvious metric gaming (reading
files it should not, shelling out, using the network). It is NOT a security boundary against a
determined adversary: that needs OS-level isolation (Docker ``--network none``, Phase 8).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

ALLOWED_IMPORT_ROOTS = {
    # numerics / data / ML
    "numpy", "pandas", "scipy", "sklearn", "lightgbm",
    # stdlib helpers that cannot touch the outside world
    "math", "json", "re", "time", "warnings", "collections", "itertools", "functools",
    "typing", "random", "statistics", "dataclasses", "copy", "string", "numbers", "operator",
    "pathlib", "csv", "heapq", "bisect", "decimal", "fractions", "enum", "abc", "__future__",
}

# Names that must never be called or referenced.
BANNED_NAMES = {
    "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "breakpoint",
    "input", "getattr", "setattr", "delattr", "memoryview", "exit", "quit",
}

# String constants that look like escaping the working directory.
_ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\\\\|~)")
_FILES_ALLOWED_READ = {"train.csv", "val_features.csv", "test_features.csv"}
_FILES_ALLOWED_WRITE = {"predictions_val.csv", "predictions_test.csv"}
# Anything else that looks like a data file is rejected (e.g. guessing at hidden label files).
_FILE_LIKE = re.compile(r"\.(?:csv|parquet|feather|json|pkl|pickle|npy|npz|txt|db|sqlite|h5|joblib)$", re.I)


@dataclass
class AllowlistResult:
    ok: bool
    violations: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return "OK" if self.ok else "; ".join(self.violations)


def check_script(code: str) -> AllowlistResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return AllowlistResult(False, [f"SyntaxError: {e.msg} (line {e.lineno})"])

    v: list[str] = []
    # Pieces of f-strings (e.g. the ".csv" in f"predictions_{name}.csv") are not whole paths.
    fstring_parts = {id(c) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)
                     for c in ast.walk(n) if isinstance(c, ast.Constant)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    v.append(f"line {node.lineno}: import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORT_ROOTS:
                v.append(f"line {node.lineno}: import from '{'.' * node.level}{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            v.append(f"line {node.lineno}: use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in {"__name__", "__init__"}:
                v.append(f"line {node.lineno}: access to dunder attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in fstring_parts:
            s = node.value.strip()
            if len(s) > 200 or "\n" in s:
                continue  # prose / docstring, not a path
            if _ABS_PATH.match(s) or ".." in re.split(r"[\\/]", s):
                v.append(f"line {node.lineno}: path '{s[:60]}' leaves the working directory")
            elif (_FILE_LIKE.search(s) and not re.search(r"\s", s)  # prose such as a print() message is not a path
                  and s not in _FILES_ALLOWED_READ | _FILES_ALLOWED_WRITE):
                v.append(f"line {node.lineno}: file '{s[:60]}' is not provided; allowed files are "
                         f"{sorted(_FILES_ALLOWED_READ | _FILES_ALLOWED_WRITE)}")
    # de-duplicate, keep order
    seen, out = set(), []
    for m in v:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return AllowlistResult(not out, out)
