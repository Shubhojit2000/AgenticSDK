"""MCP server exposing dataset tools for ONE task (train split only).

    python -m agenticsdk.tools.server --dataset credit_g --seed 0          # stdio transport

Works with any MCP client. To poke at it by hand with the MCP Inspector (needs Node):

    npx @modelcontextprotocol/inspector venv\\Scripts\\python -m agenticsdk.tools.server --dataset credit_g
"""
from __future__ import annotations

import argparse
import json

from mcp.server.mcpserver import MCPServer

from agenticsdk.tools.dataset_tools import TaskTools, ToolError


def build_server(dataset: str, seed: int = 0) -> MCPServer:
    tools = TaskTools(dataset, seed)
    server = MCPServer("agenticsdk-dataset-tools")

    def guarded(fn, *a, **kw) -> str:
        try:
            return fn(*a, **kw)
        except ToolError as e:   # a readable error the model can act on, not a stack trace
            return f"ERROR: {e}"

    @server.tool()
    def profile_dataset() -> str:
        """Profile the training data: row/column counts, per-column type, missingness and cardinality,
        target distribution, duplicate rows, constant and high-cardinality columns, and the strongest
        univariate relationships with the target. Call this first."""
        return guarded(tools.profile_dataset)

    @server.tool()
    def inspect_column(name: str) -> str:
        """Detailed statistics for one feature column: quantiles, skew, outliers and how the target
        varies with the column (by quintile for numeric columns, by value for categoricals)."""
        return guarded(tools.inspect_column, name)

    @server.tool()
    def run_quick_cv(model_family: str, params_json: str = "{}", encoding: str = "ordinal", scale: bool = False,
                     missing_indicators: bool = False, folds: int = 3) -> str:
        """Cross-validate a candidate model on the training data (at most 6000 rows, k-fold, in a sandbox)
        and return the mean and std of the task metric (higher is better).
        model_family: linear | bagged_trees | gradient_boosting | knn | mlp.
        params_json: JSON object of optional hyperparameters. linear: C, alpha; bagged_trees: n_estimators,
        max_depth, min_samples_leaf; gradient_boosting: n_estimators, learning_rate, num_leaves,
        min_child_samples; knn: n_neighbors; mlp: hidden, alpha, max_iter.
        encoding: ordinal | onehot (for categorical columns). scale: standardize numeric columns.
        missing_indicators: add missing-value indicator columns. folds: 2 to 5."""
        try:
            params = json.loads(params_json or "{}")
        except json.JSONDecodeError as e:
            return f"ERROR: params_json is not valid JSON: {e}"
        if not isinstance(params, dict):
            return "ERROR: params_json must be a JSON object"
        return guarded(tools.run_quick_cv, model_family, params, encoding, scale, missing_indicators, folds)

    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build_server(a.dataset, a.seed).run()   # stdio; stdout carries the protocol, so nothing else may print


if __name__ == "__main__":
    main()
