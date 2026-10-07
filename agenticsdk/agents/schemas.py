"""Pydantic schemas: every agent output and the graph state."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---- LLM-facing outputs (keep these flat and simple: Gemini structured output is picky) --------
ModelFamily = Literal["linear", "bagged_trees", "gradient_boosting", "knn_svm", "mlp", "stacked_ensemble"]
FeatureApproach = Literal["minimal", "encoding_focused", "interactions_polynomial", "transforms_scaling",
                          "missingness_frequency", "aggregations_binning"]


class Strategy(BaseModel):
    """Planner output: what to build, before any code exists.

    `model_family` and `feature_approach` are the structured fields that escalation is judged on:
    a new strategy must differ from every earlier one in the (model_family, feature_approach) pair."""
    model_family: ModelFamily = Field(description="The primary model family.")
    feature_approach: FeatureApproach = Field(description="The main feature-engineering approach.")
    summary: str = Field(description="One or two sentences: the overall modelling approach.")
    preprocessing: list[str] = Field(description="Concrete preprocessing / feature steps, in order.")
    model: str = Field(description="Which model family or ensemble to train and why.")
    validation: str = Field(description="How the script will validate itself on the training data.")
    risks: list[str] = Field(description="Main ways this could fail (leakage, overfitting, time limit, ...).")

    @property
    def key(self) -> tuple[str, str]:
        return (self.model_family, self.feature_approach)


class FeaturePlan(BaseModel):
    """Feature agent output: concrete feature steps for the chosen strategy, using real column names."""
    steps: list[str] = Field(description="Ordered, concrete feature steps that name actual columns from the profile.")
    drop_columns: list[str] = Field(description="Column names to drop (ids, constants, leakage suspects); may be empty.")
    rationale: str = Field(description="One or two sentences on why these steps suit this data and model family.")


class Verdict(BaseModel):
    """Critic output."""
    decision: Literal["PASS", "RETRY_CODE", "RETRY_STRATEGY"] = Field(
        description="PASS to accept the result; RETRY_CODE to repair or tune the same strategy; "
                    "RETRY_STRATEGY to abandon the current model family / feature approach and plan a different one.")
    reasoning: str = Field(description="Two or three sentences grounded in the numbers shown.")
    feedback: str = Field(description="If RETRY_CODE: specific, actionable changes for the next script. "
                                      "If RETRY_STRATEGY: what is wrong with the approach and what to try instead. "
                                      "If PASS: empty string.")


# ---- internal records ---------------------------------------------------------------------------
class ExecSummary(BaseModel):
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    timed_out: bool = False
    blocked: bool = False


class Attempt(BaseModel):
    index: int
    strategy_index: int = 1            # which strategy (1-based) this attempt implemented
    code: str
    exec: ExecSummary
    val_valid: bool = False
    val_score: float | None = None
    error: str | None = None           # why this attempt produced no valid, scored predictions
    workdir: str = ""
    verdict_decision: str | None = None
    verdict_source: Literal["llm", "harness", "override"] | None = None


class Event(BaseModel):
    node: str
    detail: str
    data: dict = Field(default_factory=dict)


class AgentState(BaseModel):
    # task
    run_id: str
    dataset_key: str
    seed: int
    kind: Literal["binary", "multiclass", "regression"]
    metric_name: str
    classes: list[str] | None = None
    task_description: str
    profile: str = ""
    budget_seconds: int = 60                 # wall-clock limit for each script execution
    max_code_retries: int = 3                # code retries allowed per strategy
    max_strategy_retries: int = 2            # RETRY_STRATEGY escalations allowed per run
    max_total_attempts: int = 6              # hard cap on scripts executed, across all strategies
    first_family: str | None = None          # experiment knob: force the first strategy's model family
    use_tools: bool = False                  # Phase 3: planner/feature agents investigate through MCP tools
    planner_tool_budget: int = 3             # max tool calls per planning round
    feature_tool_budget: int = 2             # max tool calls per feature round
    # reference scores (harness-computed; the Critic may see val scores, never test)
    baselines_val: dict[str, float] = Field(default_factory=dict)
    baselines_test: dict[str, float] = Field(default_factory=dict)   # used by the report only
    # progress
    strategy_history: list[Strategy] = Field(default_factory=list)
    current_strategy: Strategy | None = None
    feature_plan: FeaturePlan | None = None
    current_code: str = ""
    attempts: list[Attempt] = Field(default_factory=list)
    code_retry_count: int = 0                # reset whenever a new strategy starts
    strategy_retry_count: int = 0
    verdict: Verdict | None = None
    status: Literal["running", "passed", "exhausted", "failed"] = "running"
    # outcome (written by report_node)
    best_attempt: int | None = None
    final_val_score: float | None = None
    final_test_score: float | None = None
    final_normalized: float | None = None
    tool_log: list[dict] = Field(default_factory=list)   # {agent, tool, arguments, ok}
    trace: list[Event] = Field(default_factory=list)
