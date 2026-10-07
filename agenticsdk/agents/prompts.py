"""Prompt templates. Nothing in a prompt may be non-deterministic (timings, temp paths, run ids):
the replay cache keys on the exact prompt text."""
from __future__ import annotations

import json
import re

from agenticsdk.agents.schemas import AgentState, Attempt

PLANNER_SYSTEM = """You are the planning agent of an autonomous machine-learning engineering system.
Given a tabular dataset profile and a task, you design a concrete modelling strategy that another
agent will turn into a Python script. You do not write code. Be specific and pragmatic: the script
has a hard wall-clock limit, uses only numpy, pandas, scipy, scikit-learn and lightgbm, and has no
internet access. Prefer strong, robust choices over exotic ones.
You must fill the structured fields model_family and feature_approach: they are how the system tells
one strategy from another. If earlier strategies are shown, your new strategy must use a
(model_family, feature_approach) pair that none of them used."""

FEATURE_SYSTEM = """You are the feature-engineering agent of an autonomous machine-learning engineering system.
Given a dataset profile and a modelling strategy, list the concrete feature steps the coding agent
should implement. Name real columns from the profile. Fit every transformation on training rows only
(or on features only, never the label) and apply it identically to validation and test rows. Keep the
list short and high-value: 3 to 7 steps. Do not write code."""

TRAINING_SYSTEM = """You are the coding agent of an autonomous machine-learning engineering system.
You write ONE complete, self-contained Python script that trains a model and writes predictions.

Hard rules (violations are rejected before the script runs):
- Allowed imports only: numpy, pandas, scipy, sklearn, lightgbm, and harmless stdlib modules
  (math, json, re, time, warnings, collections, itertools, functools, typing, random, statistics,
  copy, pathlib). Do NOT import os, sys, subprocess, socket or anything else.
- Read only these files from the current directory: train.csv, val_features.csv, test_features.csv.
  Write only predictions_val.csv and predictions_test.csv. No other file paths, no absolute paths.
- Do not use eval, exec, compile, getattr, exit, quit or dunder attributes (to stop on an error,
  raise an exception instead of calling exit()).
- The script must finish within the wall-clock limit given in the task, including model fitting.
- Fix every random seed (random_state=0 or the seed you are told). Do not print timings or
  timestamps. Print a few short lines about what you did and your own cross-validation score.

Output format: reply with a short plan (2-3 sentences) and then ONE fenced ```python code block
containing the whole script. No other code blocks."""

CRITIC_SYSTEM = """You are the critic agent of an autonomous machine-learning engineering system.
You are shown the result of one attempt: a validation score computed by the evaluation harness
(not by the script), reference scores from simple frozen baselines on the same validation split, and
the history of earlier attempts and strategies. Higher scores are always better. Decide:
- PASS when the result is solid for this dataset and further tweaks are unlikely to help much;
- RETRY_CODE when there is a concrete, plausible way to improve it within the SAME strategy (repair,
  tuning, fixing a bug); give specific feedback;
- RETRY_STRATEGY when the model family or feature approach itself looks wrong for this data, for
  example the score is far below the baselines and tweaks to the same approach are unlikely to close
  the gap, or earlier code retries of this strategy did not help. Say what to try instead.
Judge only from the numbers shown; the harness score is the truth, anything the script printed is
advisory. A score below the weakest baseline is never good enough."""


EXPLORE_SYSTEM = """You are the investigation step of an autonomous machine-learning engineering system.
Before a strategy is designed you may call tools to look at the data and to cross-validate candidate
models quickly. Tool results are real measurements on the training split: trust them over assumptions.
Call at most one tool per turn, choose the call that will change your decision most, and stop
(reply in plain text, no tool call) as soon as you know enough. Your final text should be a short note
of what you learned and which modelling directions the measurements favour or rule out.
Tools: profile_dataset (overview, call first), inspect_column(name), run_quick_cv(model_family, ...)."""

MAX_TOOL_RESULT_CHARS = 2500


def task_block(s: AgentState, with_profile: bool = True) -> str:
    if s.kind == "regression":
        fmt = ("predictions_val.csv and predictions_test.csv must have columns exactly: id, prediction\n"
               "  - id: 0-based row position in the corresponding features file (0,1,2,...)\n"
               "  - prediction: the predicted numeric target")
    else:
        fmt = ("predictions_val.csv and predictions_test.csv must have columns exactly: id, then one column per class\n"
               f"  - class columns, in this order and spelled exactly: {s.classes}\n"
               "  - id: 0-based row position in the corresponding features file (0,1,2,...)\n"
               "  - each class column holds the predicted probability of that class; each row sums to 1\n"
               "  - the class labels are strings; the target column in train.csv may be read as numbers, "
               "so convert with .astype(str) before comparing to the class names above")
    return (f"TASK: {s.task_description}\n"
            f"Task type: {s.kind}. Metric (higher is better): {s.metric_name}"
            + (" = negative log-loss; probabilities are clipped to [1e-6, 1-1e-6], so calibrated "
               "probabilities matter more than hard accuracy." if s.metric_name == "neg_log_loss"
               else " = R^2 on the held-out rows.") + "\n"
            f"Files in the working directory: train.csv (features plus the label column named 'target'), "
            f"val_features.csv, test_features.csv (features only, same columns minus 'target').\n"
            f"Each script run has a hard limit of {s.budget_seconds} seconds of wall-clock time.\n"
            f"Output format:\n{fmt}\n\n"
            + (f"DATASET PROFILE (train split):\n{s.profile}" if with_profile else
               "DATASET (train split): " + "; ".join(s.profile.splitlines()[:2])
               + ". The full profile is not shown: use the investigation tools."))


def _strategy_text(s: AgentState) -> str:
    st = s.current_strategy
    return (f"Model family: {st.model_family}. Feature approach: {st.feature_approach}.\n"
            f"Summary: {st.summary}\nPreprocessing:\n" + "\n".join(f"- {p}" for p in st.preprocessing)
            + f"\nModel: {st.model}\nValidation: {st.validation}\nRisks:\n" + "\n".join(f"- {r}" for r in st.risks))


def _feature_text(s: AgentState) -> str:
    fp = s.feature_plan
    if fp is None:
        return ""
    drop = f"\nDrop columns: {', '.join(fp.drop_columns)}" if fp.drop_columns else ""
    return "\nFEATURE PLAN:\n" + "\n".join(f"- {x}" for x in fp.steps) + drop + f"\nRationale: {fp.rationale}"


def _best_by_strategy(s: AgentState) -> dict[int, float]:
    out: dict[int, float] = {}
    for a in s.attempts:
        if a.val_valid and (a.strategy_index not in out or a.val_score > out[a.strategy_index]):
            out[a.strategy_index] = a.val_score
    return out


def _strategy_one_line(i: int, st, best: float | None) -> str:
    score = f"best harness val score {best:.6f}" if best is not None else "no valid score"
    return f"  {i}. model_family={st.model_family}, feature_approach={st.feature_approach}: {st.summary} ({score})"


def findings_text(steps: list[dict], notes: str = "") -> str:
    """Render the tool calls made so far (deterministic: no timings)."""
    if not steps and not notes:
        return ""
    out = ["INVESTIGATION SO FAR:"]
    for i, st in enumerate(steps, 1):
        args = json.dumps(st["arguments"], sort_keys=True)
        res = st["result"] if len(st["result"]) <= MAX_TOOL_RESULT_CHARS else st["result"][:MAX_TOOL_RESULT_CHARS] + " ..."
        out.append(f"[{i}] {st['tool']}({args}) ->\n{res}")
    if notes.strip():
        out.append("Your notes: " + notes.strip())
    return "\n".join(out)


def explore_prompt(s: AgentState, role: str, steps: list[dict], left: int) -> str:
    base = task_block(s, with_profile=(role == "feature"))
    if role == "planner":
        goal = "You are about to design the modelling strategy for this task."
        if s.strategy_history:   # re-planning: investigate what was NOT tried, do not repeat old measurements
            goal += ("\n\n" + _history_block(s) + "\nInvestigate directions the earlier strategies did not cover; "
                     "do not repeat a measurement that is already implied by the history.")
    else:
        goal = "You are about to plan feature engineering for this strategy:\n" + _strategy_text(s)
    found = findings_text(steps)
    return (f"{base}\n\n{goal}\n" + (f"\n{found}\n" if found else "")
            + f"\nYou have {left} tool call(s) left. Call a tool, or reply in text if you know enough.")


def _history_block(s: AgentState) -> str:
    """Earlier, rejected strategies with their harness validation scores and the Critic's reasoning."""
    best = _best_by_strategy(s)
    base = ", ".join(f"{k}={v:.6f}" for k, v in sorted(s.baselines_val.items()))
    return ("EARLIER STRATEGIES (the Critic rejected the last one):\n"
            + "\n".join(_strategy_one_line(i, st, best.get(i)) for i, st in enumerate(s.strategy_history, 1))
            + f"\nFrozen baseline validation scores: {base}"
            + (f"\nCritic reasoning: {s.verdict.reasoning}\nCritic feedback: {s.verdict.feedback}" if s.verdict else "")
            + "\nThe new strategy must use a (model_family, feature_approach) pair that none of the above used.")


def planner_prompt(s: AgentState, retry_note: str = "", findings: str = "") -> str:
    out = [task_block(s, with_profile=not s.use_tools)]
    if s.strategy_history:
        out.append("\n" + _history_block(s))
    elif s.first_family:
        out.append(f"\nConstraint for this run: the first strategy's model_family must be '{s.first_family}'.")
    if findings:
        out.append("\n" + findings)
    if retry_note:
        out.append("\n" + retry_note)
    out.append("\nDesign the modelling strategy now.")
    return "\n".join(out)


def feature_prompt(s: AgentState, findings: str = "") -> str:
    return (task_block(s) + "\n\nSTRATEGY:\n" + _strategy_text(s) + ("\n\n" + findings if findings else "")
            + "\n\nList the concrete feature steps for this strategy now.")


def training_prompt(s: AgentState) -> str:
    out = [task_block(s), "\nSTRATEGY TO IMPLEMENT:\n" + _strategy_text(s) + _feature_text(s)]
    cur = len(s.strategy_history)
    if s.attempts and s.attempts[-1].strategy_index != cur:
        best = _best_by_strategy(s)
        out.append("\nA different earlier strategy was rejected"
                   + (f" (its best harness validation score was {max(best.values()):.6f})." if best else ".")
                   + " Write a fresh script for the strategy above; do not reuse the earlier approach.")
        out.append("\nWrite the complete script now.")
    elif s.attempts:
        last = s.attempts[-1]
        out.append(f"\nThis is attempt {len(s.attempts) + 1}. The previous script was:\n```python\n{last.code}\n```")
        if last.error:
            out.append(f"It produced NO valid scored predictions. Problem:\n{last.error}")
        else:
            out.append(f"It produced valid predictions with a harness validation score of {last.val_score:.6f}.")
        if last.exec.stdout.strip():
            out.append("Its stdout:\n" + last.exec.stdout.strip())
        if s.verdict and s.verdict.feedback:
            out.append("Critic feedback to act on:\n" + s.verdict.feedback)
        out.append("Write the complete corrected/improved script (not a diff).")
    else:
        out.append("\nWrite the complete script now.")
    return "\n".join(out)


def critic_prompt(s: AgentState, attempt: Attempt) -> str:
    base = ", ".join(f"{k}={v:.6f}" for k, v in sorted(s.baselines_val.items()))
    hist = "\n".join(
        f"- attempt {a.index} (strategy {a.strategy_index}): "
        + (f"val={a.val_score:.6f}" if a.val_valid else f"FAILED ({(a.error or '')[:120]})")
        for a in s.attempts[:-1]) or "- (none)"
    best = _best_by_strategy(s)
    cur = len(s.strategy_history)
    prior = "\n".join(_strategy_one_line(i, st, best.get(i))
                      for i, st in enumerate(s.strategy_history[:-1], 1)) or "  (none; this is the first strategy)"
    return (f"Dataset: {s.dataset_key}. Metric (higher is better): {s.metric_name}.\n"
            f"Reference validation scores from frozen baselines "
            f"(B0 = default random forest, B1 = default LightGBM, B2 = FLAML AutoML with the same time budget): {base}\n\n"
            f"Current strategy (number {cur}):\n{_strategy_text(s)}{_feature_text(s)}\n\n"
            f"Earlier strategies already abandoned:\n{prior}\n\n"
            f"Attempt {attempt.index} overall (at most {s.max_total_attempts}). "
            f"Harness validation score: {attempt.val_score:.6f}\n"
            f"Script stdout:\n{attempt.exec.stdout.strip() or '(empty)'}\n\n"
            f"Earlier attempts:\n{hist}\n\n"
            f"Code retries left for this strategy: {s.max_code_retries - s.code_retry_count}. "
            f"Strategy changes left: {s.max_strategy_retries - s.strategy_retry_count}.\nGive your verdict.")


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(text: str) -> str | None:
    """Return the longest fenced code block, or None if the reply has none."""
    blocks = _FENCE.findall(text)
    return max(blocks, key=len).strip() if blocks else None
