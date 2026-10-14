---
title: AgenticSDK
emoji: 🧪
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multi-agent ML engineering with honest baselines
---

# AgenticSDK demo

Upload a CSV and pick the column to predict. A planner, a feature agent, a coding agent and a critic build a model
for it; the result is scored on held-out rows the agents never see and compared with three frozen baselines
(default random forest, default LightGBM, FLAML).

This is a free demo with caps: files up to 5 MB and 20,000 rows, 4 scripts per run, one run at a time, three
runs per hour per visitor, and a daily allowance of model calls. Uploaded data and results are deleted when you
press delete, and old runs are cleaned up automatically. Do not upload anything confidential.

Source and write-up: https://github.com/Shubhojit2000/AgenticSDK
