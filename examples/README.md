# Examples

- `credit_g.csv`: the German credit data (1,000 rows, 20 features, column `class` = good / bad credit risk), the UCI
  Statlog German Credit dataset as served by OpenML (`credit-g`; see the OpenML / UCI page for its license). Upload
  it in the web UI, or use the
  "example" button there, and predict `class`.
- `recorded_run_credit_g/`: one real run of the full system on that data, so you can see what the agents do without an
  API key. It is the run `bench_A_full_credit_g_s0` of the benchmark (planner with tools, one strategy, three scripts,
  critic verdicts, the final held-out score against the three baselines). View it with

  ```bash
  python -m agenticsdk.agents.show examples/recorded_run_credit_g
  ```

  The scripts the coding agent wrote are in `recorded_run_credit_g/attempt_*/solution.py`.
