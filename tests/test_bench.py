"""Phase 7 plumbing: statistics, benchmark bookkeeping, failure-taxonomy agreement, the single-agent ablation."""
import json

import pytest

from agenticsdk import bench, stats, taxonomy
from agenticsdk.agents import react
from agenticsdk.llm import LLMClient, RawResult, ToolStep
from tests.test_graph_mocked import GOOD, fence


def write_run(root, variant, ds, seed, norm, test=0.5, valid=True, cost=0.01, attempts=2, beats=(True, True, False)):
    d = root / f"bench_{variant}_{ds}_s{seed}"
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({
        "status": "passed" if valid else "failed", "dataset_key": ds, "final_normalized": norm if valid else None,
        "final_test_score": test if valid else None,
        "baselines_test": {"B0": 0.4, "B1": 0.45, "B2": 0.55} if beats else {},
        "strategy_history": [{}], "attempts": [{"index": i + 1} for i in range(attempts)],
        "llm_usage": {"calls": 10, "input_tokens": 1000, "output_tokens": 500, "cost_usd_live": cost,
                      "cost_usd_replayed": 0.0, "retries": 1}}))
    (d / "llm_calls.jsonl").write_text(json.dumps({"latency_s": 2.0}) + "\n")
    return d


@pytest.fixture
def runs(tmp_path):
    # 6 datasets x 2 seeds; variant X beats Y everywhere by 0.2; Y has one failed run
    for i in range(6):
        for s in range(2):
            write_run(tmp_path, "A_full", f"d{i}", s, 1.0 + 0.01 * i, test=0.6)
            write_run(tmp_path, "E_no_memory", f"d{i}", s, 0.8 + 0.01 * i, test=0.5, valid=not (i == 0 and s == 0))
    return tmp_path


def test_load_runs_parses_variants_datasets_seeds_and_costs(runs):
    df = stats.load_runs(runs_dir=runs)
    assert len(df) == 24 and set(df.variant) == {"A_full", "E_no_memory"}
    r = df[(df.variant == "A_full") & (df.dataset == "d3") & (df.seed == 1)].iloc[0]
    assert r.attempts == 2 and r.tokens == 1500 and r.cost == pytest.approx(0.01) and r.seconds == 2.0
    assert bool(r.beats_B0) and bool(r.beats_B1) and bool(r.beats_B2)   # 0.6 > 0.55


def test_summary_counts_failures_as_worst_case_and_gives_a_cluster_bootstrap_ci(runs):
    s = stats.summary(stats.load_runs(runs_dir=runs)).set_index("variant")
    assert s.loc["E_no_memory", "failed"] == 1 and s.loc["A_full", "failed"] == 0
    a = s.loc["A_full"]
    assert a.ci_lo <= a.norm_mean <= a.ci_hi and a.win_B2 == 1.0
    assert s.loc["E_no_memory", "norm_mean"] < a.norm_mean
    assert a.cost_per_norm_point == pytest.approx(0.01 / a.norm_mean, rel=0.05)


def test_paired_comparison_reports_wins_losses_and_a_wilcoxon_over_datasets(runs):
    c = stats.compare(stats.load_runs(runs_dir=runs), "E_no_memory", "A_full")
    assert c["pairs"] == 12 and c["datasets"] == 6 and c["wins"] == 12 and c["losses"] == 0
    assert c["mean_diff"] > 0.19 and c["datasets_better"] == 6 and c["wilcoxon_p"] == pytest.approx(0.03125)
    assert "Wilcoxon" in stats.render(stats.load_runs(runs_dir=runs))
    assert stats.render(stats.load_runs("none_", runs_dir=runs)) == "(no benchmark runs yet)"


def test_stats_groups_keep_the_unseen_datasets_out_of_the_dev_tables(runs):
    write_run(runs, "A_full", "adult", 0, 0.9, test=0.6)       # one of the 3 never-touched datasets
    assert set(stats.load_runs(runs_dir=runs).dataset) == {f"d{i}" for i in range(6)}
    assert set(stats.load_runs(runs_dir=runs, group="unseen").dataset) == {"adult"}
    assert "adult" in set(stats.load_runs(runs_dir=runs, group=None).dataset)


def test_sign_test_against_baselines_counts_datasets_and_gives_an_exact_p(runs):
    df = stats.load_runs(runs_dir=runs)
    r = stats.vs_baseline(df, "A_full", "B2")          # 0.6 > 0.55 on every one of the 6 datasets
    assert r["win_datasets"] == 6 and r["lose_datasets"] == 0 and r["sign_p"] == pytest.approx(2 * 0.5 ** 6)
    r = stats.vs_baseline(df, "E_no_memory", "B2")     # 0.5 < 0.55 everywhere (valid runs)
    assert r["win_datasets"] == 0 and r["lose_datasets"] == 6
    assert "A_full" in stats.render_vs_baselines(df)


def test_bench_bookkeeping_skips_finished_runs_and_plans_the_rest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "RUNS_DIR", tmp_path)
    write_run(tmp_path, "E_no_memory", "credit_g", 0, 1.0)
    assert bench.done("E_no_memory", "credit_g", 0) and not bench.done("E_no_memory", "credit_g", 1)
    bench.cmd_plan(["E_no_memory"], ["credit_g"], [0, 1, 2])
    out = capsys.readouterr().out
    assert "3 runs in the grid" in out and "left   2/3" in out
    assert bench.run_id("A_full", "adult", 2) == "bench_A_full_adult_s2"
    assert set(bench.VARIANTS) == set(stats.VARIANT_ORDER)


def test_bench_run_stops_before_the_daily_quota(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(bench, "live_calls_today", lambda since=None, model=None: 440)
    called = []
    monkeypatch.setattr(bench, "run_one", lambda *a, **k: called.append(a))
    bench.cmd_run(["E_no_memory"], ["credit_g"], [0], 450, None)
    assert called == [] and "stopping" in capsys.readouterr().out


def test_cohen_kappa_and_agreement_report(tmp_path, monkeypatch):
    assert taxonomy.cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert taxonomy.cohen_kappa(list("aabb"), list("abab")) == pytest.approx(0.0)
    assert taxonomy.cohen_kappa(list("aaab"), list("aaab")) == 1.0
    monkeypatch.setattr(taxonomy, "OUT_DIR", tmp_path)
    (tmp_path / "to_label.csv").write_text("id,dataset,error_or_score,label\nr#1,x,e,timeout\nr#2,x,e,bad_strategy\nr#3,x,e,\n")
    (tmp_path / "judge_labels.jsonl").write_text("\n".join(json.dumps(d) for d in [
        {"id": "r#1", "category": "timeout", "reason": ""}, {"id": "r#2", "category": "other", "reason": ""},
        {"id": "r#3", "category": "timeout", "reason": ""}]))
    text = taxonomy.kappa_report()
    assert "2 hand-labelled" in text and "agreement 50%" in text
    rep = taxonomy.report()
    assert "3 failed attempts (2 labelled on the sheet, 0 judge labels corrected" in rep and "bad_strategy" in rep   # a sheet label beats the judge
    (tmp_path / "judge_corrections.csv").write_text("id,category,note\nr#3,no_output,judge was wrong\nr#2,other,ignored: the sheet label wins\n")
    rep = taxonomy.report()
    assert "1 judge labels corrected" in rep and "no_output" in rep and "bad_strategy" in rep   # sheet > correction > judge
    (tmp_path / "judge_corrections.csv").write_text("id,category,note\nr#3,not_a_category,x\n")
    with pytest.raises(ValueError):
        taxonomy.report()


def test_failed_attempts_are_collected_from_reports(tmp_path):
    d = tmp_path / "bench_E_no_memory_credit_g_s0"
    (d / "attempt_1").mkdir(parents=True)
    (d / "attempt_1" / "solution.py").write_text("print(1)")
    (d / "report.json").write_text(json.dumps({"dataset_key": "credit_g", "kind": "binary", "metric_name": "neg_log_loss",
        "baselines_val": {"B0": -0.5}, "attempts": [
            {"index": 1, "error": "boom", "exec": {"stdout": ""}}, {"index": 2, "val_valid": True, "val_score": -0.4, "exec": {}},
            {"index": 3, "val_valid": True, "val_score": -0.9, "exec": {}}]}))
    items = taxonomy.failed_attempts(tmp_path)
    assert [i["id"] for i in items] == ["bench_E_no_memory_credit_g_s0#1", "bench_E_no_memory_credit_g_s0#3"]
    assert items[0]["code"] == "print(1)"


# ---- ablation C: the single ReAct agent ---------------------------------------------------------------------
def test_react_agent_investigates_writes_a_script_and_stops_on_done(tmp_path, monkeypatch):
    monkeypatch.setattr(react, "RUNS_DIR", tmp_path)
    calls = []

    def backend(model, system, prompt, schema, thinking_budget, temperature, tools=None):
        calls.append((bool(tools), prompt))
        if tools and len(calls) == 1:
            return RawResult(ToolStep(tool="profile_dataset", arguments={}).model_dump_json(), 1, 1)
        if tools:   # no more investigation: the model answers with the script
            return RawResult(ToolStep(text=fence(GOOD)).model_dump_json(), 1, 1)
        return RawResult("DONE", 1, 1)

    real = react.LLMClient
    monkeypatch.setattr(react, "LLMClient", lambda **kw: real(**kw, cache_dir=tmp_path / "cache", backend=backend,
                                                              sleep=lambda s: None))
    out = react.run_react("credit_g", 0, "r1")
    assert out.status == "passed" and len(out.attempts) == 1 and out.attempts[0].val_valid
    assert out.tool_log[0]["tool"] == "profile_dataset" and out.best_attempt == 1
    rep = json.loads((tmp_path / "r1" / "report.json").read_text())
    assert rep["variant"] == "react" and rep["strategy_history"] == [] and rep["final_test_score"] is not None
    assert "What your investigation found" in calls[1][1] and "harness validation score" in calls[2][1]
    df = stats.load_runs("r", runs_dir=tmp_path)
    assert df.empty   # not a bench_ run: the benchmark only reads bench_* directories
    assert LLMClient is not None
