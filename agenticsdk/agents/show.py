"""Print a finished (or running) run as a readable timeline.

    python -m agenticsdk.agents.show runs/cp2
    python -m agenticsdk.agents.show cp2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def render(run_dir: Path) -> str:
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    out = [f"run: {run_dir.name}"]
    for e in events:
        node, d, data = e["node"], e["detail"], e["data"]
        if node == "planner" and "strategy" in data:
            st = data["strategy"]
            out.append(f"\n== STRATEGY {data['strategy_index']}: {st['model_family']} + {st['feature_approach']}")
            out.append(f"   {st['summary']}")
        elif node == "planner":
            out.append(f"   [planner] {d}")
        elif node == "tool":
            args = json.dumps(data["arguments"], sort_keys=True)
            out.append(f"   [tool:{data['agent']}] {data['tool']}({args})" + ("" if data["ok"] else "  ** ERROR **"))
            out.append("      -> " + data["result"][:260].replace("\n", " | "))
        elif node == "feature":
            out.append(f"   [feature] {d}")
            out.extend(f"      - {s}" for s in data["plan"]["steps"])
        elif node == "training":
            out.append(f"   [training] {d}")
        elif node == "execute":
            out.append(f"   [execute] {d}" + (f"  ERROR: {(data['error'] or '')[:100]!r}" if data.get("error") else ""))
        elif node == "critic":
            out.append(f"   [critic] {d[:220]}")
        elif node == "report":
            out.append(f"\n== REPORT: {d}")
            if "final_normalized" in data:
                out.append(f"   normalized score (test, vs B0/max(B1,B2)): {data['final_normalized']:.3f}")
    rep = run_dir / "report.json"
    if rep.exists():
        r = json.loads(rep.read_text(encoding="utf-8"))
        u = r.get("llm_usage", {})
        out.append(f"   baselines (test): {r.get('baselines_test')}")
        out.append(f"   LLM: {u.get('api_calls')} API calls, {u.get('cache_hits')} cache hits, "
                   f"{u.get('input_tokens')} in / {u.get('output_tokens')} out tokens")
    return "\n".join(out)


if __name__ == "__main__":
    arg = Path(sys.argv[1])
    print(render(arg if arg.exists() else ROOT / "runs" / arg))
