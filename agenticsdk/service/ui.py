"""Gradio front end for the AgenticSDK service (Phase 8). It only talks to the HTTP API, never to the agents directly.

    uvicorn agenticsdk.service.app:app --port 7860      # mounts this UI at /ui
    AGENTICSDK_API=http://host:7860 python -m agenticsdk.service.ui   # or run the UI on its own against a remote API
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import httpx
import pandas as pd

DEFAULT_API = f"http://127.0.0.1:{os.environ.get('PORT', '7860')}"
EXAMPLE_CSV = Path(__file__).resolve().parents[2] / "examples" / "credit_g.csv"
STATUS_TEXT = {
    "queued": "Queued.", "preparing": "Fitting the three baselines on your data (random forest, LightGBM, FLAML)...",
    "running": "The agents are working.", "awaiting_approval": "**Waiting for you:** approve or reject the proposed strategy.",
    "done": "Finished.", "failed": "The run ended without a usable model.",
}
EMPTY_STRATS = pd.DataFrame(columns=["#", "model family", "features", "human", "summary"])
EMPTY_ATTEMPTS = pd.DataFrame(columns=["#", "strategy", "validation score", "note"])


def api() -> str:
    return os.environ.get("AGENTICSDK_API", DEFAULT_API).rstrip("/")


def _get(path: str) -> dict:
    r = httpx.get(f"{api()}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path: str, **kw) -> dict:
    r = httpx.post(f"{api()}{path}", timeout=120, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except json.JSONDecodeError:
            detail = r.text
        raise gr.Error(str(detail))
    return r.json()


def baseline_chart(run: dict):
    """Baselines vs the agent on the held-out test rows (higher is better in every unit used here)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = run.get("baselines") or {}
    res = run.get("result") or {}
    names = {"B0": "default\nrandom forest", "B1": "default\nLightGBM", "B2": "FLAML\nAutoML"}
    labels = [names[k] for k in ("B0", "B1", "B2") if k in base]
    vals = [base[k]["test"] for k in ("B0", "B1", "B2") if k in base]
    colors = ["#9aa5b1"] * len(vals)
    if res.get("test") is not None:
        labels.append("agents")
        vals.append(res["test"])
        colors.append("#2563eb")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    if vals:
        ax.bar(labels, vals, color=colors)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.25 or abs(hi) * 0.05 or 0.05
        ax.set_ylim(lo - pad, hi + pad)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("test score (higher is better)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def on_file(f):
    if f is None:
        return gr.update(choices=[], value=None)
    cols = list(pd.read_csv(f.name if hasattr(f, "name") else f, nrows=5).columns)
    return gr.update(choices=cols, value=cols[-1])


NO_KEY = ("**No Gemini API key found.** Create a free key at https://aistudio.google.com/apikey, put "
          "`GOOGLE_API_KEY=...` in the `.env` file in the project folder (copy `.env.example`), and restart the service. "
          "Until then the agents cannot run.")


def key_banner() -> str:
    try:
        return "" if _get("/health").get("key_configured", True) else NO_KEY
    except Exception:  # noqa: BLE001 - a banner must never break the page
        return ""


def start(f, target, task, hitl):
    if f is None or not target:
        raise gr.Error("Upload a CSV and choose the column to predict.")
    if key_banner():
        raise gr.Error("No Gemini API key found: add GOOGLE_API_KEY to .env (see the banner at the top) and restart.")
    path = f.name if hasattr(f, "name") else f
    with open(path, "rb") as fh:
        out = _post("/runs", files={"file": (os.path.basename(path), fh, "text/csv")},
                    data={"target": target, "task": task, "hitl": str(bool(hitl)).lower()})
    return out["run_id"], f"Run `{out['run_id']}` started ({out['task']} task)."


def refresh(run_id):
    """One poll: status text, strategy table, attempt table, chart, approval panel and download files."""
    if not run_id:
        return ("Upload a file and press Start.", EMPTY_STRATS, EMPTY_ATTEMPTS, baseline_chart({}), gr.update(visible=False),
                "", None, None)
    run = _get(f"/runs/{run_id}")
    status = run["status"]
    text = f"**{status}.** {STATUS_TEXT.get(status, '')}"
    if run.get("error"):
        text += f"\n\n{run['error']}"
    if run.get("result") and run["result"].get("test") is not None:
        r = run["result"]
        text += (f"\n\nHeld-out test score **{r['test']:.4f}**; validation {r['val']:.4f}; normalized {r['normalized']:.2f}"
                 " (0 = default random forest, 1 = the stronger of LightGBM and FLAML)." if r.get("normalized") is not None
                 else f"\n\nHeld-out test score **{r['test']:.4f}**.")
    strategies = pd.DataFrame([{"#": s["index"], "model family": s["model_family"], "features": s["feature_approach"],
                                "human": s.get("human") or "", "summary": s["summary"]}
                               for s in run["strategies"]]) if run["strategies"] else EMPTY_STRATS
    attempts = pd.DataFrame([{"#": a["index"], "strategy": a.get("strategy_index"),
                              "validation score": None if a.get("val_score") is None else round(a["val_score"], 5),
                              "note": (a.get("error") or "ok")[:140].replace("\n", " ")} for a in run["attempts"]]) \
        if run["attempts"] else EMPTY_ATTEMPTS
    prop = run.get("proposal")
    panel = gr.update(visible=bool(prop))
    prop_text = json.dumps(prop["strategy"], indent=2) if prop else ""
    script = predictions = None
    if status == "done" and run.get("result", {}).get("best_attempt"):
        script = _download(run_id, "script", "solution.py")
        predictions = _download(run_id, "predictions", "predictions_test.csv")
    return text, strategies, attempts, baseline_chart(run), panel, prop_text, script, predictions


def _download(run_id: str, kind: str, name: str):
    import tempfile
    r = httpx.get(f"{api()}/runs/{run_id}/{kind}", timeout=60)
    if r.status_code != 200:
        return None
    path = os.path.join(tempfile.mkdtemp(), f"{run_id}_{name}")
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def decide(run_id, feedback, approve: bool):
    _post(f"/runs/{run_id}/{'approve' if approve else 'reject'}", json={"feedback": feedback or ""})
    return ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AgenticSDK") as ui:
        gr.Markdown("# AgenticSDK\nUpload a table and the column to predict. A planner, a feature agent, a coding agent "
                    "and a critic build a model; the result is compared with three frozen baselines on held-out rows "
                    "that the agents never see.")
        banner = gr.Markdown()
        run_id = gr.State("")
        with gr.Row():
            with gr.Column(scale=1):
                f = gr.File(label="CSV file", file_types=[".csv"])
                target = gr.Dropdown(label="Column to predict", choices=[])
                if EXAMPLE_CSV.exists():   # clicking the row loads the file and picks its last column (`class`) as the target
                    gr.Examples(examples=[[str(EXAMPLE_CSV)]], inputs=[f], outputs=[target], fn=on_file, run_on_click=True,
                                cache_examples=False, label="No data at hand? Click to load an example "
                                "(German credit risk, 1,000 rows; the column to predict is `class`)")
                task = gr.Radio(["auto", "binary", "multiclass", "regression"], value="auto", label="Task type")
                hitl = gr.Checkbox(label="Ask me to approve each strategy before it runs")
                go = gr.Button("Start", variant="primary")
                msg = gr.Markdown()
            with gr.Column(scale=2):
                status = gr.Markdown("Upload a file and press Start.")
                chart = gr.Plot(label="Baselines vs agents (held-out test rows)")
                with gr.Group(visible=False) as approval:
                    gr.Markdown("### Proposed strategy")
                    proposal = gr.Code(language="json", interactive=False)
                    feedback = gr.Textbox(label="Feedback for the planner (used if you reject)")
                    with gr.Row():
                        ok = gr.Button("Approve", variant="primary")
                        no = gr.Button("Reject")
                strategies = gr.Dataframe(label="Strategies", wrap=True, interactive=False)
                attempts = gr.Dataframe(label="Attempts (harness validation score)", wrap=True, interactive=False)
                with gr.Row():
                    script_f = gr.File(label="Final script")
                    pred_f = gr.File(label="Predictions on the held-out test rows")
        outputs = [status, strategies, attempts, chart, approval, proposal, script_f, pred_f]
        ui.load(key_banner, None, banner)
        f.change(on_file, f, target)
        go.click(start, [f, target, task, hitl], [run_id, msg]).then(refresh, run_id, outputs)
        ok.click(decide, [run_id, feedback, gr.State(True)], feedback).then(refresh, run_id, outputs)
        no.click(decide, [run_id, feedback, gr.State(False)], feedback).then(refresh, run_id, outputs)
        gr.Timer(2.0).tick(refresh, run_id, outputs)
    return ui


if __name__ == "__main__":
    build_ui().launch()
