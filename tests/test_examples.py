"""What a stranger sees first: the example CSV, the recorded run, the key check and the UI's example button."""
import ast
from pathlib import Path

import pandas as pd

from agenticsdk.agents.show import render
from agenticsdk.service import app as app_mod
from agenticsdk.service import ui
from tests.test_service import env  # noqa: F401  (fixture: TestClient with the spawn mocked and temp dirs)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_example_csv_is_uploadable_through_the_api(env):  # noqa: F811
    client, spawned, _ = env
    r = client.post("/runs", files={"file": ("credit_g.csv", (EXAMPLES / "credit_g.csv").read_bytes(), "text/csv")},
                    data={"target": "class", "task": "auto", "hitl": "false"})
    assert r.status_code == 202 and r.json()["task"] == "binary" and len(spawned) == 1


def test_the_ui_example_button_picks_the_class_column():
    df = pd.read_csv(EXAMPLES / "credit_g.csv")
    assert df.shape == (1000, 21) and set(df["class"]) == {"good", "bad"}
    upd = ui.on_file(str(EXAMPLES / "credit_g.csv"))
    assert upd["value"] == "class" and "class" in [c[0] if isinstance(c, (list, tuple)) else c for c in upd["choices"]]


def test_recorded_run_renders_and_its_scripts_are_valid_python():
    run = EXAMPLES / "recorded_run_credit_g"
    text = render(run)
    assert "STRATEGY 1" in text and "REPORT: status=passed" in text and "baselines (test)" in text
    scripts = sorted(run.glob("attempt_*/solution.py"))
    assert len(scripts) == 3
    for s in scripts:
        ast.parse(s.read_text(encoding="utf-8"))


def test_key_check_reads_the_environment_and_the_banner_explains_what_to_do(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)   # do not read the developer's real .env
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert app_mod.key_configured() is False
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    assert app_mod.key_configured() is True
    monkeypatch.setattr(ui, "_get", lambda path: {"key_configured": False})
    assert "GOOGLE_API_KEY" in ui.key_banner()
    monkeypatch.setattr(ui, "_get", lambda path: {"key_configured": True})
    assert ui.key_banner() == ""
