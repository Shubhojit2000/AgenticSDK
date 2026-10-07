import os
import time

import psutil
import pytest

from agenticsdk.harness.allowlist import check_script
from agenticsdk.harness.sandbox import prepare_workdir, run_script


@pytest.fixture()
def wd(tmp_path):
    prepare_workdir("credit_g", 0, tmp_path / "w")
    return tmp_path / "w"


def test_workdir_contains_no_hidden_labels(wd):
    import pandas as pd
    assert sorted(p.name for p in wd.iterdir()) == ["test_features.csv", "train.csv", "val_features.csv"]
    assert "target" in pd.read_csv(wd / "train.csv").columns
    assert "target" not in pd.read_csv(wd / "val_features.csv").columns
    assert "__target__" not in pd.read_csv(wd / "test_features.csv").columns


def test_successful_run_captures_stdout(wd):
    r = run_script("import pandas as pd\nprint(pd.read_csv('train.csv').shape)", wd, timeout=60)
    assert r.ok and "(600, 21)" in r.stdout


def test_exception_is_reported_not_raised(wd):
    r = run_script("raise ValueError('boom')", wd, timeout=60)
    assert not r.ok and r.exit_code != 0 and "ValueError: boom" in r.stderr


def test_infinite_loop_is_killed_within_timeout(wd):
    t0 = time.time()
    r = run_script("while True:\n    pass", wd, timeout=3)
    assert r.timed_out and not r.ok
    assert time.time() - t0 < 15


def test_sleeping_script_is_killed(wd):
    r = run_script("import time\ntime.sleep(60)", wd, timeout=2)
    assert r.timed_out


def test_secrets_are_not_inherited(wd, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "SECRET123")
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET456")
    # os is blocked by the allow-list, so read the environment with screening disabled
    r = run_script("import os\nprint('GOOGLE' in ' '.join(os.environ), 'GEMINI' in ' '.join(os.environ))",
                   wd, timeout=30, screen=False)
    assert r.ok and r.stdout.strip() == "False False"


def test_process_tree_is_killed(wd):
    # a script that spawns a long-lived child; after the timeout no descendant may survive
    code = ("import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "import time; time.sleep(120)\n")
    before = {p.pid for p in psutil.Process(os.getpid()).children(recursive=True)}
    r = run_script(code, wd, timeout=3, screen=False)
    assert r.timed_out
    time.sleep(0.5)
    leftover = {p.pid for p in psutil.Process(os.getpid()).children(recursive=True)} - before
    assert not leftover


@pytest.mark.parametrize("code,fragment", [
    ("import os", "import of 'os'"),
    ("import subprocess", "subprocess"),
    ("import socket", "socket"),
    ("import requests", "requests"),
    ("from os import path", "from 'os'"),
    ("import pandas as pd\npd.read_csv('/etc/passwd')", "leaves the working directory"),
    ("import pandas as pd\npd.read_csv('C:\\\\Users\\\\x\\\\labels.csv')", "leaves the working directory"),
    ("import pandas as pd\npd.read_csv('../labels.csv')", "leaves the working directory"),
    ("import pandas as pd\npd.read_csv('test_labels.csv')", "not provided"),
    ("eval('1+1')", "'eval'"),
    ("exec('x=1')", "'exec'"),
    ("__import__('os')", "__import__"),
    ("x = ().__class__.__bases__", "dunder"),
    ("def f(:", "SyntaxError"),
])
def test_allowlist_rejects(code, fragment):
    res = check_script(code)
    assert not res.ok
    assert any(fragment in v for v in res.violations), res.violations


def test_allowlist_accepts_typical_solution():
    code = '''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
train = pd.read_csv("train.csv")
val = pd.read_csv("val_features.csv")
test = pd.read_csv("test_features.csv")
out = pd.DataFrame({"id": np.arange(len(val))})
out.to_csv("predictions_val.csv", index=False)
out.to_csv("predictions_test.csv", index=False)
'''
    assert check_script(code).ok


def test_allowlist_accepts_fstring_path_pieces():
    code = ('name = "val"', 'open(f"predictions_{name}.csv", "w")')
    assert check_script("\n".join(code)).ok


def test_allowlist_accepts_prose_that_mentions_a_filename():
    assert check_script('print("Predictions saved to predictions_val.csv and predictions_test.csv")').ok


def test_blocked_script_is_never_executed(wd):
    r = run_script("import os\nopen('ran.txt','w').write('x')", wd, timeout=10)
    assert r.blocked and not r.ok and r.exit_code is None
    assert not (wd / "ran.txt").exists() and not (wd / "solution.py").exists()
