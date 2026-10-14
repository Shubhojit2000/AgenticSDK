"""Assemble a folder that can be pushed to a Hugging Face Space (Docker SDK).

    python deploy/huggingface/build_space.py [target_dir]        # default: deploy/huggingface/space

Then, from that folder (needs your own Hugging Face account and a write token; this script never touches the network):

    git init && git remote add origin https://huggingface.co/spaces/<you>/agenticsdk
    git add . && git commit -m "AgenticSDK demo" && git push origin HEAD:main

and in the Space's Settings > Variables and secrets add the secret GOOGLE_API_KEY (a free Gemini key) and the variables
DEMO_MODE=1 and DEMO_DAILY_CALLS=300.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def build(target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(ROOT / "agenticsdk", target / "agenticsdk",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "Dockerfile", ".dockerignore"):
        shutil.copy(ROOT / name, target / name)
    shutil.copy(Path(__file__).parent / "README.md", target / "README.md")
    return target


if __name__ == "__main__":
    out = build(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "space")
    print("Space folder ready:", out)
