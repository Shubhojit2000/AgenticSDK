"""``python -m agenticsdk.service [--port 7860] [--host 127.0.0.1]``: run the web service (API at /docs, UI at /ui)."""
import argparse

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    uvicorn.run("agenticsdk.service.app:app", host=a.host, port=a.port)
