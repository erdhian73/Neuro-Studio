#!/usr/bin/env python3
"""Smoke-test a *deployed* endpoint over plain HTTPS.

Usage::

    python scripts/smoke_test.py https://<workspace>--tribe-ad-pretest-tribeadanalyzer-healthz.us-east.modal.run
    python scripts/smoke_test.py https://....modal.run --video samples/ad.mp4

Exits non-zero on any failed check so it can gate CI.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def check_health(base: str, timeout: float = 120.0) -> dict:
    url = base.rstrip("/")
    if url.endswith("-healthz.modal.run"):
        resp = requests.get(url, timeout=timeout)
    else:
        resp = requests.get(f"{url}/healthz", timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    assert body.get("ok") is True, body
    print("healthz OK:", json.dumps(body, indent=2))
    return body


def analyze(base: str, video: Path, timeout: float = 1800.0) -> dict:
    payload = {
        "asset_id": video.stem,
        "filename": video.name,
        "label": video.stem,
        "creative_context": "smoke test",
        "video_base64": base64.b64encode(video.read_bytes()).decode("ascii"),
    }
    resp = requests.post(base.rstrip("/"), json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    diag = body["diagnostics"]
    print(
        "analyze OK:",
        f"trs={body['predictions']['n_trs']}",
        f"grid={body['brain_grid']['width']}x{body['brain_grid']['height']}",
        f"peaks={len(diag['peak_windows'])}",
        f"flat={len(diag['low_engagement_segments'])}",
        f"verdict={body['interpretation']['verdict']['overall_verdict']}",
    )
    assert body["predictions"]["n_vertices"] == 20484
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="healthz or analyze endpoint URL")
    parser.add_argument("--video", type=Path, help="optional mp4 to analyse")
    args = parser.parse_args()

    check_health(args.url)
    if args.video:
        analyze(args.url, args.video)
    print("\nall smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
