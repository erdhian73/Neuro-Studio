"""Pipeline-level tests (fake predictor, no GPU) + static checks on app.py."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from agent.config import N_VERTICES_PER_HEMI, N_VERTICES_TOTAL
from agent.pipeline import (
    Stimulus,
    analyse_stimulus,
    build_report_payload,
    report_to_json_bytes,
)
from agent.render import render_report_html

REPO_ROOT = Path(__file__).resolve().parents[1]


def fake_predict(path: Path) -> np.ndarray:
    rng = np.random.default_rng(len(Path(path).name))
    preds = rng.normal(size=(24, N_VERTICES_TOTAL)).cumsum(axis=0) * 0.1
    preds[4:9, :N_VERTICES_PER_HEMI] += 2.5
    return preds


@pytest.fixture()
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "ad.mp4"
    path.write_bytes(b"\x00" * 2048)
    return path


def test_analyse_stimulus_returns_complete_report(video_file: Path) -> None:
    report = analyse_stimulus(Stimulus("ad-1", video_file), fake_predict, use_llm=False)

    for key in ("asset", "predictions", "regions", "diagnostics", "brain_grid",
                "interpretation", "timings"):
        assert key in report
    assert report["predictions"]["n_vertices"] == N_VERTICES_TOTAL
    assert report["interpretation"]["degraded"] is True  # LLM disabled
    assert report["asset"]["bytes"] == 2048

    # Must be JSON-serialisable end to end.
    encoded = report_to_json_bytes(report)
    assert b'"brain_grid"' in encoded


def test_report_payload_and_html(video_file: Path) -> None:
    reports = [
        analyse_stimulus(Stimulus(f"ad-{i}", video_file), fake_predict, use_llm=False)
        for i in range(2)
    ]
    payload = build_report_payload({"name": "Launch <script>"}, reports, include_brain_grid=False)
    assert payload["winner"] in {r["asset"]["id"] for r in reports}
    assert len(payload["ranking"]) == 2
    assert all("brain_grid" not in c for c in payload["creatives"])

    html = render_report_html(payload)
    assert html.startswith("<!doctype html>")
    assert "Launch &lt;script&gt;" in html  # escaping works
    assert "not measurements\nof real viewers" in html or "not measurements of real viewers" in html
    assert "not a forecast" in html


def test_app_py_declares_required_modal_primitives() -> None:
    """Guard the deployment contract with a static check (no GPU needed)."""

    tree = ast.parse((REPO_ROOT / "app.py").read_text())
    decorators: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                decorators.append(ast.unparse(dec))

    joined = "\n".join(decorators)
    assert "app.cls(" in joined, "GPU class must use @app.cls"
    assert "modal.enter()" in joined, "model must load once via @modal.enter"
    assert "modal.fastapi_endpoint(" in joined, "must expose an HTTPS endpoint"
    assert "modal.method()" in joined, "class needs @modal.method entrypoints"

    cls_call = next(d for d in decorators if d.startswith("app.cls("))
    assert "gpu=GPU_TYPE" in cls_call
    assert "cpu=WORKER_CPU" in cls_call
    assert "memory=WORKER_MEMORY_GIB" in cls_call
    assert "volumes=" in cls_call
    assert "timeout=MAX_TIMEOUT_S" in cls_call
    assert "scaledown_window=SCALEDOWN_WINDOW_S" in cls_call

    source = (REPO_ROOT / "app.py").read_text()
    # Documented Modal limits and the TRIBE/uv requirements.
    assert "MAX_TIMEOUT_S = 24 * 60 * 60" in source
    assert "SCALEDOWN_WINDOW_S = 7 * 60" in source, "idle GPU must scale down in 7 min"
    assert "WORKER_CPU = 6" in source
    assert "WORKER_MEMORY_GIB = 32" in source
    assert 'GPU_TYPE = os.environ.get("TRIBE_GPU", "L40S")' in source
    assert "from tribev2 import TribeModel" in source
    assert "modal.Volume.from_name" in source
    assert "uvx" in source  # PATH fix for TRIBE's whisperx subprocess
