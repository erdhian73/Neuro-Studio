"""Studio payload contract: cache and live paths must emit identical shapes.

The frontend renders one component tree against whichever payload it is given,
so any divergence between the offline cache and the deployed endpoints is a
silent UI bug. These tests pin the shared builder and the live-job plumbing.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.studio import (  # noqa: E402
    NETWORK_ZONES,
    build_studio_payload,
    fmt_mmss,
    normalise,
    zones_for_network,
)
from agent.config import N_VERTICES_TOTAL  # noqa: E402
from agent.pipeline import STAGES, Stimulus, analyse_stimulus  # noqa: E402
from scripts.generate_sample_cache import build_studio_payload as cache_builder  # noqa: E402


def fake_predict(path: Path) -> np.ndarray:
    """Deterministic predictions with a clear mid-clip burst."""

    rng = np.random.default_rng(len(Path(path).name))
    preds = rng.normal(size=(24, N_VERTICES_TOTAL)).cumsum(axis=0) * 0.1
    preds[4:9, : N_VERTICES_TOTAL // 2] += 2.5
    return preds


@pytest.fixture()
def report(tmp_path: Path) -> dict:
    video = tmp_path / "ad.mp4"
    video.write_bytes(b"\x00" * 4096)
    return analyse_stimulus(Stimulus("ad-1", video), fake_predict, use_llm=False)


def test_cache_and_live_builders_agree(report: dict) -> None:
    """The script's envelope and the API block must share one implementation."""

    kwargs = {"sample": "ad-1", "video": "/videos/ad.mp4", "duration_s": 24.0}
    live = build_studio_payload(report, **kwargs)
    cached = cache_builder(report, **kwargs)

    assert live["studio"] == cached["studio"]
    # The cache adds only generation metadata on top.
    assert set(cached) - set(live) == {"generated_at"}


def test_studio_block_has_every_field_the_ui_reads(report: dict) -> None:
    studio = build_studio_payload(
        report, sample="ad-1", video="/videos/ad.mp4", duration_s=24.0
    )["studio"]

    for key in (
        "sample",
        "video",
        "schema_version",
        "synthetic",
        "fmri_curves",
        "networks",
        "diagnostics_by_timestamp",
        "rewritten_voiceover_hooks",
        "verdict",
        "muse_image",
        "low_engagement_segments",
        "note",
    ):
        assert key in studio, key

    # Per-second series must line up with the TR count, or the playhead
    # sampling in the UI would index past the end.
    n_trs = report["predictions"]["n_trs"]
    assert len(studio["fmri_curves"]["per_second"]) == n_trs
    assert studio["fmri_curves"]["n_vertices"] == report["predictions"]["n_vertices"]
    for series in studio["networks"]["networks"].values():
        assert len(series) == n_trs


def test_unsampled_network_is_reported_static_not_animated(report: dict) -> None:
    """`Language / Broca` has no grid cells, so it must not claim a series."""

    networks = build_studio_payload(
        report, sample="ad-1", video="/videos/ad.mp4", duration_s=24.0
    )["studio"]["networks"]

    assert "Language / Broca" not in networks["networks"]
    assert networks["coverage"]["Language / Broca"]["has_series"] is False
    # ...but it still gets an honestly-labelled clip-level mean.
    assert "Language / Broca" in networks["static_region_mean_z"]
    assert "static_note" in networks


def test_low_engagement_segments_are_passed_through(report: dict) -> None:
    studio = build_studio_payload(
        report, sample="ad-1", video="/videos/ad.mp4", duration_s=24.0
    )["studio"]

    expected = report["diagnostics"]["low_engagement_segments"]
    assert studio["low_engagement_segments"] == expected


def test_zone_mapping_is_disjoint() -> None:
    """Overlapping zones produced duplicate bars, so this is a hard invariant."""

    seen: dict[str, str] = {}
    for network, zones in NETWORK_ZONES.items():
        for zone in zones:
            assert zone not in seen, f"{zone} claimed by {seen.get(zone)} and {network}"
            seen[zone] = network


def test_normalise_is_bounded_and_flat_safe() -> None:
    assert normalise([]) == []
    assert normalise([5.0, 5.0, 5.0]) == [50.0, 50.0, 50.0]
    assert normalise([-1.0, 0.0, 1.0]) == [0.0, 50.0, 100.0]


def test_fmt_mmss_matches_ui_format() -> None:
    assert fmt_mmss(0) == "00:00"
    assert fmt_mmss(7.4) == "00:07"
    assert fmt_mmss(65) == "01:05"


def test_zones_for_network_strips_hemisphere_prefix() -> None:
    names = ["left_posterolateral", "right_posterolateral", "left_inferior_lateral"]
    assert zones_for_network("Visual V1-V4", names) == [0, 1]
    assert zones_for_network("Auditory", names) == [2]


def test_progress_hook_emits_real_stages_in_order(tmp_path: Path) -> None:
    """`/jobs` streams these events, so their order and payloads are a contract."""

    video = tmp_path / "ad.mp4"
    video.write_bytes(b"\x00" * 1024)

    events: list[tuple[str, dict]] = []
    analyse_stimulus(
        Stimulus("ad-1", video),
        fake_predict,
        use_llm=False,
        progress=lambda stage, payload: events.append((stage, payload)),
    )

    stages = [stage for stage, _ in events]
    assert stages == list(STAGES)
    assert stages[-1] == "done"

    by_stage = dict(events)
    assert by_stage["predict"]["n_vertices"] == N_VERTICES_TOTAL
    assert by_stage["predict"]["n_trs"] == 24
    assert "peak_windows" in by_stage["diagnostics"]["diagnostics"]
    assert by_stage["brain_grid"]["brain_grid"]["n_frames"] == 24
    assert by_stage["done"]["degraded"] is True  # LLM disabled


def test_analyse_stimulus_without_progress_still_works(tmp_path: Path) -> None:
    """The hook is optional; existing callers must not have to pass it."""

    video = tmp_path / "ad.mp4"
    video.write_bytes(b"\x00" * 512)
    report = analyse_stimulus(Stimulus("ad-1", video), fake_predict, use_llm=False)
    assert report["asset"]["id"] == "ad-1"


# ---------------------------------------------------------------------------
# Warmup contract
#
# The page pings `GET /warmup` on load to pull the L40S out of scale-down. That
# only works if the handler is declared on the GPU class (a plain
# `@app.function` would wake a second, model-less container) and if the frontend
# derives the same hostname for it. Both are static properties of the source, so
# they are checked here rather than against a live deployment.
# ---------------------------------------------------------------------------

REPO_ROOT_FOR_APP = Path(__file__).resolve().parents[1]


def _warmup_node() -> ast.FunctionDef:
    tree = ast.parse((REPO_ROOT_FOR_APP / "app.py").read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TribeAdAnalyzer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "warmup":
                    return item

    raise AssertionError("TribeAdAnalyzer.warmup is missing: the page pings GET /warmup")


def test_warmup_is_a_get_endpoint_on_the_gpu_class() -> None:
    node = _warmup_node()

    decorators = [ast.unparse(dec) for dec in node.decorator_list]
    # `ast.unparse` renders string literals with single quotes, hence "method='GET'".
    assert any(
        dec.startswith("modal.fastapi_endpoint(") and "method='GET'" in dec for dec in decorators
    ), decorators

    # Being inside TribeAdAnalyzer is what makes the call pay @modal.enter()
    # (the 30 GB weight load) on the L40S instead of on a web container.
    assert not any(dec.startswith(("modal.function(", "app.function(")) for dec in decorators)


def test_warmup_does_not_run_inference() -> None:
    """A ping must never read a video, start a job, or call the model API."""

    called = {
        node.func.id
        for node in ast.walk(_warmup_node())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for forbidden in ("open", "_predict_path"):
        assert forbidden not in called, f"{forbidden}() does not belong in a warmup ping"

    source = (
        ast.get_source_segment((REPO_ROOT_FOR_APP / "app.py").read_text(), _warmup_node()) or ""
    )
    for forbidden in ("analyze_video", "video_base64", "MuseSparkClient", "job_state"):
        assert forbidden not in source, f"warmup must not touch {forbidden}"


def test_warmup_reports_uptime_per_request() -> None:
    """`container_ready_s` must be measured per call, not cached in @modal.enter.

    `@modal.enter()` runs once per container, so a value stored there freezes:
    verified live on the deployed app, where every ping reported the model-load
    time (12.86 s) minutes later. The field is computed inside the handler
    instead, from the module-scope clock that starts with the container.
    """

    source = (REPO_ROOT_FOR_APP / "app.py").read_text()
    assert "_CONTAINER_STARTED_AT = time.time()" in source
    assert "round(time.time() - _CONTAINER_STARTED_AT, 2)" in source

    def source_of(name: str) -> str:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TribeAdAnalyzer":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == name:
                        return ast.get_source_segment(source, item) or ""
        raise AssertionError(f"TribeAdAnalyzer.{name} is missing")

    assert "ready_at_s" not in source_of("load_model"), "uptime must not be snapshotted"
    assert "_CONTAINER_STARTED_AT" in source_of("warmup")


def test_frontend_pings_warmup_and_derives_its_hostname() -> None:
    api = (REPO_ROOT_FOR_APP / "frontend/src/lib/api.ts").read_text()
    page = (REPO_ROOT_FOR_APP / "frontend/src/app/page.tsx").read_text()

    # Slug registered in both the name union and the slug map.
    assert "| 'warmup'" in api
    assert "warmup: 'warmup'," in api
    # A pasted /warmup URL must be normalised like every other endpoint.
    assert "jobs|job-events|job-status|healthz|warmup" in api
    # Fire-and-forget: failures are swallowed so the page never shows an error.
    assert "export function pingWarmup(" in api
    assert ".catch(() => undefined)" in api
    # Called once from the main page on mount.
    assert "pingWarmup()" in page
    assert "useEffect(() => {\n    void pingWarmup()\n  }, [])" in page

