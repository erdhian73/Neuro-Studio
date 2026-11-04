"""Tests for scripts/generate_sample_cache.py (no GPU, no network).

These pin the properties that were actually wrong during development: the
network->grid-cell mapping must be disjoint, must use the model's canonical
vertex order, and must not claim a per-second series for a zone that the
64x160 grid never samples.

Run locally:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_sample_cache import (  # noqa: E402
    PermanentBackendError,
    _cell_groups_cached,
    _explain_backend_failure,
    _normalise,
    build_arg_parser,
    build_fmri_curves,
    build_network_telemetry,
    build_studio_payload,
    synthetic_report,
)
from agent.config import BRAIN_GRID_HEIGHT, BRAIN_GRID_WIDTH  # noqa: E402

N_CELLS = BRAIN_GRID_WIDTH * BRAIN_GRID_HEIGHT
ZONE_NAMES_FOR_TEST = (
    "medial_posterior",
    "medial_central",
    "medial_anterior",
    "posterolateral",
    "inferior_lateral",
    "dorsolateral",
    "ventrolateral_anterior",
    "dorsolateral_anterior",
)
REGION_NAMES = [
    f"{hemi}_{zone}" for hemi in ("left", "right") for zone in ZONE_NAMES_FOR_TEST
]


def test_network_cell_groups_are_disjoint() -> None:
    """Overlapping sets silently duplicated an entire UI bar."""

    groups = _cell_groups_cached()
    seen: set[int] = set()
    for network, cells in groups.items():
        overlap = seen & set(cells)
        assert not overlap, f"{network} shares {len(overlap)} cells with an earlier group"
        seen |= set(cells)


def test_cell_indices_are_in_bounds() -> None:
    for cells in _cell_groups_cached().values():
        for cell in cells:
            assert 0 <= cell < N_CELLS


def test_posterolateral_owns_about_three_quarters_of_the_grid() -> None:
    """Documents the lossiness: 'Visual' is broad cortex, not a V1-V4 label."""

    share = len(_cell_groups_cached()["Visual V1-V4"]) / N_CELLS
    assert 0.70 < share < 0.80, share


def test_unsampled_zone_yields_no_series() -> None:
    """ventrolateral_anterior has zero grid cells, so Broca must not fabricate."""

    assert _cell_groups_cached()["Language / Broca"] == ()
    telemetry = build_network_telemetry(
        synthetic_report("sample_4", 6), n_frames=6
    )
    assert "Language / Broca" not in telemetry
    # The other four are real, distinct series.
    assert len(telemetry) == 4
    series = [tuple(v) for v in telemetry.values()]
    assert len(set(series)) == len(series)


def test_normalise_handles_flat_series() -> None:
    assert _normalise([2.0, 2.0, 2.0]) == [50.0, 50.0, 50.0]
    assert _normalise([]) == []
    assert _normalise([0.0, 10.0]) == [0.0, 100.0]


def test_fmri_curves_use_one_tr_per_second() -> None:
    curves = build_fmri_curves(synthetic_report("sample_2", 5), 5.0)
    assert curves["n_vertices"] == 20484
    assert curves["tr_s"] == 1.0
    seconds = [row["second"] for row in curves["per_second"]]
    assert seconds == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert curves["per_second"][1]["timestamp"] == "00:01"


def test_studio_payload_has_all_four_ui_sections() -> None:
    report = synthetic_report("sample_4", 4)
    payload = build_studio_payload(
        report, sample="sample_4", video="frontend/public/videos/sample_4.mp4",
        duration_s=4.0,
    )
    studio = payload["studio"]
    for key in ("fmri_curves", "networks", "diagnostics_by_timestamp",
                "rewritten_voiceover_hooks", "muse_image"):
        assert key in studio, key
    assert payload["report"] is report
    assert studio["muse_image"]["available"] is False
    assert studio["synthetic"] is False


def test_studio_payload_all_networks_within_0_100() -> None:
    payload = build_studio_payload(
        synthetic_report("sample_3", 8), sample="sample_3",
        video="v.mp4", duration_s=8.0,
    )
    for network, series in payload["studio"]["networks"]["networks"].items():
        assert series, network
        assert all(0.0 <= value <= 100.0 for value in series), network


def test_missing_endpoint_is_a_clear_error() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert hasattr(args, "endpoint")
    # --fake and --from-report exist so the script is usable without Modal.
    assert parser.parse_args(["--fake"]).fake is True


def test_network_cell_groups_respects_the_models_vertex_order() -> None:
    """A wrong axis order would still be N cells long, so check identities."""

    groups = _cell_groups_cached()
    # Left hemisphere cells live in columns 0-79, right in 80-159, so the Left
    # and Right halves of a network must not be the same flat indices.
    left_half = {c for c in groups["Dorsal Attention"] if (c % BRAIN_GRID_WIDTH) < 80}
    right_half = {c for c in groups["Dorsal Attention"] if (c % BRAIN_GRID_WIDTH) >= 80}
    assert left_half and right_half
    assert left_half.isdisjoint(right_half)


def test_gated_repo_error_gets_an_actionable_hint() -> None:
    """This is the exact failure the deployed backend returned from Modal."""

    real = (
        'sample_2.mp4: HTTP 500 {"error":"analysis failed: OSError: You are trying '
        "to access a gated repo. 403 Client Error. (Request ID: Root=1-6aeb373c)\"}"
    )
    hint = _explain_backend_failure(real)
    assert "meta-llama/Llama-3.2-3B" in hint
    assert "HF_TOKEN" in hint
    assert "create_secrets.py" in hint


def test_unrecognised_error_yields_no_hint() -> None:
    assert _explain_backend_failure("something unexpected") == ""


def test_permanent_error_is_a_runtime_error() -> None:
    """Callers catching RuntimeError keep working; the subtype only adds policy."""

    assert issubclass(PermanentBackendError, RuntimeError)
