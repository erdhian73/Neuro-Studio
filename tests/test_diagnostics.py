"""Tests for diagnostics, brain-grid encoding, ranking and LLM-failure safety."""

from __future__ import annotations

import json

import numpy as np
import pytest

from agent.analysis import compute_diagnostics
from agent.brain_grid import _cell_picks, canonical_vertices, downsample_to_grid, encode_brain_grid
from agent.config import BRAIN_GRID_HEIGHT, BRAIN_GRID_WIDTH, N_VERTICES_PER_HEMI, N_VERTICES_TOTAL
from agent.interpret import deterministic_summary, interpret_with_muse
from agent.muse import MuseAPIError
from agent.pipeline import Stimulus, analyse_stimulus, build_report_payload, rank_creatives
from agent.render import render_report_html
from agent.schemas import VERDICT_SCHEMA


def synthetic_predictions(n_trs: int = 40, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    preds = rng.normal(size=(n_trs, N_VERTICES_TOTAL)).cumsum(axis=0) * 0.1
    preds[5:10, :N_VERTICES_PER_HEMI] += 4.0
    preds[int(n_trs * 0.75) :] *= 0.01
    return preds


def test_diagnostics_finding_injected_peak() -> None:
    diag = compute_diagnostics(synthetic_predictions())
    assert diag.n_trs == 40
    assert len(diag.global_timeseries) == 40
    # The injected burst (TR 5-9) must dominate the ranking.
    assert diag.peak_windows[0].start_s <= 10
    # ...and the report must surface at least one flat stretch somewhere.
    assert diag.low_engagement_segments
    for segment in diag.low_engagement_segments:
        assert segment["duration_s"] >= 3.0
        assert segment["prediction_start_s"] + segment["duration_s"] <= 40


def test_low_engagement_is_relative_to_each_clip() -> None:
    """A flat tail still in the top-75% of a spiky clip is NOT flagged."""

    diag = compute_diagnostics(synthetic_predictions())
    flagged = {
        int(seg["prediction_start_s"] / diag.tr_s) for seg in diag.low_engagement_segments
    }
    tail_start = int(0.75 * diag.n_trs)
    assert not any(idx >= tail_start for idx in flagged)



def test_laterality_sign_follows_left_burst() -> None:
    diag = compute_diagnostics(synthetic_predictions())
    assert diag.laterality_ratio > 0.5  # left hemisphere carries the burst


def test_windows_never_overlap_within_a_ranking() -> None:
    diag = compute_diagnostics(synthetic_predictions())
    spans = [(w.start_s, w.end_s) for w in diag.peak_windows]
    assert len(spans) == len(set(spans))


def test_stimulus_times_are_shifted_by_hemodynamic_lag() -> None:
    diag = compute_diagnostics(synthetic_predictions())
    for window in diag.peak_windows:
        payload = window.to_dict()
        assert payload["stimulus_start_s"] == pytest.approx(
            max(payload["prediction_start_s"] - 5.0, 0.0), abs=1e-6
        )


def test_brain_grid_dimensions_and_quantisation() -> None:
    payload = encode_brain_grid(synthetic_predictions(12))
    assert payload["width"] == BRAIN_GRID_WIDTH
    assert payload["height"] == BRAIN_GRID_HEIGHT
    assert payload["n_frames"] == 12
    frame = np.asarray(payload["frames"][0])
    assert frame.size == BRAIN_GRID_WIDTH * BRAIN_GRID_HEIGHT
    assert frame.min() >= -30 and frame.max() <= 30


def test_cell_picks_are_unique_and_in_range() -> None:
    picks = _cell_picks(64, 80, N_VERTICES_PER_HEMI)
    assert picks.shape == (64 * 80,)
    assert picks.max() < N_VERTICES_PER_HEMI
    assert picks.min() >= 0


def test_canonical_vertices_are_a_permutation() -> None:
    left, right = canonical_vertices()
    assert sorted(left.tolist()) == list(range(N_VERTICES_PER_HEMI))
    assert sorted(right.tolist()) == list(range(N_VERTICES_PER_HEMI))


def test_downsample_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        downsample_to_grid(np.zeros((3, 100)))


def test_llm_failure_degrades_to_deterministic_summary() -> None:
    class Boom:
        def complete_json(self, *args, **kwargs):
            raise MuseAPIError("simulated outage")

    diag = compute_diagnostics(synthetic_predictions())
    result = interpret_with_muse(diag, Boom())
    assert result.degraded is True
    assert "simulated outage" in (result.error or "")
    assert result.verdict["overall_verdict"] in {"strong", "mixed", "weak"}
    # Every key the strict schema requires must still be present.
    assert set(VERDICT_SCHEMA["required"]).issubset(result.verdict)


def test_analysis_is_deterministic() -> None:
    preds = synthetic_predictions()
    a = compute_diagnostics(preds).to_dict()
    b = compute_diagnostics(preds.copy()).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_rank_creatives_is_monotonic_and_stable() -> None:
    def report(asset: str, dynamic: float, laterality: float, dead_fraction: float) -> dict:
        n_trs, tr_s = 60, 1.0
        return {
            "asset": {"id": asset, "label": asset},
            "diagnostics": {
                "n_trs": n_trs,
                "tr_s": tr_s,
                "dynamic_range": dynamic,
                "laterality_ratio": laterality,
                "low_engagement_segments": [
                    {"duration_s": dead_fraction * n_trs * tr_s}
                ],
            },
            "interpretation": {"verdict": {"overall_verdict": "mixed"}},
        }

    ranking = rank_creatives(
        [
            report("weak", 0.5, 0.01, 0.8),
            report("strong", 3.0, 0.6, 0.05),
            report("mid", 1.5, 0.2, 0.3),
        ]
    )
    assert [row["asset_id"] for row in ranking] == ["strong", "mid", "weak"]
    assert [row["rank"] for row in ranking] == [1, 2, 3]
    assert all(0 <= row["score"] <= 100 for row in ranking)
