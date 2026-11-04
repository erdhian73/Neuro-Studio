"""Unit tests for the deterministic core (no GPU, no network, no API keys).

Run locally:  python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.analysis import (
    PredictionShapeError,
    compute_diagnostics,
    region_timeseries,
    validate_predictions,
)
from agent.config import N_VERTICES_PER_HEMI, N_VERTICES_TOTAL
from agent.pipeline import rank_creatives
from agent.roi import mesh_region_map


def synthetic_predictions(n_trs: int = 30, seed: int = 0) -> np.ndarray:
    """Random walk with an injected burst and a damped tail."""

    rng = np.random.default_rng(seed)
    preds = rng.normal(size=(n_trs, N_VERTICES_TOTAL)).cumsum(axis=0) * 0.1
    preds[3:7, :N_VERTICES_PER_HEMI] += 3.0  # left-heavy burst early on
    preds[int(n_trs * 0.7) :] *= 0.01  # flat tail
    return preds


def test_validate_predictions_rejects_wrong_width() -> None:
    with pytest.raises(PredictionShapeError):
        validate_predictions(np.zeros((5, 100)))


def test_validate_predictions_rejects_nan() -> None:
    bad = np.zeros((4, N_VERTICES_TOTAL))
    bad[2, 7] = np.nan
    with pytest.raises(PredictionShapeError):
        validate_predictions(bad)


def test_validate_predictions_accepts_documented_shape() -> None:
    out = validate_predictions(np.ones((9, N_VERTICES_TOTAL)))
    assert out.shape == (9, N_VERTICES_TOTAL)
    assert out.dtype == np.float64


def test_region_map_is_complete_and_hemispheric() -> None:
    regions = mesh_region_map()
    assert regions.labels.size == N_VERTICES_TOTAL
    assert regions.labels.min() >= 0
    assert regions.labels.max() < regions.n_regions
    # Left hemisphere ids must all be lower than right hemisphere ids.
    left_max = regions.labels[:N_VERTICES_PER_HEMI].max()
    right_min = regions.labels[N_VERTICES_PER_HEMI:].min()
    assert left_max < right_min
    assert all(name.startswith("left_") for name in regions.names[: regions.n_regions // 2])
    assert all(name.startswith("right_") for name in regions.names[regions.n_regions // 2 :])


def test_region_timeseries_shape_and_values() -> None:
    regions = mesh_region_map()
    preds = np.arange(4 * N_VERTICES_TOTAL, dtype=np.float64).reshape(4, N_VERTICES_TOTAL)
    series = region_timeseries(preds, regions)
    assert series.shape == (regions.n_regions, 4)
    # Region means must match numpy's own mean over the same vertex mask.
    for rid in range(regions.n_regions):
        mask = regions.labels == rid
        if mask.any():
            np.testing.assert_allclose(series[rid], preds[:, mask].mean(axis=1))
