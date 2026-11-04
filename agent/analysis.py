"""Turn raw TRIBE v2 vertex predictions into interpretable ad diagnostics.

Input contract
--------------
``preds`` is exactly what ``TribeModel.predict`` returns:
shape ``(n_trs, 20484)``, left hemisphere first, sampled every TR (1 s in the
released checkpoint), already shifted 5 s to compensate the hemodynamic lag.

Everything in this module is pure numpy + stdlib so it can be unit-tested
without a GPU, a network, or the 30 GB of weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import (
    DEFAULT_TR_S,
    HEMODYNAMIC_LAG_S,
    LOW_ENGAGEMENT_MIN_S,
    LOW_ENGAGEMENT_QUANTILE,
    N_KEY_WINDOWS,
    N_VERTICES_TOTAL,
)
from .roi import RegionMap, get_region_map


class PredictionShapeError(ValueError):
    """Raised when predictions do not match TRIBE's documented output shape."""


def validate_predictions(preds: np.ndarray) -> np.ndarray:
    """Coerce *preds* to float64 and enforce the documented shape."""

    arr = np.asarray(preds)
    if arr.ndim != 2:
        raise PredictionShapeError(
            f"preds must be 2-D (n_timesteps, n_vertices); got shape {arr.shape}"
        )
    if arr.shape[1] != N_VERTICES_TOTAL:
        raise PredictionShapeError(
            f"preds must have {N_VERTICES_TOTAL} vertex columns (fsaverage5 x2); "
            f"got {arr.shape[1]}"
        )
    if arr.shape[0] == 0:
        raise PredictionShapeError("preds contains no time steps")
    arr = arr.astype(np.float64, copy=False)
    if not np.isfinite(arr).all():
        raise PredictionShapeError("preds contains NaN or infinite values")
    return arr


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    """Row-wise z-score with a safe denominator for flat rows."""

    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.where(std < 1e-12, 1.0, std)


def region_timeseries(preds: np.ndarray, regions: RegionMap) -> np.ndarray:
    """Mean activation per region per time step -> ``(n_regions, n_trs)``.

    Region means are computed on the *raw* predicted values (same units as
    TRIBE output), so ratios and orderings stay interpretable.
    """

    n_trs = preds.shape[0]
    out = np.zeros((regions.n_regions, n_trs), dtype=np.float64)
    for rid in range(regions.n_regions):
        mask = regions.labels == rid
        if not mask.any():  # pragma: no cover - guarded by zone catch-all
            continue
        out[rid] = preds[:, mask].mean(axis=1)
    return out


def hemisphere_timeseries(preds: np.ndarray) -> dict[str, np.ndarray]:
    """Split TRIBE's concatenated vector into left/right mean time series."""

    half = preds.shape[1] // 2
    return {
        "left": preds[:, :half].mean(axis=1),
        "right": preds[:, half:].mean(axis=1),
    }


@dataclass
class TimeWindow:
    """One ranked window of the ad, in both prediction time and video time."""

    rank: int
    start_s: float  # within the prediction timeline
    end_s: float
    score: float  # mean z-scored brain response inside the window
    top_regions: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "prediction_start_s": round(self.start_s, 3),
            "prediction_end_s": round(self.end_s, 3),
            # The viewer sees a moment ~5 s before the BOLD response TRIBE models.
            "stimulus_start_s": round(max(self.start_s - HEMODYNAMIC_LAG_S, 0.0), 3),
            "stimulus_end_s": round(max(self.end_s - HEMODYNAMIC_LAG_S, 0.0), 3),
            "score": round(float(self.score), 4),
            "top_regions": [
                {"region": name, "score": round(float(val), 4)}
                for name, val in self.top_regions
            ],
        }


@dataclass
class BrainDiagnostics:
    """Deterministic, model-side diagnostics for a single stimulus."""

    n_trs: int
    tr_s: float
    global_timeseries: list[float]  # z-scored across time, length n_trs
    hemisphere_mean: dict[str, float]
    laterality_ratio: float  # (L - R) / (|L| + |R|), in [-1, 1]
    region_mean_z: dict[str, float]
    peak_windows: list[TimeWindow]
    quiet_windows: list[TimeWindow]
    low_engagement_segments: list[dict[str, float]]
    dynamic_range: float
    instantaneous_peaks: list[dict[str, float]]
    region_strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trs": self.n_trs,
            "tr_s": self.tr_s,
            "dynamic_range": round(float(self.dynamic_range), 4),
            "hemisphere_mean": {
                k: round(float(v), 4) for k, v in self.hemisphere_mean.items()
            },
            "laterality_ratio": round(float(self.laterality_ratio), 4),
            "region_mean_z": {
                k: round(float(v), 4) for k, v in self.region_mean_z.items()
            },
            "peak_windows": [w.to_dict() for w in self.peak_windows],
            "quiet_windows": [w.to_dict() for w in self.quiet_windows],
            "low_engagement_segments": self.low_engagement_segments,
            "instantaneous_peaks": self.instantaneous_peaks,
            "global_timeseries": [round(float(v), 4) for v in self.global_timeseries],
            "region_strategy": self.region_strategy,
        }


def _window_scores(series: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, edge-padded so output aligns with input time."""

    if window <= 1 or series.size <= 1:
        return series.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad = window // 2
    padded = np.pad(series, (pad, pad), mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    return smooth[: series.size]


def _top_regions_at(
    zscores: np.ndarray, regions: RegionMap, start_idx: int, end_idx: int, k: int = 3
) -> list[tuple[str, float]]:
    """Regions whose z-scored response is highest inside ``[start, end)``."""

    segment = zscores[:, start_idx:end_idx]
    scores = segment.mean(axis=1) if segment.size else np.zeros(regions.n_regions)
    order = np.argsort(scores)[::-1][:k]
    return [(regions.names[i], float(scores[i])) for i in order]


def _find_low_engagement(
    global_z: np.ndarray,
    tr_s: float,
    *,
    quantile: float = LOW_ENGAGEMENT_QUANTILE,
    min_duration_s: float = LOW_ENGAGEMENT_MIN_S,
) -> list[dict[str, float]]:
    """Contiguous stretches where the response stays in its own bottom quantile.

    The quantile is computed per clip, so this is a *relative* measure -- which
    parts are flattest *for this ad* -- never an absolute claim about attention
    or emotion.
    """

    if global_z.size == 0:
        return []
    threshold = float(np.quantile(global_z, quantile))
    flat = global_z <= threshold

    segments: list[dict[str, float]] = []
    index, n = 0, flat.size
    while index < n:
        if not flat[index]:
            index += 1
            continue
        start = index
        while index < n and flat[index]:
            index += 1
        duration = (index - start) * tr_s
        if duration + 1e-9 >= min_duration_s:
            segments.append(
                {
                    "prediction_start_s": round(start * tr_s, 3),
                    "prediction_end_s": round(index * tr_s, 3),
                    "duration_s": round(float(duration), 3),
                    "stimulus_start_s": round(
                        max(start * tr_s - HEMODYNAMIC_LAG_S, 0.0), 3
                    ),
                    "mean_z_in_segment": round(float(global_z[start:index].mean()), 4),
                    "threshold_z": round(threshold, 4),
                }
            )
    return segments



def compute_diagnostics(
    preds: np.ndarray,
    *,
    tr_s: float = DEFAULT_TR_S,
    window_s: float = 4.0,
    regions: RegionMap | None = None,
    n_key_windows: int = N_KEY_WINDOWS,
) -> BrainDiagnostics:
    """Full deterministic diagnostic pass over one stimulus' predictions."""

    arr = validate_predictions(preds)
    region_map = regions or get_region_map()

    n_trs = int(arr.shape[0])
    region_ts = region_timeseries(arr, region_map)  # (n_regions, n_trs)
    hemi_ts = hemisphere_timeseries(arr)
    region_z = _zscore_rows(region_ts)

    # The "global" response is the mean across regions. Averaging z-scores of
    # regions (rather than raw vertices) keeps every region equally weighted.
    global_z = region_z.mean(axis=0)

    window = max(int(round(window_s / tr_s)), 1)
    smoothed = _window_scores(global_z, window)

    # Rank non-overlapping candidate windows so the report never cites the same
    # seconds twice. try/except keeps code path explicit for short clips.
    stride = max(window, 1)
    candidates: list[tuple[int, int, float]] = []
    for start in range(0, max(n_trs - window + 1, 1), stride):
        end = min(start + window, n_trs)
        candidates.append((start, end, float(smoothed[start:end].mean())))
    if not candidates:
        candidates = [(0, n_trs, float(smoothed.mean()))]

    ranked = sorted(candidates, key=lambda item: item[2], reverse=True)
    n_pick = max(1, min(n_key_windows, len(ranked)))

    used: set[tuple[int, int]] = set()

    def _take(pool: list[tuple[int, int, float]]) -> list[TimeWindow]:
        picked: list[TimeWindow] = []
        for start, end, score in pool:
            if len(picked) >= n_pick:
                break
            if (start, end) in used:
                continue
            used.add((start, end))
            picked.append(
                TimeWindow(
                    rank=len(picked) + 1,
                    start_s=start * tr_s,
                    end_s=end * tr_s,
                    score=score,
                    top_regions=_top_regions_at(region_z, region_map, start, end),
                )
            )
        return picked

    peak_windows = _take(ranked)
    quiet_windows = _take(sorted(candidates, key=lambda item: item[2]))

    hemi_means = {k: float(v.mean()) for k, v in hemi_ts.items()}
    denom = abs(hemi_means["left"]) + abs(hemi_means["right"])
    laterality = (hemi_means["left"] - hemi_means["right"]) / denom if denom > 1e-12 else 0.0

    peak_idx = int(np.argmax(global_z))
    peak_frame_z = region_z[:, peak_idx]
    top_order = np.argsort(peak_frame_z)[::-1][:5]

    return BrainDiagnostics(
        n_trs=n_trs,
        tr_s=float(tr_s),
        global_timeseries=[float(v) for v in global_z],
        hemisphere_mean=hemi_means,
        laterality_ratio=float(laterality),
        region_mean_z={
            name: float(region_z[i].mean()) for i, name in enumerate(region_map.names)
        },
        peak_windows=peak_windows,
        quiet_windows=quiet_windows,
        low_engagement_segments=_find_low_engagement(global_z, tr_s),
        dynamic_range=float(global_z.max() - global_z.min()),
        instantaneous_peaks=[
            {
                "prediction_s": round(peak_idx * tr_s, 3),
                "stimulus_s": round(max(peak_idx * tr_s - HEMODYNAMIC_LAG_S, 0.0), 3),
                "z": round(float(peak_frame_z[i]), 4),
                "region": region_map.names[i],
            }
            for i in top_order
        ],
        region_strategy=region_map.strategy,
    )
