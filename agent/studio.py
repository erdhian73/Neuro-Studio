"""Studio payload: one shape shared by the cache script and the live API.

`frontend/public/cache/*.json` is generated offline by
``scripts/generate_sample_cache.py``; the deployed ``/jobs`` and ``/studio``
endpoints must emit the *same* structure or the UI would need two code paths.
Rather than copying ~150 lines, this module owns the canonical builder and the
script delegates to it.

Why the helpers live here and not in the script: only ``agent/`` is copied into
the Modal image (``add_local_dir("agent", ...)``), so importing from
``scripts/`` inside a deployed container would fail at import time.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Sequence

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
HEMODYNAMIC_LAG_S = 5.0

# UI label -> coarse fsaverage5 zones from agent/roi.py's ZONE_NAMES.
#
# These sets are deliberately DISJOINT. The coarse mesh zones are geometric
# bands, not a parcellation: `posterolateral` alone covers 75% of the 64x160
# grid, and `medial_anterior` / `ventrolateral_anterior` /
# `dorsolateral_anterior` sample ZERO grid cells (only 5 of the 8 zones
# survive downsampling). Overlapping sets therefore produced both duplicated
# bars (Auditory == Language/Broca) and a "Visual" bar covering 96% of cortex.
# Each zone is assigned to exactly one network below; a network whose zone has
# no grid cells is reported as unavailable rather than silently borrowing one.
#
# `Language / Broca` needs ventrolateral_anterior, which has no grid cells, so
# it is honestly unavailable from brain_grid alone -- use
# diagnostics.region_mean_z for it instead.
NETWORK_ZONES: dict[str, tuple[str, ...]] = {
    "Visual V1-V4": ("posterolateral",),
    "Auditory": ("inferior_lateral",),
    "Language / Broca": ("ventrolateral_anterior",),
    "Dorsal Attention": ("dorsolateral",),
    "Default Mode": ("medial_posterior",),
}

# Zones the mesh model computes but that no network above owns.
UNASSIGNED_ZONES: tuple[str, ...] = (
    "medial_central",
    "medial_anterior",
    "dorsolateral_anterior",
)

STUDIO_NOTE = (
    "TRIBE v2 predictions for an average subject, not measurements from real "
    "viewers. Per-second network percentages are a 0-100 normalisation of "
    "TRIBE grid responses grouped by coarse geometric zones."
)


def zones_for_network(network: str, region_names: Sequence[str]) -> list[int]:
    """Indices in ``region_names`` (e.g. ``left_posterolateral``) for a network."""

    suffixes = NETWORK_ZONES[network]
    picked: list[int] = []
    for index, name in enumerate(region_names):
        stem = str(name)
        if "_" in stem and stem.split("_", 1)[0] in {"left", "right"}:
            stem = stem.split("_", 1)[1]
        if stem in suffixes:
            picked.append(index)
    return picked


def normalise(values: Sequence[float]) -> list[float]:
    """Map a series to 0-100 relative to its own range.

    A flat series has no relative structure, so it becomes a constant 50
    instead of dividing by zero.
    """

    if not values:
        return []
    low, high = min(values), max(values)
    span = high - low
    if span <= 1e-9:
        return [50.0 for _ in values]
    return [round(100.0 * (value - low) / span, 1) for value in values]


def fmt_mmss(seconds: float) -> str:
    """Match the UI's ``00:07`` timestamp format."""

    total = max(int(round(seconds)), 0)
    return f"{total // 60:02d}:{total % 60:02d}"


def build_fmri_curves(report: dict[str, Any], duration_s: float | None) -> dict[str, Any]:
    """Per-second cortical curves over the 20,484-vertex basis.

    One TR is one second (``diagnostics.tr_s == 1.0``), so ``second`` is simply
    the row index. Storing all 20,484 floats per second would be ~80 kB/s of
    JSON, so each second carries the 20,484-vertex mean plus the hemisphere
    means, which is the same signal the UI plots.
    """

    diagnostics = report.get("diagnostics") or {}
    predictions = report.get("predictions") or {}
    tr_s = float(diagnostics.get("tr_s") or predictions.get("tr_s") or 1.0)
    series = [float(value) for value in (diagnostics.get("global_timeseries") or [])]
    hemispheres = diagnostics.get("hemisphere_mean") or {}

    curves: list[dict[str, Any]] = []
    for index, value in enumerate(series):
        second = round(index * tr_s, 3)
        curves.append(
            {
                "second": second,
                "timestamp": fmt_mmss(second),
                "global_z": round(value, 4),
                # Per-TR left/right power is not reported upstream; only the
                # clip-wide split is available, and it is labelled as such.
                "hemisphere_split": {
                    key: round(float(val), 4)
                    for key, val in hemispheres.items()
                    if isinstance(val, (int, float))
                },
            }
        )

    return {
        "n_vertices": int(predictions.get("n_vertices") or 20484),
        "summary": "mean over the 20,484-vertex fsaverage5 prediction",
        "mesh": predictions.get("mesh") or "fsaverage5",
        "subject": predictions.get("subject") or "average",
        "tr_s": tr_s,
        "hemodynamic_lag_s": float(
            predictions.get("hemodynamic_lag_s") or HEMODYNAMIC_LAG_S
        ),
        "duration_s": duration_s,
        "per_second": curves,
    }



def build_network_telemetry(
    report: dict[str, Any], n_frames: int
) -> dict[str, list[float]]:
    """Per-second 0-100 percentages for the five telemetry networks.

    ``brain_grid`` frames are a *downsampled* 64x160 cortical strip (10,240
    sampled vertices), so a grid cell is not a vertex index and cell means are
    shifted by the per-clip z-scoring. Indexing frames by region index would
    therefore be meaningless.

    Instead we reconstruct the exact vertex grouping the model used: build the
    same canonical vertex order, recompute which grid cell samples which
    vertex (``agent.brain_grid.canonical_vertices`` + ``_cell_picks``), assign
    every vertex a zone with ``agent.roi._zone_for_coordinates``, and average
    only the cells that sample that network's zones. This needs nilearn +
    nibabel (already required by ``agent.brain_grid``); when they are absent we
    return ``{}`` and the caller records that telemetry is unavailable rather
    than inventing a series.
    """

    grid = report.get("brain_grid") or {}
    frames = grid.get("frames") or []
    diagnostics = report.get("diagnostics") or {}
    region_names = [str(name) for name in (diagnostics.get("region_mean_z") or {})]

    if not frames or not region_names:
        return {}

    try:
        cells = _network_cell_groups(region_names)
    except Exception as exc:  # missing nilearn/nibabel or unexpected mesh
        logger.warning(
            "cannot compute network telemetry (%s: %s)", type(exc).__name__, exc
        )
        return {}

    raw: dict[str, list[float]] = {}
    for network, picked in cells.items():
        if not picked:
            logger.info("no grid cells matched network %r; omitting it", network)
            continue
        series: list[float] = []
        for position in range(min(n_frames, len(frames))):
            frame = frames[position]
            total = 0.0
            count = 0
            for cell in picked:
                if cell < len(frame):
                    total += float(frame[cell])
                    count += 1
            series.append(total / float(count) if count else 0.0)
        raw[network] = series

    return {network: normalise(values) for network, values in raw.items()}


@lru_cache(maxsize=1)
def _cell_groups_cached() -> dict[str, tuple[int, ...]]:
    """Memoise the (expensive) mesh walk; the mapping never changes at runtime."""

    import numpy as np

    from .brain_grid import _cell_picks, canonical_vertices
    from .config import BRAIN_GRID_HEIGHT, BRAIN_GRID_WIDTH
    from .roi import ZONE_NAMES, _fsaverage5_pial_coordinates, _zone_for_coordinates

    left_idx, right_idx = canonical_vertices()
    half_w = BRAIN_GRID_WIDTH // 2
    rows = BRAIN_GRID_HEIGHT

    coords = _fsaverage5_pial_coordinates()
    groups: dict[str, list[int]] = {network: [] for network in NETWORK_ZONES}
    for hemi, offset, order in (
        ("left", 0, left_idx),
        ("right", half_w, right_idx),
    ):
        # picks[j] is the index into the sorted hemisphere that cell j samples.
        picks = _cell_picks(rows, half_w, order.size)
        zone_of_cell = _zone_for_coordinates(
            np.asarray(coords[hemi], dtype=np.float64)
        )[order][picks]
        for row in range(rows):
            for col in range(half_w):
                zone_id = int(zone_of_cell[row * half_w + col])
                if not 0 <= zone_id < len(ZONE_NAMES):
                    continue
                zone_name = ZONE_NAMES[zone_id]
                flat = row * BRAIN_GRID_WIDTH + offset + col
                for network, suffixes in NETWORK_ZONES.items():
                    if zone_name in suffixes:
                        groups[network].append(flat)
    return {network: tuple(cells) for network, cells in groups.items()}


def network_cells(network: str) -> tuple[int, ...]:
    """Flat ``frame`` indices this network samples (empty if unsampled)."""

    return _cell_groups_cached().get(network, ())


# Backwards-compatible private alias: scripts/generate_sample_cache.py and its
# tests were written against this name before the builder moved here.
_network_cells_cached = network_cells


def _network_cell_groups(region_names: Sequence[str]) -> dict[str, list[int]]:
    """Flat ``frame`` indices per network, using the model's own vertex order.

    ``region_names`` is only used to confirm the report actually carries region
    data; the cell mapping itself is static geometry.
    """

    if not region_names:
        return {}
    return {network: list(cells) for network, cells in _cell_groups_cached().items()}


def build_studio_payload(
    report: dict[str, Any],
    *,
    sample: str,
    video: str,
    duration_s: float | None,
    image_pair: dict[str, Any] | None = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Wrap a backend report in the exact shapes the studio reads.

    Single source of truth for both ``scripts/generate_sample_cache.py`` (offline
    cache) and the deployed ``/jobs`` + ``/studio`` endpoints (live), so the UI
    never needs two code paths.
    """

    diagnostics = report.get("diagnostics") or {}
    predictions = report.get("predictions") or {}
    interpretation = report.get("interpretation") or {}
    verdict = interpretation.get("verdict") or {}
    n_trs = int(predictions.get("n_trs") or diagnostics.get("n_trs") or 0)

    curves = build_fmri_curves(report, duration_s)
    telemetry = build_network_telemetry(report, n_trs)

    region_names = [str(name) for name in (diagnostics.get("region_mean_z") or {})]
    network_coverage = (
        {
            network: {
                "src_zone_count": len(zones_for_network(network, region_names)),
                "has_series": bool(telemetry.get(network)),
                "seconds": len(telemetry.get(network) or []),
            }
            for network in NETWORK_ZONES
        }
        if region_names
        else {}
    )

    # Diagnostic text per timestamp. `stimulus_time` is already lag-corrected
    # upstream, so these labels line up with what the viewer sees.
    diagnostics_by_timestamp: list[dict[str, Any]] = []
    for moment in verdict.get("moments") or []:
        diagnostics_by_timestamp.append(
            {
                "timestamp": str(moment.get("stimulus_time") or ""),
                "kind": "moment",
                "evidence": str(moment.get("evidence") or ""),
                "interpretation": str(moment.get("interpretation") or ""),
                "confidence": moment.get("confidence"),
            }
        )
    for weak in verdict.get("weak_moments") or []:
        diagnostics_by_timestamp.append(
            {
                "timestamp": str(weak.get("stimulus_time") or ""),
                "kind": "weak_moment",
                "evidence": str(weak.get("evidence") or ""),
                "interpretation": str(weak.get("risk") or ""),
                "confidence": None,
            }
        )
    for window in diagnostics.get("peak_windows") or []:
        diagnostics_by_timestamp.append(
            {
                "timestamp": fmt_mmss(float(window.get("stimulus_start_s") or 0.0)),
                "kind": "peak_window",
                "evidence": f"score {window.get('score')}",
                "interpretation": ", ".join(
                    f"{region.get('region')} {region.get('score')}"
                    for region in (window.get("top_regions") or [])
                ),
                "confidence": None,
            }
        )

    # Share of the 64x160 grid each network's zones cover. `posterolateral`
    # alone is ~75% of the grid, so "Visual V1-V4" is broad posterior cortex,
    # not a V1-V4 label -- record it so nobody over-reads the bar.
    total_cells = int((report.get("brain_grid") or {}).get("width") or 160) * int(
        (report.get("brain_grid") or {}).get("height") or 64
    )
    grid_share = {
        network: round(len(network_cells(network)) / total_cells, 4)
        for network in NETWORK_ZONES
    }

    telemetry_block: dict[str, Any] = {
        "n_seconds": n_trs,
        "networks": telemetry,
        "coverage": network_coverage,
        "derivation": (
            "0-100 normalisation of TRIBE brain_grid cells grouped by coarse "
            "fsaverage5 zones; approximate network mapping, see zone_map"
        ),
        "zone_map": {key: list(val) for key, val in NETWORK_ZONES.items()},
        "unassigned_zones": list(UNASSIGNED_ZONES),
        "grid_cell_share": grid_share,
    }
    # Networks whose zones have no grid cells get a clip-level mean from
    # diagnostics.region_mean_z, clearly labelled as static (not per-second).
    static_means: dict[str, float] = {}
    for network in NETWORK_ZONES:
        if telemetry.get(network):
            continue
        picked = zones_for_network(network, region_names)
        values = [
            float((diagnostics.get("region_mean_z") or {}).get(region_names[i], 0.0))
            for i in picked
        ]
        if values:
            static_means[network] = round(sum(values) / len(values), 4)
    if static_means:
        telemetry_block["static_region_mean_z"] = static_means
        telemetry_block["static_note"] = (
            "Networks listed here have no cells in the 64x160 grid (their "
            "coarse zones are unsampled), so only a clip-level mean exists. "
            "The UI must not animate these as a per-second series."
        )
    if not telemetry:
        telemetry_block["unavailable_reason"] = (
            "brain_grid/region data missing, or nilearn/nibabel unavailable in "
            "this environment; rerun where agent.brain_grid can import"
        )

    studio: dict[str, Any] = {
        "sample": sample,
        "video": video,
        "schema_version": SCHEMA_VERSION,
        "synthetic": synthetic,
        "fmri_curves": curves,
        "networks": telemetry_block,
        "diagnostics_by_timestamp": diagnostics_by_timestamp,
        "rewritten_voiceover_hooks": [
            {
                "target_time": str(item.get("target_time") or ""),
                "hook": str(item.get("change") or ""),
                "rationale": str(item.get("rationale") or ""),
            }
            for item in (verdict.get("rewrite_suggestions") or [])
        ],
        "verdict": {
            "overall": verdict.get("overall_verdict"),
            "confidence": verdict.get("verdict_confidence"),
            "summary": verdict.get("summary"),
            "lateralisation_note": verdict.get("lateralisation_note"),
        },
        "muse_image": {
            "model": "muse-image-1.0",
            "available": image_pair is not None,
            "before_after": image_pair,
            "note": (
                "Scene Replacement images are generated separately; the "
                "deployed /analyze route does not call Muse Image."
            ),
        },
        # Dead zones, already lag-corrected to *stimulus* time upstream, so the
        # timeline can shade exactly the span the viewer sees.
        "low_engagement_segments": list(
            diagnostics.get("low_engagement_segments") or []
        ),
        "note": STUDIO_NOTE,
    }

    return {
        "sample": sample,
        "video": video,
        "duration_s": duration_s,
        "studio": studio,
        # Full backend report kept verbatim so nothing is lost or rewritten.
        "report": report,
    }


# ---------------------------------------------------------------------------
# Private aliases
#
# ``scripts/generate_sample_cache.py`` and ``tests/test_generate_sample_cache.py``
# were written against these names before this module moved out of the script.
# Re-exporting them keeps both working without a second implementation.
# ---------------------------------------------------------------------------
_zones_for_network = zones_for_network
_network_cell_groups = _network_cell_groups
_normalise = normalise
_fmt_mmss = fmt_mmss
