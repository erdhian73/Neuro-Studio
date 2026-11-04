"""Pipeline orchestration: stimulus -> TRIBE v2 -> diagnostics -> Muse Spark.

This module contains **no Modal code**. It receives a callable ``predict`` so
the same logic runs under unit tests with a fake predictor, and on Modal with
the real GPU-backed TRIBE model.
"""

from __future__ import annotations

import html
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .analysis import compute_diagnostics, validate_predictions
from .brain_grid import encode_brain_grid
from .config import DEFAULT_TR_S, ENCODER_ASSETS, MUSE_SPARK_MODEL, TRIBE_REPO_ID
from .interpret import Interpretation, deterministic_summary, interpret_with_muse
from .muse import MuseSparkClient
from .roi import get_region_map

logger = logging.getLogger(__name__)

PredictFn = Callable[[Path], np.ndarray]

# Emits (stage, payload) as each pipeline stage finishes. Used by the Modal
# /jobs endpoint to stream real progress to the browser instead of guessing a
# progress bar, and by tests to assert stage ordering.
ProgressFn = Callable[[str, dict[str, Any]], None]

# Ordered stage names exactly as they occur in ``analyse_stimulus``. The
# frontend maps these to labels, so treat them as a public contract.
STAGES = ("predict", "diagnostics", "brain_grid", "interpret", "done")


@dataclass
class Stimulus:
    """One creative to analyse."""

    asset_id: str
    path: Path
    label: str | None = None
    creative_context: str | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.label:
            self.label = self.asset_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.asset_id,
            "label": self.label,
            "filename": self.path.name,
            "bytes": self.path.stat().st_size if self.path.exists() else None,
        }


def analyse_stimulus(
    stimulus: Stimulus,
    predict: PredictFn,
    *,
    tr_s: float = DEFAULT_TR_S,
    muse: MuseSparkClient | None = None,
    use_llm: bool = True,
    region_strategy: str | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Run one creative end-to-end and return a JSON-ready report section.

    ``progress`` is called after each stage with ``(stage, payload)`` where the
    payload is the real intermediate result for that stage. Nothing is
    estimated: a stage only reports after its work has actually completed.
    """

    def emit(stage: str, payload: dict[str, Any]) -> None:
        if progress is not None:
            progress(stage, payload)

    started = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    preds = validate_predictions(predict(stimulus.path))
    timings["tribe_inference_s"] = round(time.perf_counter() - t0, 3)
    emit(
        "predict",
        {
            "n_trs": int(preds.shape[0]),
            "n_vertices": int(preds.shape[1]),
            "tr_s": float(tr_s),
            "duration_s": round(float(preds.shape[0]) * float(tr_s), 3),
            "elapsed_s": timings["tribe_inference_s"],
        },
    )

    t0 = time.perf_counter()
    regions = get_region_map(region_strategy)
    diagnostics = compute_diagnostics(preds, tr_s=tr_s, regions=regions)
    timings["diagnostics_s"] = round(time.perf_counter() - t0, 3)
    emit(
        "diagnostics",
        {
            "diagnostics": diagnostics.to_dict(),
            "elapsed_s": timings["diagnostics_s"],
        },
    )

    t0 = time.perf_counter()
    brain_grid = encode_brain_grid(preds, tr_s=tr_s)
    timings["brain_grid_s"] = round(time.perf_counter() - t0, 3)
    emit(
        "brain_grid",
        {
            "brain_grid": brain_grid,
            "elapsed_s": timings["brain_grid_s"],
        },
    )

    if use_llm and muse is not None:
        t0 = time.perf_counter()
        interpretation: Interpretation = interpret_with_muse(
            diagnostics,
            muse,
            asset=stimulus.as_dict(),
            creative_context=stimulus.creative_context,
        )
        timings["muse_interpretation_s"] = round(time.perf_counter() - t0, 3)
    else:
        interpretation = deterministic_summary(
            diagnostics,
            reason="LLM disabled for this request" if not use_llm else "no client",
        )
        timings["muse_interpretation_s"] = 0.0

    timings["total_s"] = round(time.perf_counter() - started, 3)
    emit(
        "interpret",
        {
            "interpretation": interpretation.to_dict(),
            "elapsed_s": timings["muse_interpretation_s"],
        },
    )

    report = {
        "asset": stimulus.as_dict(),
        "predictions": {
            "n_trs": int(preds.shape[0]),
            "n_vertices": int(preds.shape[1]),
            "tr_s": float(tr_s),
            "mesh": "fsaverage5",
            "subject": "average",
            "hemodynamic_lag_s": 5.0,
        },
        "regions": regions.describe(),
        "diagnostics": diagnostics.to_dict(),
        "brain_grid": brain_grid,
        "interpretation": interpretation.to_dict(),
        "timings": timings,
    }
    emit("done", {"timings": timings, "degraded": interpretation.degraded})
    return report


def rank_creatives(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank creatives by a transparent, deterministic composite score.

    Score (0-100) mixes three normalised signals so no single artefact can
    dominate. All inputs come from the diagnostics, never from the LLM text::

        engagement = 100 * clip(dynamic_range / 3.0, 0, 1)
        focus      = 100 * clip(|laterality_ratio|, 0, 1)
        continuity = 100 * clip(1 - dead_time_fraction, 0, 1)

        score = 0.5*engagement + 0.2*focus + 0.3*continuity
    """

    rows: list[dict[str, Any]] = []
    for report in reports:
        diag = report.get("diagnostics", {})
        n_trs = int(diag.get("n_trs") or 0)
        tr_s = float(diag.get("tr_s") or DEFAULT_TR_S)
        total_s = max(n_trs * tr_s, tr_s)
        dead_s = sum(
            float(seg.get("duration_s", 0.0))
            for seg in diag.get("low_engagement_segments", [])
        )

        engagement = 100.0 * float(np.clip(float(diag.get("dynamic_range", 0.0)) / 3.0, 0, 1))
        focus = 100.0 * float(np.clip(abs(float(diag.get("laterality_ratio", 0.0))), 0, 1))
        continuity = 100.0 * float(np.clip(1.0 - dead_s / total_s, 0, 1))
        score = 0.5 * engagement + 0.2 * focus + 0.3 * continuity

        rows.append(
            {
                "asset_id": report.get("asset", {}).get("id"),
                "label": report.get("asset", {}).get("label"),
                "score": round(score, 2),
                "components": {
                    "engagement": round(engagement, 2),
                    "focus": round(focus, 2),
                    "continuity": round(continuity, 2),
                },
                "dead_time_s": round(float(dead_s), 3),
                "duration_s": round(float(total_s), 3),
                "verdict": report.get("interpretation", {})
                .get("verdict", {})
                .get("overall_verdict"),
            }
        )

    rows.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def build_report_payload(
    campaign: dict[str, Any],
    reports: list[dict[str, Any]],
    *,
    include_brain_grid: bool = True,
) -> dict[str, Any]:
    """Assemble the campaign-level response."""

    ranking = rank_creatives(reports)
    winner = ranking[0]["asset_id"] if ranking else None
    creatives = []
    for report in reports:
        entry = dict(report)
        if not include_brain_grid:
            entry.pop("brain_grid", None)
        creatives.append(entry)

    return {
        "campaign": campaign,
        "method": {
            "brain_model": f"TRIBE v2 ({TRIBE_REPO_ID})",
            "encoders": ENCODER_ASSETS,
            "interpreter": MUSE_SPARK_MODEL,
            "scoring": "0.5*engagement + 0.2*focus + 0.3*continuity (0-100)",
            "region_strategy": reports[0].get("regions", {}).get("strategy")
            if reports
            else None,
        },
        "ranking": ranking,
        "winner": winner,
        "creatives": creatives,
        "disclaimers": [
            "All brain values are TRIBE v2 model predictions for an average "
            "subject, not measurements from real viewers.",
            "This tool does not predict sales, persuasion, or purchase intent.",
            "Use it to generate hypotheses to validate with real audience tests.",
            "Region labels are coarse fsaverage5 geometric zones unless an "
            "atlas is explicitly enabled.",
        ],
    }


def report_to_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, default=str).encode("utf-8")

