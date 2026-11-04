"""Muse Spark interpretation layer.

Division of labour (this is the anti-hallucination design)
---------------------------------------------------------
* **numpy** computes every number: peaks, dead zones, laterality, region means.
* **Muse Spark** only *interprets* those numbers under a strict JSON schema, and
  must cite the evidence it used and state confidence + caveats.
* The prompt forbids inventing timings, percentages or anatomical facts, and
  says TRIBE output is a model *prediction* for an average subject -- not
  measured brain data and not a claim about persuasion.

If the LLM is unavailable, callers fall back to
:func:`deterministic_summary`, so the API never returns a broken report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from .analysis import BrainDiagnostics
from .config import N_KEY_WINDOWS
from .muse import MuseAPIError, MuseSparkClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a neuroscience-literate creative analyst. You interpret predicted fMRI \
responses produced by Meta's TRIBE v2 encoding model for advertising videos.

Hard rules:
1. Use ONLY the numbers given to you. Never invent timings, percentages, brain \
regions, or studies. If a number is absent, say it is absent.
2. TRIBE v2 outputs are MODEL PREDICTIONS for an "average" subject on the \
fsaverage5 cortical mesh. They are not measurements from real viewers and they \
do not establish that anyone will buy anything. Never state persuasion, emotion, \
or purchase intent as fact.
3. Time bookkeeping: TRIBE predictions lag the stimulus by 5 seconds \
(hemodynamic delay). Cite times as "stimulus second" using the stimulus_* fields.
4. Region names describe coarse geometric zones of the cortical mesh. Do not \
upgrade them into precise Brodmann areas or named gyri.
5. Confidence must track evidence quality: repeated, high-magnitude, consistent \
signals => higher confidence; single-frame or low dynamic range => low.
6. Prefer falsifiable statements and name the evidence field behind each claim.

Return compact, concrete language suitable for a creative director."""


def build_evidence_payload(
    diagnostics: BrainDiagnostics,
    *,
    asset: dict[str, Any] | None = None,
    creative_context: str | None = None,
    max_windows: int = N_KEY_WINDOWS,
) -> dict[str, Any]:
    """Assemble the *only* facts the LLM is allowed to reason about."""

    diag = diagnostics.to_dict()
    payload: dict[str, Any] = {
        "asset": asset or {},
        "prediction_metadata": {
            "model": "TRIBE v2 (facebook/tribev2)",
            "mesh": "fsaverage5 (10242 vertices per hemisphere)",
            "subject": "average subject",
            "hemodynamic_lag_s": 5.0,
            "tr_s": diag["tr_s"],
            "n_time_steps": diag["n_trs"],
            "region_strategy": diag["region_strategy"],
        },
        "global_dynamic_range_z": diag["dynamic_range"],
        "laterality_ratio": {
            "value": diag["laterality_ratio"],
            "definition": "(left - right) / (|left| + |right|); "
            "0 = symmetric, +1 = fully left, -1 = fully right",
        },
        "hemisphere_mean": diag["hemisphere_mean"],
        "peak_windows": diag["peak_windows"][:max_windows],
        "quiet_windows": diag["quiet_windows"][:max_windows],
        "low_engagement_segments": diag["low_engagement_segments"],
        "instantaneous_peaks": diag["instantaneous_peaks"],
        "region_mean_z": dict(
            sorted(diag["region_mean_z"].items(), key=lambda kv: kv[1], reverse=True)
        ),
    }
    if creative_context:
        payload["creative_context"] = creative_context
    return payload


def build_prompt(evidence: dict[str, Any]) -> str:
    return (
        "Here is the complete evidence set for one advertisement, computed "
        "deterministically from TRIBE v2 predictions. Interpret it.\n\n"
        "Requirements:\n"
        "- overall_verdict: 'strong' only if peak windows show high dynamic range "
        "AND few/late low-engagement segments; 'weak' if low-engagement segments "
        "dominate the middle; otherwise 'mixed'.\n"
        "- moments: describe the top peak windows (at most 5) using their "
        "stimulus_start_s / stimulus_end_s and the regions listed.\n"
        "- weak_moments: use low_engagement_segments and quiet_windows.\n"
        "- rewrite_suggestions: concrete, edit-level changes anchored to times.\n"
        "- caveats: always include that this is a model prediction for an "
        "average subject, not a measurement of real viewers.\n\n"
        f"EVIDENCE JSON:\n{json.dumps(evidence, indent=2)}"
    )


@dataclass
class Interpretation:
    """LLM verdict plus provenance, or a deterministic fallback."""

    verdict: dict[str, Any]
    provenance: dict[str, Any]
    degraded: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "provenance": self.provenance,
            "degraded": self.degraded,
            "error": self.error,
        }


FALLBACK_CAVEATS = (
    "Deterministic fallback: Muse Spark was unavailable, so this verdict was "
    "derived only from the numeric diagnostics.",
    "TRIBE v2 predicts responses for an average subject; it is not a "
    "measurement of real viewers.",
    "Region names are coarse geometric zones of the fsaverage5 mesh, not an "
    "anatomical parcellation.",
)


def deterministic_summary(diagnostics: BrainDiagnostics, reason: str | None = None) -> Interpretation:
    """Rule-based verdict used when the LLM cannot be called.

    Deliberately conservative: it reports numbers and a coarse label, and never
    attempts creative advice.
    """

    diag = diagnostics.to_dict()
    dead = diag["low_engagement_segments"]
    total_s = (diag["n_trs"] - 1) * diag["tr_s"] if diag["n_trs"] > 1 else diag["tr_s"]
    dead_s = sum(seg["duration_s"] for seg in dead)
    dead_fraction = dead_s / total_s if total_s > 0 else 0.0

    if dead_fraction > 0.5:
        label = "weak"
    elif dead_fraction < 0.15 and diag["dynamic_range"] > 1.5:
        label = "strong"
    else:
        label = "mixed"

    moments = [
        {
            "stimulus_time": f"{w['stimulus_start_s']}s-{w['stimulus_end_s']}s",
            "evidence": f"peak window score z={w['score']}",
            "interpretation": "Highest predicted response window in this clip.",
            "confidence": 0.5,
        }
        for w in diag["peak_windows"][:3]
    ]
    weak_moments = [
        {
            "stimulus_time": f"{seg['stimulus_start_s']}s",
            "evidence": (
                f"response stayed in the bottom quartile for {seg['duration_s']}s"
            ),
            "risk": "Sustained flat predicted response; consider recutting.",
        }
        for seg in dead[:3]
    ]

    verdict = {
        "overall_verdict": label,
        "verdict_confidence": 0.3,
        "summary": (
            f"Predicted response dynamic range z={diag['dynamic_range']} with "
            f"{len(dead)} low-engagement segment(s) covering "
            f"{dead_s:.1f}s of {total_s:.1f}s "
            f"({dead_fraction * 100:.0f}%)."
        ),
        "lateralisation_note": (
            f"laterality_ratio={diag['laterality_ratio']} "
            "(0 is symmetric; sign indicates hemisphere dominance)."
        ),
        "moments": moments,
        "weak_moments": weak_moments,
        "rewrite_suggestions": [],
        "next_experiment": (
            "Re-run with Muse Spark enabled to get creative guidance, and "
            "A/B test the recut against this version."
        ),
        "caveats": list(FALLBACK_CAVEATS),
    }
    return Interpretation(
        verdict=verdict,
        provenance={"model": "deterministic_rules_v1", "reasoning_effort": None},
        degraded=True,
        error=reason,
    )


def interpret_with_muse(
    diagnostics: BrainDiagnostics,
    client: MuseSparkClient,
    *,
    asset: dict[str, Any] | None = None,
    creative_context: str | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int = 12000,
) -> Interpretation:
    """Call Muse Spark for a schema-constrained verdict.

    Any failure degrades gracefully to :func:`deterministic_summary` so a
    transient Model API problem never breaks the endpoint.
    """

    from .schemas import VERDICT_SCHEMA

    evidence = build_evidence_payload(
        diagnostics, asset=asset, creative_context=creative_context
    )
    try:
        result = client.complete_json(
            build_prompt(evidence),
            VERDICT_SCHEMA,
            schema_name="ad_brain_verdict",
            system=SYSTEM_PROMPT,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
    except (MuseAPIError, ValueError) as exc:
        logger.warning("Muse Spark interpretation failed, using fallback: %s", exc)
        return deterministic_summary(diagnostics, reason=str(exc))

    return Interpretation(
        verdict=result.data,
        provenance={
            "model": result.model,
            "reasoning_effort": result.reasoning_effort,
            "usage": result.usage,
            "latency_s": round(result.latency_s, 3),
            "attempts": result.attempts,
            "endpoint": "POST /v1/chat/completions",
            "evidence_sha256": _evidence_digest(evidence),
        },
    )


def _evidence_digest(evidence: dict[str, Any]) -> str:
    """Stable hash of the evidence the LLM saw, for audit/replay."""

    import hashlib

    blob = json.dumps(evidence, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
