"""Strict JSON schemas handed to Muse Spark.

Constraint set mirrors what Meta Model API enforces when ``strict: true``:
root is a plain object, every object sets ``additionalProperties: false``, and
``required`` lists every property. No ``oneOf``/``allOf``, no recursion.
"""

from __future__ import annotations

from typing import Any

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "overall_verdict",
        "verdict_confidence",
        "summary",
        "lateralisation_note",
        "moments",
        "weak_moments",
        "rewrite_suggestions",
        "next_experiment",
        "caveats",
    ],
    "properties": {
        "overall_verdict": {"type": "string", "enum": ["strong", "mixed", "weak"]},
        "verdict_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "lateralisation_note": {"type": "string"},
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["stimulus_time", "evidence", "interpretation", "confidence"],
                "properties": {
                    "stimulus_time": {"type": "string"},
                    "evidence": {"type": "string"},
                    "interpretation": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "weak_moments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["stimulus_time", "evidence", "risk"],
                "properties": {
                    "stimulus_time": {"type": "string"},
                    "evidence": {"type": "string"},
                    "risk": {"type": "string"},
                },
            },
        },
        "rewrite_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target_time", "change", "rationale"],
                "properties": {
                    "target_time": {"type": "string"},
                    "change": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "next_experiment": {"type": "string"},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
}

CREATIVE_BRIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["concept", "visual_prompt", "rationale", "target_regions"],
    "properties": {
        "concept": {"type": "string"},
        "visual_prompt": {"type": "string"},
        "rationale": {"type": "string"},
        "target_regions": {"type": "array", "items": {"type": "string"}},
    },
}
