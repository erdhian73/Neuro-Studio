#!/usr/bin/env python3
"""Pre-compute the Neuro Studio static cache from the deployed Modal pipeline.

For every staged clip in ``frontend/public/videos/`` this script calls the
deployed TRIBE v2 + Muse Spark 1.3 pipeline and writes one JSON file to
``frontend/public/cache/``::

    sample_2.mp4 -> frontend/public/cache/sample_2.json
    sample_3.mp4 -> frontend/public/cache/sample_3.json
    sample_4.mp4 -> frontend/public/cache/sample_4.json

Why a static cache
------------------
TRIBE inference needs an L40S and minutes per clip; the studio is a static
Next.js page. Committing the results keeps the demo instant and offline.

What the cache contains
-----------------------
The backend report is preserved verbatim under ``report`` so nothing is lost,
and a ``studio`` block is added with exactly the shapes the UI binds to. Two
fields deserve emphasis because the backend does *not* ship them directly:

``fmri_curves``
    TRIBE has one TR per second (``tr_s == 1.0``), so index == second. The full
    20,484-vertex prediction is far too large for the browser (~80 kB/second),
    so each second stores ``global_z`` (the 20,484-vertex mean, identical to
    ``diagnostics.global_timeseries``), the hemisphere means, and the
    ``vertices`` summary block proving the 20,484-vertex basis.

``networks``
    ``diagnostics.region_mean_z`` is a single mean per coarse zone, *not* a
    time series, so the per-second percentages the telemetry strip animates are
    derived here from ``brain_grid`` frames (grouped by the UI's five networks)
    and normalised 0-100. This derivation is explicit and reproducible; it is
    not an inverse-variance estimate of anything the model did not compute.

Honest behaviour
----------------
* If ``--endpoint`` / ``$TRIBE_ENDPOINT`` is unset the script still validates
  the clips and then exits non-zero with instructions. It never writes a JSON
  file containing fabricated numbers.
* ``--from-report`` converts an already-exported report without touching Modal,
  which is how you regenerate the cache offline.
* ``--fake`` writes deterministic *synthetic* data for UI work. Every synthetic
  file is stamped ``"synthetic": true`` plus a warning so it can never be
  mistaken for real TRIBE output.
* Muse Image 1.0 is only called with ``--images`` because it costs money and
  the deployed ``/analyze`` route does not invoke it.

Usage
-----
    # real run (needs a deployed `modal deploy app.py`)
    export TRIBE_ENDPOINT=https://<workspace>--tribe-ad-pretest-analyze.modal.run
    python scripts/generate_sample_cache.py

    # regenerate offline from saved reports
    python scripts/generate_sample_cache.py --from-report out/sample_2.json ...

    # UI-only placeholder data
    python scripts/generate_sample_cache.py --fake
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = REPO_ROOT / "frontend" / "public" / "videos"
DEFAULT_CACHE_DIR = REPO_ROOT / "frontend" / "public" / "cache"

# Running `python scripts/generate_sample_cache.py` puts scripts/ on sys.path,
# not the repo root, so `import agent` (and `from agent.muse import ...`) would
# fail. The heavy imports are intentionally lazy (they need GPU-only deps on
# Modal), so the path fix has to live here rather than at module top.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Clips wired to the studio's 03 / 02 / 04 buttons. sample_4 is derived from
# sample_2 (4s excerpt, 8-12s, contrast + saturation boosted) -- see README.
DEFAULT_SAMPLES = ("sample_2", "sample_3", "sample_4")

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


def warn(message: str) -> None:
    """Diagnostics go to stderr so ``--print-json`` stays pipeable."""

    print(f"[warn] {message}", file=sys.stderr)


def info(message: str) -> None:
    print(f"[info] {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Media probing (optional; degrades gracefully when ffprobe is absent)
# ---------------------------------------------------------------------------


def probe_duration(path: Path, ffprobe: str | None) -> float | None:
    """Return the clip duration in seconds, or ``None`` if unknowable."""

    if not ffprobe:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        warn(f"ffprobe failed on {path.name}: {exc}")
        return None
    text = completed.stdout.strip()
    try:
        return round(float(text), 3) if text else None
    except ValueError:
        return None


def resolve_ffprobe(explicit: str | None) -> str | None:
    """Locate ffprobe: explicit flag, PATH, then ``static_ffmpeg`` if present."""

    if explicit:
        return explicit
    from shutil import which

    found = which("ffprobe")
    if found:
        return found
    try:  # pragma: no cover - depends on the local environment
        import static_ffmpeg

        static_ffmpeg.add_paths()
        return which("ffprobe")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Network telemetry derivation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Studio payload (re-exported)
#
# The implementation lives in ``agent/studio.py`` so the deployed Modal app can
# import it: only ``agent/`` is copied into the image, so a live endpoint cannot
# reach into ``scripts/``. Keeping the builder there means the offline cache and
# the live ``/studio`` endpoint emit byte-identical shapes by construction.
# ---------------------------------------------------------------------------

from agent.studio import (  # noqa: E402
    HEMODYNAMIC_LAG_S,
    NETWORK_ZONES,
    SCHEMA_VERSION,
    STUDIO_NOTE,
    UNASSIGNED_ZONES,
    _cell_groups_cached,
    _network_cells_cached,
    _zones_for_network,
    build_fmri_curves,
    build_network_telemetry,
    build_studio_payload as _build_studio_payload,
    fmt_mmss as _fmt_mmss,
    normalise as _normalise,
)


def build_studio_payload(
    report: dict[str, Any],
    *,
    sample: str,
    video: str,
    duration_s: float | None,
    image_pair: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cache-file envelope: the studio block plus generation metadata."""

    payload = _build_studio_payload(
        report,
        sample=sample,
        video=video,
        duration_s=duration_s,
        image_pair=image_pair,
    )
    return {
        "sample": payload["sample"],
        "video": payload["video"],
        "duration_s": payload["duration_s"],
        "generated_at": _utc_now(),
        "studio": payload["studio"],
        # Full backend report kept verbatim so nothing is lost or rewritten.
        "report": payload["report"],
    }


# ---------------------------------------------------------------------------
# Modal backend client
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PermanentBackendError(RuntimeError):
    """A failure that retrying cannot fix (bad auth, gated model, bad payload).

    Distinguishing this from a transient error matters because the Modal class
    sets ``retries=modal.Retries(max_retries=2)``: every retry reloads ~30 GB of
    weights onto the L40S. Retrying an HTTP 403 wastes minutes of GPU time and
    bills for it, so we stop after the first permanent failure.
    """


def _explain_backend_failure(text: str) -> str:
    """Turn a raw backend traceback into an actionable hint, when we can."""

    lowered = text.lower()
    if "gated repo" in lowered or "403 client error" in lowered:
        return (
            "The worker could not download a gated Hugging Face model.\n"
            "  TRIBE v2's text encoder is meta-llama/Llama-3.2-3B (see\n"
            "  agent/config.py ENCODER_ASSETS). Fix with either:\n"
            "    1. Accept the licence at\n"
            "       https://huggingface.co/meta-llama/Llama-3.2-3B\n"
            "       while signed in as the account that owns HF_TOKEN, then\n"
            "    2. Re-set the secret and redeploy:\n"
            "         export MODEL_API_KEY='LLM_...'\n"
            "         export HF_TOKEN='hf_...'\n"
            "         python scripts/create_secrets.py\n"
            "         modal deploy app.py\n"
            "  Verify with: modal shell -> python -c \"import os,requests;"
            "print(requests.get('https://huggingface.co/api/models/"
            "meta-llama/Llama-3.2-3B',headers={'Authorization':'Bearer '+os.environ['HF_TOKEN']}).status_code)\""
        )
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "The L40S ran out of VRAM; try a shorter clip."
    if "timeout" in lowered:
        return "The request timed out; the clip may be too long for one pass."
    return ""


def analyze_remote(
    endpoint: str,
    video: Path,
    *,
    context: str,
    timeout: float,
) -> dict[str, Any]:
    """POST the clip to the deployed Modal ``/analyze`` route.

    Mirrors ``scripts/smoke_test.py``: the whole file goes in as
    ``video_base64`` because the worker cannot see the local filesystem.
    """

    import requests

    payload = {
        "asset_id": video.stem,
        "filename": video.name,
        "label": video.stem,
        "creative_context": context,
        "video_base64": base64.b64encode(video.read_bytes()).decode("ascii"),
    }
    response = requests.post(endpoint.rstrip("/"), json=payload, timeout=timeout)
    if response.status_code != 200:
        raise PermanentBackendError(
            f"{video.name}: HTTP {response.status_code} {response.text[:400]}"
        ) from None
    body = response.json()
    if "error" in body:
        raise PermanentBackendError(f"{video.name}: backend error {body['error']}")
    return body


def generate_before_after(
    prompt: str,
    *,
    size: str,
    output_format: str,
) -> dict[str, Any]:
    """Generate the Muse Image 1.0 Scene Replacement "after" frame.

    The "before" frame comes from the clip itself, so only one image is
    generated. Images are base64 so the JSON stays self-contained.
    """

    from agent.muse import MuseImageClient

    client = MuseImageClient()
    images = client.generate(prompt, size=size, output_format=output_format, n=1)
    first = images[0]
    return {
        "prompt": prompt,
        "size": size,
        "output_format": output_format,
        "after_base64": base64.b64encode(first.png_bytes).decode("ascii"),
        "revised_prompt": first.revised_prompt,
        "note": "BEFORE frame is extracted from the source clip at the UI's timestamp.",
    }


# ---------------------------------------------------------------------------
# Synthetic (UI-only) data
# ---------------------------------------------------------------------------

ZONE_ORDER = (
    "medial_posterior",
    "medial_central",
    "medial_anterior",
    "posterolateral",
    "inferior_lateral",
    "dorsolateral",
    "ventrolateral_anterior",
    "dorsolateral_anterior",
)


def synthetic_report(sample: str, n_trs: int) -> dict[str, Any]:
    """Deterministic placeholder report so the UI can be built offline.

    Every value derives from ``sample`` so runs repeat exactly and the data is
    obviously not real inference. Callers must mark the file synthetic.
    """

    import math

    seed = sum(ord(char) for char in sample)
    series = [
        round(math.sin((index + seed) / 3.0) + 0.35 * math.cos(index / 1.7), 4)
        for index in range(n_trs)
    ]
    zones = [f"{hemi}_{name}" for hemi in ("left", "right") for name in ZONE_ORDER]
    region_mean_z = {
        name: round(0.5 * math.sin(i + seed), 4) for i, name in enumerate(zones)
    }
    frames = [
        [round(((i * 7 + t * 3) % 41 - 20) / 10.0, 1) for i in range(64 * 160)]
        for t in range(n_trs)
    ]
    return {
        "asset": {"id": sample, "label": sample, "path": f"{sample}.mp4"},
        "predictions": {
            "n_trs": n_trs,
            "n_vertices": 20484,
            "tr_s": 1.0,
            "mesh": "fsaverage5",
            "subject": "average",
            "hemodynamic_lag_s": 5.0,
        },
        "regions": {"strategy": "mesh", "names": zones},
        "diagnostics": {
            "n_trs": n_trs,
            "tr_s": 1.0,
            "dynamic_range": 2.4,
            "hemisphere_mean": {"left": 0.12, "right": -0.09},
            "laterality_ratio": 0.07,
            "region_mean_z": region_mean_z,
            "peak_windows": [
                {
                    "rank": 1,
                    "prediction_start_s": 6.0,
                    "prediction_end_s": 9.0,
                    "stimulus_start_s": 1.0,
                    "stimulus_end_s": 4.0,
                    "score": 1.9,
                    "top_regions": [{"region": "left_posterolateral", "score": 2.2}],
                }
            ],
            "quiet_windows": [],
            "low_engagement_segments": [
                {"prediction_start_s": 12.0, "duration_s": 4.0}
            ],
            "instantaneous_peaks": [],
            "global_timeseries": series,
            "region_strategy": "mesh",
        },
        "brain_grid": {
            "width": 160,
            "height": 64,
            "tr_s": 1.0,
            "quantisation": 10,
            "z_range": [-3.0, 3.0],
            "layout": "cortical_strip_left_right",
            "n_frames": n_trs,
            "frames": frames,
        },
        "interpretation": {
            "verdict": {
                "overall_verdict": "mixed",
                "verdict_confidence": 0.5,
                "summary": "SYNTHETIC placeholder verdict (not produced by Muse Spark).",
                "lateralisation_note": "Synthetic.",
                "moments": [
                    {
                        "stimulus_time": "00:07",
                        "evidence": "synthetic evidence",
                        "interpretation": "synthetic interpretation",
                        "confidence": 0.5,
                    }
                ],
                "weak_moments": [],
                "rewrite_suggestions": [
                    {
                        "target_time": "00:11",
                        "change": "Synthetic hook placeholder.",
                        "rationale": "Synthetic rationale.",
                    }
                ],
                "next_experiment": "Synthetic.",
                "caveats": ["SYNTHETIC DATA - replace with a real backend run."],
            },
            "provenance": "synthetic",
            "degraded": True,
            "error": "synthetic mode",
        },
        "timings": {},
        "worker": {"gpu": None, "model_load_s": None, "device": None},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_cache(
    payload: dict[str, Any], cache_dir: Path, sample: str, *, overwrite: bool
) -> Path:
    destination = cache_dir / f"{sample}.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass --overwrite to replace")
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def _mark_synthetic(payload: dict[str, Any]) -> dict[str, Any]:
    """Make synthetic data impossible to mistake for real inference."""

    payload["synthetic"] = True
    studio = payload.setdefault("studio", {})
    studio["synthetic"] = True
    studio["warnings"] = [
        "SYNTHETIC DATA: generated locally for UI development only.",
        "No TRIBE v2 or Muse inference was run. Do not use for any claim.",
    ]
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frontend/public/cache/*.json from the Modal pipeline."
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("TRIBE_ENDPOINT"),
        help="deployed /analyze URL (env TRIBE_ENDPOINT)",
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=list(DEFAULT_SAMPLES),
        help=f"sample stems to process (default: {' '.join(DEFAULT_SAMPLES)})",
    )
    parser.add_argument(
        "--context",
        default="Neuro-creative studio demo clip",
        help="creative_context passed to Muse Spark",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--ffprobe", default=None, help="path to ffprobe (optional)")
    parser.add_argument(
        "--from-report",
        type=Path,
        nargs="+",
        default=None,
        metavar="REPORT_JSON",
        help="convert saved backend report(s) instead of calling Modal",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="write deterministic synthetic data for UI work (clearly marked)",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="also call Muse Image 1.0 for the Before/After panel (costs money)",
    )
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--image-format", default="png")
    parser.add_argument(
        "--image-prompt",
        default=(
            "Cinematic product advertisement still, visually striking, high "
            "contrast and saturated colours, strong focal point, no text."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing JSON")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="exit 0 even if some samples fail (failed ones are skipped)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    video_dir: Path = args.video_dir
    cache_dir: Path = args.cache_dir
    ffprobe = resolve_ffprobe(args.ffprobe)

    info(f"video dir : {video_dir}")
    info(f"cache dir : {cache_dir}")
    info(f"ffprobe   : {ffprobe or 'unavailable (durations will be null)'}")

    # --- collect one report per sample ---------------------------------
    reports: dict[str, dict[str, Any]] = {}
    if args.from_report:
        for path in args.from_report:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            stem = Path(path).stem
            reports[stem] = data.get("report", data)
            info(f"loaded report for {stem} from {path}")
    elif args.fake:
        for sample in args.samples:
            video = video_dir / f"{sample}.mp4"
            duration = probe_duration(video, ffprobe) if video.exists() else None
            n_trs = max(1, int(round(duration))) if duration else 20
            reports[sample] = synthetic_report(sample, n_trs)
            info(f"synthetic report for {sample} ({n_trs} seconds)")
    else:
        if not args.endpoint:
            warn(
                "no --endpoint / TRIBE_ENDPOINT set. Deploy with "
                "`modal deploy app.py`, then copy the /analyze URL. "
                "Nothing was written."
            )
            info("alternatives: --from-report <report.json> | --fake (UI placeholder)")
            return 2
        info(f"endpoint  : {args.endpoint}")
        for sample in args.samples:
            video = video_dir / f"{sample}.mp4"
            if not video.exists():
                raise FileNotFoundError(f"missing video {video}")
            info(f"analyzing {video.name} ({video.stat().st_size / 1e6:.1f} MB) ...")
            try:
                reports[sample] = analyze_remote(
                    args.endpoint, video, context=args.context, timeout=args.timeout
                )
            except PermanentBackendError as exc:
                hint = _explain_backend_failure(str(exc))
                warn(f"{sample} failed permanently: {exc}")
                if hint:
                    warn(hint)
                # A permanent failure will fail identically for every remaining
                # clip, so stop paying for more model loads. Reports already
                # collected are still written below -- a transient failure on
                # one clip must not throw away good data from the others.
                warn("stopping: this error will not resolve on retry")
                break
            except Exception as exc:
                warn(f"{sample} failed: {exc}")
                if not args.allow_partial:
                    raise

    if not reports:
        warn("no reports produced; nothing written")
        return 1

    # --- build + write --------------------------------------------------
    written: list[str] = []
    for sample, report in reports.items():
        video = video_dir / f"{sample}.mp4"
        duration = probe_duration(video, ffprobe) if video.exists() else None
        if duration is None:
            n_trs = (report.get("predictions") or {}).get("n_trs")
            duration = float(n_trs) if n_trs else None

        image_pair = None
        if args.images and not args.fake:
            info(f"generating Muse Image 1.0 for {sample} ...")
            try:
                image_pair = generate_before_after(
                    args.image_prompt,
                    size=args.image_size,
                    output_format=args.image_format,
                )
            except Exception as exc:
                warn(f"Muse Image failed for {sample}: {exc}")

        video_ref = (
            str(video.relative_to(REPO_ROOT)) if video.exists() else f"{sample}.mp4"
        )
        payload = build_studio_payload(
            report,
            sample=sample,
            video=video_ref,
            duration_s=duration,
            image_pair=image_pair,
        )
        if args.fake:
            payload = _mark_synthetic(payload)

        destination = write_cache(payload, cache_dir, sample, overwrite=args.overwrite)
        written.append(str(destination))
        info(f"wrote {destination} ({destination.stat().st_size / 1024:.0f} kB)")

    print(json.dumps({"written": written}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileExistsError as exc:
        warn(str(exc))
        sys.exit(1)
    except FileNotFoundError as exc:
        warn(str(exc))
        sys.exit(1)
    except RuntimeError as exc:
        warn(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        warn("interrupted; no further files written")
        sys.exit(130)









