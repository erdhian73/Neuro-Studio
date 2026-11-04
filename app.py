"""TRIBE v2 x Muse: agentic fMRI pretesting for ad videos, on Modal.

Deployment
----------
    modal deploy app.py          # creates the persistent app + HTTPS endpoint
    modal run app.py::selftest   # end-to-end smoke test on a real GPU

Why this shape
--------------
* ``modal.Image`` pins the whole cloud environment: ffmpeg (video/audio
  decoding), uv (TRIBE shells out to ``uvx whisperx`` for word timings), the
  TRIBE source tree, and the exact torch/numpy versions its pyproject demands.
* ``modal.Volume`` keeps ~30 GB of encoder weights (V-JEPA2 ViT-g, w2v-BERT 2.0,
  DINOv2-large, Llama-3.2-3B, WhisperX large-v3, best.ckpt) so cold starts skip
  the download. Weights are pulled ONCE at image build time.
* ``@app.cls`` + ``@modal.enter()`` loads TRIBE into the L40S exactly once per
  container; every request after that reuses the resident model.
* ``@modal.fastapi_endpoint`` gives the HTTPS URL; a ``modal.Dict`` gives
  stateful job tracking so long analyses never hit an HTTP or function timeout.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import modal
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from agent.config import (
    CHECKPOINT_BYTES,
    MODEL_API_KEY_ENV,
    TRIBE_REPO_ID,
    TRIBE_SRC_DIR,
)
from agent.pipeline import STAGES


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

APP_NAME = os.environ.get("TRIBE_APP_NAME", "tribe-ad-pretest")

# Persistent volumes: weights/caches (read-mostly) and per-job media workspace.
CACHE_DIR = "/cache"          # HF_HOME -- ~30 GB of encoder weights + TRIBE ckpt
WORKSPACE_DIR = "/workspace"  # uploaded videos, extracted audio, report artifacts

cache_vol = modal.Volume.from_name("tribe-ad-pretest-cache", create_if_missing=True)
workspace_vol = modal.Volume.from_name("tribe-ad-pretest-workspace", create_if_missing=True)

# Job state lives in a Dict so the HTTP layer never blocks on GPU work.
job_state = modal.Dict.from_name("tribe-ad-pretest-jobs", create_if_missing=True)

# ---------------------------------------------------------------------------
# Job event log
#
# ``/jobs`` streams *real* pipeline progress. Events are append-only JSON
# strings in a single Modal Dict key (``events:<job_id>``) rather than one key
# per event, so a reader always sees a consistent prefix and a job tidies up
# with one delete. Dict values are capped at 1 MiB, so the payload per event is
# deliberately small: full artifacts never go in the log, only the pieces the
# UI draws (see ``_compact``).
# ---------------------------------------------------------------------------
MAX_EVENT_CHARS = 900_000  # keep the whole log under the 1 MiB Dict limit
JOB_TTL_S = 6 * 60 * 60    # logs are pruned opportunistically, not by Modal


class JobLog:
    """Append-only event log for one job, shared by worker and web container."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    @property
    def key(self) -> str:
        return f"events:{self.job_id}"

    def meta_key(self) -> str:
        return f"meta:{self.job_id}"

    def read(self) -> list[dict]:
        raw = job_state.get(self.key) or ""
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def append(self, event: dict) -> dict:
        """Add one event; returns the stored event (with seq/ts filled in)."""

        event = dict(event)
        existing = self.read()
        event.setdefault("seq", len(existing))
        event.setdefault("ts", round(time.time(), 3))
        existing.append(event)

        encoded = "\n".join(json.dumps(item) for item in existing)
        if len(encoded) > MAX_EVENT_CHARS:
            # Drop the oldest non-terminal events (keep `start` and `done`/`error`).
            keep_head = [item for item in existing[:1]]
            tail = existing[1:]
            while tail and len(
                "\n".join(json.dumps(item) for item in keep_head + tail)
            ) > MAX_EVENT_CHARS:
                tail.pop(0)
            existing = keep_head + tail
            encoded = "\n".join(json.dumps(item) for item in existing)
            event["truncated"] = True

        job_state[self.key] = encoded
        return event

    def finish(self, status: str, **extra: Any) -> None:
        meta = {"job_id": self.job_id, "status": status, "finished_at": time.time()}
        meta.update(extra)
        job_state[self.meta_key()] = meta

    def meta(self) -> dict:
        return job_state.get(self.meta_key()) or {"job_id": self.job_id, "status": "pending"}


def _compact(stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Shrink a stage payload to what the UI actually draws.

    The 160x64 brain grid is ~4 KB per TR and the raw ``region_mean_z`` map is
    large; neither is streamed (the client fetches the final report instead).
    Keeping events small is what lets the whole log live in one Dict value.
    """

    if stage == "diagnostics":
        diag = dict(payload.get("diagnostics") or {})
        # Keep the scalars + windows the UI renders; drop bulky per-region maps.
        keep = (
            "n_trs",
            "tr_s",
            "dynamic_range",
            "laterality_ratio",
            "peak_windows",
            "low_engagement_segments",
            "instantaneous_peaks",
        )
        return {
            "diagnostics": {key: diag[key] for key in keep if key in diag},
            "elapsed_s": payload.get("elapsed_s"),
        }

    if stage == "brain_grid":
        grid = payload.get("brain_grid") or {}
        return {
            "brain_grid": {
                "width": grid.get("width"),
                "height": grid.get("height"),
                "n_frames": grid.get("n_frames"),
                "seconds_per_frame": grid.get("seconds_per_frame"),
            },
            "elapsed_s": payload.get("elapsed_s"),
        }

    if stage == "interpret":
        interp = payload.get("interpretation") or {}
        verdict = interp.get("verdict") or {}
        provenance = interp.get("provenance") or {}
        return {
            "interpretation": {
                "degraded": interp.get("degraded"),
                "error": interp.get("error"),
                "provenance": {
                    "model": provenance.get("model"),
                    "latency_s": provenance.get("latency_s"),
                },
                "verdict": {
                    "overall_verdict": verdict.get("overall_verdict"),
                    "verdict_confidence": verdict.get("verdict_confidence"),
                    "summary": verdict.get("summary"),
                },
            },
            "elapsed_s": payload.get("elapsed_s"),
        }

    return payload


def build_studio(
    report: dict[str, Any], *, sample: str, video: str, duration_s: float | None
) -> dict:
    """Studio block for a live report, identical to the offline cache shape.

    Both paths import the same builder from ``agent/studio.py`` (which lives in
    the image; ``scripts/`` does not), so the UI never needs two code paths.
    """

    from agent.studio import build_studio_payload

    return build_studio_payload(
        report, sample=sample, video=video, duration_s=duration_s
    )["studio"]


# Encoders TRIBE resolves through transformers/huggingface_hub. best.ckpt lives
# in facebook/tribev2 (~709 MB, verified with a HEAD request on the Hub).
PRELOAD_REPOS = (
    "facebook/tribev2",
    "meta-llama/Llama-3.2-3B",           # text encoder (gated; needs HF_TOKEN)
    "facebook/vjepa2-vitg-fpc64-256",    # video encoder (largest)
    "facebook/w2v-bert-2.0",             # audio encoder
    "facebook/dinov2-large",             # image encoder
)

PRELOAD_SCRIPT = f"""
import os, sys
from huggingface_hub import snapshot_download

repos = {PRELOAD_REPOS!r}
hf_token = os.environ.get("HF_TOKEN")
failures = []
for repo in repos:
    try:
        path = snapshot_download(repo_id=repo, token=hf_token, max_workers=8)
        print(f"[preload] {{repo}} -> {{path}}", flush=True)
    except Exception as exc:  # a single failure must not kill the build
        failures.append((repo, repr(exc)))
        print(f"[preload] WARNING {{repo}} failed: {{exc!r}}", flush=True)
print(f"[preload] done: {{len(failures)}} failure(s)", flush=True)
sys.exit(0)
"""


def _build_tribe_image() -> modal.Image:
    """Cloud environment: OS libs, Python deps, TRIBE source, warm weights."""

    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install(
            "ffmpeg",           # moviepy audio extraction + whisperx decoding
            "git",              # install TRIBE from source
            "libsndfile1",      # soundfile, a TRIBE dependency
            "curl",             # uv installer / health probes
            "build-essential",  # some wheels still need a compiler
        )
        # uv provides `uvx`, which TRIBE's ExtractWordsFromAudio shells out to
        # for WhisperX word timings.
        .run_commands(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
            "ln -sf /root/.local/bin/uvx /usr/local/bin/uvx",
            "uv --version",
        )
        # torch is pinned by TRIBE (>=2.5.1,<2.7); install CUDA wheels first so
        # pip resolves the GPU build rather than the CPU one.
        .pip_install(
            "torch==2.6.0",
            "torchvision==0.21.0",
            index_url="https://download.pytorch.org/whl/cu124",
        )
        .pip_install(
            "numpy==2.2.6",  # TRIBE pins numpy==2.2.6
            "modal==1.6.3",
            "fastapi[standard]",
            "requests",
            "huggingface_hub[hf_transfer]",
        )
        # TRIBE itself plus its plotting extra (brings nilearn for fsaverage5).
        .run_commands(
            "git clone --depth 1 https://github.com/facebookresearch/tribev2.git "
            f"{TRIBE_SRC_DIR}",
            f"uv pip install --system -e '{TRIBE_SRC_DIR}[plotting]'",
        )
        # Cache env must be set before any download happens.
        .env(
            {
                "HF_HOME": CACHE_DIR,
                "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_VERBOSITY": "warning",
                "PYTHONUNBUFFERED": "1",
                "TRIBE_CACHE_DIR": f"{CACHE_DIR}/tribe_features",
            }
        )
        .add_local_dir("agent", remote_path="/root/agent", copy=True)
        .add_local_file("app.py", remote_path="/root/app.py", copy=True)
    )



image = _build_tribe_image()

app = modal.App(APP_NAME, image=image)

# GPU worker limits. 24 h is Modal's documented maximum function timeout; we use
# it so a long ad (or a cold WhisperX install) can never be killed mid-analysis.
MAX_TIMEOUT_S = 24 * 60 * 60
# Module-scope start of *this container*, so /warmup can report how long the
# worker has been up (module scope runs once per container, before @modal.enter).
_CONTAINER_STARTED_AT = time.time()
GPU_TYPE = os.environ.get("TRIBE_GPU", "L40S")
WORKER_CPU = 6      # cores
WORKER_MEMORY_GIB = 32
# Keep the warm container (30 GB of weights already in VRAM) parked for 7 min
# between calls: long enough to absorb a burst of creatives in one campaign
# batch, short enough that an idle GPU stops billing quickly.
SCALEDOWN_WINDOW_S = 7 * 60


@app.cls(
    gpu=GPU_TYPE,
    cpu=WORKER_CPU,
    memory=WORKER_MEMORY_GIB * 1024,  # Modal expects MiB
    timeout=MAX_TIMEOUT_S,
    # Loading 30 GB of weights from the Volume takes longer than the default.
    startup_timeout=MAX_TIMEOUT_S,
    scaledown_window=SCALEDOWN_WINDOW_S,
    volumes={CACHE_DIR: cache_vol, WORKSPACE_DIR: workspace_vol},
    # Created once via `python scripts/create_secrets.py` (exact keys, no
    # dashboard clicking). Modal resolves this name at deploy/run time.
    secrets=[modal.Secret.from_name("muse-model-api")],
    retries=modal.Retries(initial_delay=2.0, max_retries=2),
)
class TribeAdAnalyzer:
    """Stateful GPU worker: TRIBE v2 is loaded once per container."""

    @modal.enter()
    def load_model(self) -> None:
        """One-time container startup: import TRIBE and load the checkpoint.

        Runs inside ``@modal.enter`` so subsequent requests only pay for
        inference. Heavy imports stay here (not at module scope) so the client
        can import this file locally without torch installed.
        """

        started = time.perf_counter()
        import torch

        from tribev2 import TribeModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_folder = os.environ.get("TRIBE_CACHE_DIR", f"{CACHE_DIR}/tribe_features")
        os.makedirs(cache_folder, exist_ok=True)
        os.makedirs(WORKSPACE_DIR, exist_ok=True)

        logger.info("Loading TRIBE v2 (%s) on %s", TRIBE_REPO_ID, self.device)
        self.model = TribeModel.from_pretrained(
            TRIBE_REPO_ID,
            cache_folder=cache_folder,
            device=self.device,
        )
        self.model_loaded_s = round(time.perf_counter() - started, 2)
        logger.info("TRIBE v2 ready in %.1fs", self.model_loaded_s)

        # Persist any freshly downloaded weights so the next container is warm.
        try:
            cache_vol.commit()
        except Exception as exc:  # pragma: no cover - commit is best effort
            logger.warning("Volume commit failed during startup: %s", exc)



    def _predict_path(self, video_path: str):
        """Run TRIBE on a video file and return the raw vertex predictions.

        TRIBE shells out to ``uvx whisperx`` for word-level timings. TRIBE's
        subprocess only filters ``MPLBACKEND`` from the environment, so we must
        hand it a ``PATH`` that contains ``uv``/``uvx`` -- otherwise the
        transcription step fails with ``FileNotFoundError: uvx``.
        """
        import numpy as np

        uv_bin = "/root/.local/bin"
        current = os.environ.get("PATH", "")
        if uv_bin not in current.split(":"):
            os.environ["PATH"] = f"{uv_bin}:{current}"

        # TRIBE shells out to `uvx whisperx` in an ephemeral uvenv that does not
        # include hf_transfer, and hf_hub raises (not falls back) when
        # HF_HUB_ENABLE_HF_TRANSFER=1 is set. We set it for our own big weight
        # downloads, so disable it just for this call. Verified on GPU:
        #   ValueError: Fast download using 'hf_transfer' is enabled ...
        #   but 'hf_transfer' package is not available in your environment.
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

        started = time.perf_counter()
        events = self.model.get_events_dataframe(video_path=video_path)
        n_events = int(len(events))
        logger.info("Built event table with %s rows in %.1fs", n_events, time.perf_counter() - started)

        preds, segments = self.model.predict(events=events)
        preds = np.asarray(preds)
        logger.info(
            "TRIBE prediction shape=%s segments=%s in %.1fs",
            preds.shape,
            len(segments),
            time.perf_counter() - started,
        )
        self.last_timings = {
            "events_build_s": round(time.perf_counter() - started, 3),
            "n_events": n_events,
        }
        return preds

    @modal.method()
    def fetch_sample_clip(self, clip_seconds: int = 20) -> dict:
        """Download + trim the public demo clip **inside the GPU container**.

        Needed because ``analyze_video`` takes a *remote* path: a file written on
        the local client's ``/tmp`` is invisible to the worker.
        """

        import subprocess

        destination = Path("/tmp/tribe_selftest.mp4")
        _download_sample(destination)

        info = {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "clip_seconds": clip_seconds,
            "trimmed": False,
        }
        if clip_seconds > 0:
            trimmed = Path(f"/tmp/tribe_selftest_{clip_seconds}s.mp4")
            result = subprocess.run(
                ["ffmpeg", "-y", "-t", str(clip_seconds), "-i", str(destination),
                 "-c", "copy", str(trimmed)],
                capture_output=True,
            )
            if result.returncode == 0 and trimmed.exists() and trimmed.stat().st_size > 0:
                info["path"] = str(trimmed)
                info["bytes"] = trimmed.stat().st_size
                info["trimmed"] = True
            else:
                logger.warning(
                    "ffmpeg trim failed (rc=%s); using the full clip: %s",
                    result.returncode,
                    result.stderr[-300:].decode("utf-8", "replace"),
                )
        return info

    @modal.method()
    def analyze_video(
        self,
        video_path: str,
        asset_id: str,
        label: str | None = None,
        creative_context: str | None = None,
        use_llm: bool = True,
        job_id: str | None = None,
    ) -> dict:
        """Full single-creative analysis (runs on the GPU worker).

        When ``job_id`` is given, every stage is appended to the ``jobs`` dict
        so ``GET /jobs/{id}/events`` can stream genuine (not simulated)
        progress to the browser while the GPU works.
        """


        from agent.config import DEFAULT_TR_S, env_float
        from agent.muse import MuseSparkClient
        from agent.pipeline import Stimulus, analyse_stimulus

        # This method can run on a DIFFERENT container from the one that wrote
        # the upload. Modal volumes are network filesystems with
        # last-write-wins caching: even after ``workspace_vol.commit()`` a
        # worker may briefly see a stale snapshot and raise
        # ``FileNotFoundError``. Reload to drop any stale cache, then poll.
        # (Observed once for sample_3 and once for sample_4 during back-to-back
        # uploads; sample_2 only ever succeeded by landing on a warm container.)
        import time

        for attempt in range(30):
            try:
                workspace_vol.reload()
            except Exception:  # pragma: no cover - reload is best-effort
                pass
            if os.path.exists(video_path):
                break
            if attempt == 0:
                logger.warning("waiting for upload to become visible: %s", video_path)
            time.sleep(1.0)
        else:
            raise FileNotFoundError(
                f"video_path never became visible on this container: {video_path}"
            )

        tr_s = env_float("TRIBE_TR_S", DEFAULT_TR_S)
        muse = MuseSparkClient() if use_llm else None

        stimulus = Stimulus(
            asset_id=asset_id,
            path=video_path,
            label=label,
            creative_context=creative_context,
        )

        # Live progress: each stage is appended to the shared job log as soon as
        # it genuinely completes, so `/jobs/{id}/events` streams real numbers.
        log = JobLog(job_id) if job_id else None

        def on_progress(stage: str, payload: dict) -> None:
            if log is None:
                return
            try:
                log.append(
                    {"job_id": job_id, "stage": stage, "data": _compact(stage, payload)}
                )
            except Exception as exc:  # never fail an analysis over telemetry
                logger.warning("job log append failed (%s): %s", stage, exc)

        report = analyse_stimulus(
            stimulus,
            self._predict_path,
            tr_s=tr_s,
            muse=muse,
            use_llm=use_llm,
            progress=on_progress,
        )
        report.setdefault("timings", {}).update(getattr(self, "last_timings", {}))
        report["worker"] = {
            "gpu": GPU_TYPE,
            "model_load_s": getattr(self, "model_loaded_s", None),
            "device": getattr(self, "device", None),
        }

        if log is not None:
            # The final report is too large for a Dict value, so the studio
            # block (the only part the UI renders) is what gets cached. Keep
            # the JSON well under the 1 MiB limit by dropping the before/after
            # image payload, which the UI loads separately.
            try:
                duration = float(
                    (report.get("predictions") or {}).get("n_trs") or 0.0
                ) * float((report.get("predictions") or {}).get("tr_s") or 1.0)
                studio = build_studio(
                    report,
                    sample=asset_id,
                    video=video_path,
                    duration_s=round(duration, 3) or None,
                )
                studio["muse_image"] = {
                    key: value
                    for key, value in studio.get("muse_image", {}).items()
                    if key != "before_after"
                }
                studio.pop("brain_grid", None)
                encoded = json.dumps(studio, default=str)
                if len(encoded) < 800_000:
                    log.finish("done", studio=encoded)
                else:  # pragma: no cover - guard against oversized studios
                    logger.warning(
                        "studio payload too large for job %s (%d chars)", job_id, len(encoded)
                    )
                    log.finish("done")
            except Exception as exc:  # never fail an analysis over telemetry
                logger.warning("could not cache studio block for %s: %s", job_id, exc)
                log.finish("done")

        return report

    @modal.method()
    def health(self) -> dict:
        """Report readiness without running inference."""

        import torch

        return {
            "ok": True,
            "app": APP_NAME,
            "tribe_loaded": hasattr(self, "model"),
            "model_load_s": getattr(self, "model_loaded_s", None),
            "device": getattr(self, "device", None),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gib": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
            if torch.cuda.is_available()
            else None,
            "uvx_on_path": any(
                os.path.exists(os.path.join(p, "uvx"))
                for p in os.environ.get("PATH", "").split(":")
                if p
            ),
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "model_api_key_present": bool(os.environ.get(MODEL_API_KEY_ENV)),
        }


    # ------------------------------------------------------------------
    # Web endpoints (FastAPI, HTTPS). Bodies are JSON; videos are base64 so a
    # single request is self-contained (Modal allows request bodies up to 4 GiB).
    # ------------------------------------------------------------------

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def analyze(self, request: Request) -> JSONResponse:
        """Synchronous analysis of one video.

        Returns the full report (diagnostics + brain animation grid + verdict).
        Use ``/jobs`` for multi-creative campaigns so no HTTP request has to
        wait for several GPU passes.
        """

        from agent.config import env_flag

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "request body must be JSON"}, status_code=400)

        video_b64 = body.get("video_base64")
        if not video_b64:
            return JSONResponse({"error": "video_base64 is required"}, status_code=400)

        asset_id = str(body.get("asset_id") or uuid.uuid4().hex[:10])
        suffix = str(body.get("filename") or "ad.mp4")
        if "." not in Path(suffix).name:
            suffix = f"{suffix}.mp4"

        try:
            raw = base64.b64decode(video_b64, validate=True)
        except Exception:
            return JSONResponse({"error": "video_base64 is not valid base64"}, status_code=400)
        if not raw:
            return JSONResponse({"error": "video_base64 decoded to 0 bytes"}, status_code=400)

        job_dir = Path(WORKSPACE_DIR) / asset_id
        job_dir.mkdir(parents=True, exist_ok=True)
        video_path = job_dir / Path(suffix).name
        video_path.write_bytes(raw)

        # The write above happened in THIS container's view of the shared
        # volume. ``analyze_video`` is a separate @modal.method that Modal may
        # schedule on a *different* container, which only sees committed data.
        # Without this commit the worker intermittently raised
        # ``FileNotFoundError: video_path does not exist: /workspace/<id>/...``
        # (reproduced on sample_3 while sample_2 happened to land on the same
        # container and succeed). Commit before the cross-container call.
        workspace_vol.commit()

        try:
            report = await self.analyze_video.remote.aio(
                str(video_path),
                asset_id,
                body.get("label"),
                body.get("creative_context"),
                env_flag("TRIBE_USE_LLM", True),
            )
        except Exception as exc:  # surface a clean error instead of a 500 trace
            logger.exception("analysis failed for %s", asset_id)
            return JSONResponse(
                {"error": f"analysis failed: {type(exc).__name__}: {exc}", "asset_id": asset_id},
                status_code=500,
            )

        report["asset"]["bytes"] = len(raw)
        return JSONResponse(report)

    @modal.fastapi_endpoint(method="GET", docs=True)
    def healthz(self) -> dict:
        """Cheap readiness probe (does not load the model)."""

        return {
            "ok": True,
            "app": APP_NAME,
            "gpu": GPU_TYPE,
            "cpu": WORKER_CPU,
            "memory_gib": WORKER_MEMORY_GIB,
            "max_timeout_s": MAX_TIMEOUT_S,
            "scaledown_window_s": SCALEDOWN_WINDOW_S,
            "endpoints": [
                "POST /analyze",
                "GET /warmup",
                "GET /healthz",
                "POST /campaign",
                "POST /jobs",
                "GET /jobs/{job_id}/events",
                "GET /jobs/{job_id}",
            ],
        }


    @modal.fastapi_endpoint(method="GET", docs=True)
    def warmup(self) -> JSONResponse:
        """Pay the cold start before anyone uploads (idempotent, cheap afterwards).

        ``GET /healthz`` answers from the *web* container without touching the
        GPU, so it proves routing but leaves the L40S asleep: the first
        ``/analyze`` or ``/jobs`` call then pays the full cold start (image pull
        + ~30 GB of weights from the Volume, minutes), which is exactly when a
        user is waiting.

        This endpoint is scheduled *on* the GPU class, so the HTTP response
        cannot be sent until the container exists and ``@modal.enter()`` has
        loaded TRIBE into VRAM. The frontend pings it silently on page load, so
        the model is resident by the time the user picks a file. No video, no
        inference, no model API spend.

        Modal sets ``MODAL_IS_REMOTE=1`` on every container it starts, so it
        cannot distinguish a fresh container from a reused one. The browser
        therefore always sees ``warm: true`` here — that means "the GPU worker
        answered", not "this call was free". ``container_ready_s`` is how long
        the container had been up when *this* request arrived, so it grows on
        later pings; ``model_load_s`` is the one-off TRIBE load.
        """

        body = {
            "ok": True,
            # Reaching this body at all proves @modal.enter() finished.
            "warm": hasattr(self, "model"),
            "device": getattr(self, "device", None),
            # Measured per request, from the container's own module-start clock.
            "container_ready_s": round(time.time() - _CONTAINER_STARTED_AT, 2),
            "model_load_s": getattr(self, "model_loaded_s", None),
            "gpu": GPU_TYPE,
            "task_id": os.environ.get("MODAL_TASK_ID", ""),
        }
        return JSONResponse(body)

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def campaign(self, request: Request) -> JSONResponse:
        """Analyse a whole campaign concurrently and return a ranked report.

        ``asyncio.gather`` fans the creatives out to *separate* GPU containers,
        so wall-clock time stays close to a single video instead of the sum.
        """

        import asyncio

        from fastapi import Response

        from agent.config import env_flag
        from agent.muse import MuseSparkClient
        from agent.pipeline import Stimulus, analyse_stimulus, build_report_payload
        from agent.render import render_report_html

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "request body must be JSON"}, status_code=400)

        creatives = body.get("creatives") or []
        if not isinstance(creatives, list) or not creatives:
            return JSONResponse({"error": "creatives must be a non-empty list"}, status_code=400)
        if len(creatives) > 12:
            return JSONResponse({"error": "at most 12 creatives per campaign"}, status_code=400)

        campaign_id = str(body.get("campaign_id") or uuid.uuid4().hex[:12])
        root = Path(WORKSPACE_DIR) / "campaigns" / campaign_id
        root.mkdir(parents=True, exist_ok=True)

        jobs = []
        for index, item in enumerate(creatives):
            asset_id = str(item.get("asset_id") or f"creative-{index + 1}")
            filename = Path(str(item.get("filename") or f"creative-{index + 1}.mp4")).name
            try:
                raw = base64.b64decode(item.get("video_base64") or "", validate=True)
            except Exception:
                return JSONResponse(
                    {"error": f"creatives[{index}].video_base64 is not valid base64"},
                    status_code=400,
                )
            if not raw:
                return JSONResponse(
                    {"error": f"creatives[{index}].video_base64 is empty"}, status_code=400
                )
            path = root / filename
            path.write_bytes(raw)
            jobs.append(
                {
                    "asset_id": asset_id,
                    "path": str(path),
                    "label": item.get("label"),
                    "creative_context": item.get("creative_context"),
                }
            )

        use_llm = env_flag("TRIBE_USE_LLM", bool(body.get("use_llm", True)))
        tr_s = float(body.get("tr_s") or 1.0)
        muse = MuseSparkClient() if use_llm else None

        def run_one(job: dict, tr_s: float) -> dict:
            return analyse_stimulus(
                Stimulus(job["asset_id"], job["path"], job["label"], job["creative_context"]),
                self._predict_path,
                tr_s=tr_s,
                muse=muse,
                use_llm=use_llm,
            )

        loop = asyncio.get_running_loop()
        reports = await asyncio.gather(
            *[loop.run_in_executor(None, run_one, job, tr_s) for job in jobs],
            return_exceptions=True,
        )

        ok_reports = [r for r in reports if isinstance(r, dict)]
        errors = [
            {"asset_id": jobs[i]["asset_id"], "error": f"{type(r).__name__}: {r}"}
            for i, r in enumerate(reports)
            if not isinstance(r, dict)
        ]
        if not ok_reports:
            return JSONResponse(
                {"error": "all creatives failed", "details": errors}, status_code=500
            )

        payload = build_report_payload(
            {
                "id": campaign_id,
                "name": body.get("campaign_name") or campaign_id,
                "n_requested": len(jobs),
                "n_analysed": len(ok_reports),
            },
            ok_reports,
            include_brain_grid=False,
        )
        if errors:
            payload["errors"] = errors

        # Persist JSON + HTML artifacts next to the uploaded media.
        (root / "report.json").write_text(json.dumps(payload, indent=2, default=str))
        (root / "report.html").write_text(render_report_html(payload))
        try:
            workspace_vol.commit()
        except Exception as exc:  # pragma: no cover
            logger.warning("workspace commit failed: %s", exc)

        if str(request.query_params.get("format", "json")).lower() == "html":
            return Response(render_report_html(payload), media_type="text/html")
        payload["artifacts"] = {
            "campaign_id": campaign_id,
            "workspace_dir": str(root),
            "files": ["report.json", "report.html"],
        }
        return JSONResponse(payload)


    # ------------------------------------------------------------------
    # Live jobs: start an analysis without blocking, then stream real
    # progress. POST /jobs returns immediately with a job_id; the GPU work
    # continues in a Modal function call, and /jobs/{id}/events replays the
    # stage log as Server-Sent Events so the browser updates as work lands.
    # ------------------------------------------------------------------

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def jobs(self, request: Request) -> JSONResponse:
        """Queue one creative and return a ``job_id`` immediately (HTTP 202).

        Body: ``asset_id``, ``filename``, ``video_base64``, optional ``label``,
        ``creative_context``, ``use_llm``. Same payload as ``/analyze``; the
        difference is that this returns in milliseconds and the caller watches
        ``GET /jobs/{id}/events`` for progress.
        """

        from agent.config import env_flag

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "request body must be JSON"}, status_code=400)

        video_b64 = body.get("video_base64")
        if not video_b64:
            return JSONResponse({"error": "video_base64 is required"}, status_code=400)
        try:
            raw = base64.b64decode(video_b64, validate=True)
        except Exception:
            return JSONResponse({"error": "video_base64 is not valid base64"}, status_code=400)
        if not raw:
            return JSONResponse({"error": "video_base64 decoded to 0 bytes"}, status_code=400)

        asset_id = str(body.get("asset_id") or uuid.uuid4().hex[:10])
        filename = Path(str(body.get("filename") or "ad.mp4")).name
        if "." not in filename:
            filename = f"{filename}.mp4"

        job_id = uuid.uuid4().hex[:16]
        job_dir = Path(WORKSPACE_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        video_path = job_dir / filename
        video_path.write_bytes(raw)
        # The GPU call may land on another container (see /analyze), so the
        # upload has to be committed before the cross-container call.
        workspace_vol.commit()

        log = JobLog(job_id)
        log.append(
            {
                "job_id": job_id,
                "stage": "queued",
                "data": {
                    "asset_id": asset_id,
                    "filename": filename,
                    "bytes": len(raw),
                    "stages": list(STAGES),
                },
            }
        )

        # Fire-and-forget: the Modal function call keeps running after this
        # response, which is what makes the HTTP request return immediately.
        call = self.analyze_video.spawn(
            str(video_path),
            asset_id,
            body.get("label"),
            body.get("creative_context"),
            env_flag("TRIBE_USE_LLM", True),
            job_id,
        )
        job_state[log.meta_key()] = {
            "job_id": job_id,
            "status": "running",
            "call_id": call.object_id,
            "queued_at": time.time(),
        }

        return JSONResponse(
            {"job_id": job_id, "events_url": f"/jobs/{job_id}/events"},
            status_code=202,
        )

    @modal.fastapi_endpoint(method="GET", docs=True)
    def job_events(self, job_id: str) -> Response:
        """Server-Sent Events stream of real pipeline progress.

        Replays everything already logged, then polls the shared Dict until the
        job reaches a terminal state. Each frame is ``event: <stage>`` with the
        event JSON as ``data``, so the browser can use a plain ``EventSource``.
        """

        import asyncio

        from fastapi.responses import StreamingResponse

        log = JobLog(job_id)

        async def stream():
            sent = 0
            idle_s = 0.0
            while True:
                events = log.read()
                for event in events[sent:]:
                    yield (
                        f"event: {event.get('stage', 'message')}\n"
                        f"data: {json.dumps(event, default=str)}\n\n"
                    )
                sent = len(events)

                meta = log.meta()
                status = str(meta.get("status") or "running")
                if status in {"done", "error"}:
                    tail = {k: v for k, v in meta.items() if k != "status"}
                    yield (
                        "event: end\n"
                        f"data: {json.dumps({'status': status, **tail}, default=str)}\n\n"
                    )
                    return
                # Give up on abandoned streams instead of holding the
                # connection open forever (the job itself is unaffected).
                if sent == 0 and idle_s > 60:
                    yield 'event: error\ndata: {"error": "no events for 60s"}\n\n'
                    return
                await asyncio.sleep(0.75)
                idle_s += 0.75

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @modal.fastapi_endpoint(method="GET", docs=True)
    def job_status(self, job_id: str) -> JSONResponse:
        """Poll one job: state, seen stages, and the report once finished."""

        log = JobLog(job_id)
        meta = log.meta()
        events = log.read()

        body: dict = {
            "job_id": job_id,
            "status": meta.get("status", "pending"),
            "events": len(events),
            "stages": [event.get("stage") for event in events],
        }
        report = job_state.get(f"report:{job_id}")
        if report:
            body["report"] = json.loads(report)
        if meta.get("studio"):
            body["studio"] = json.loads(meta["studio"])
        if meta.get("error"):
            body["error"] = meta["error"]
        return JSONResponse(body)


# ---------------------------------------------------------------------------
# Local entrypoints (run with `modal run app.py::<name>`)
# ---------------------------------------------------------------------------

SAMPLE_VIDEO_URL = "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4"


def _download_sample(destination: Path) -> Path:
    """Fetch TRIBE's own public demo clip (used by their notebook)."""

    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(SAMPLE_VIDEO_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    return destination


@app.local_entrypoint()
def selftest(clip_seconds: int = 20, use_llm: bool = True) -> None:
    """GPU smoke test: health + real TRIBE inference + Muse interpretation.

    Run: ``modal run app.py::selftest``
    """

    import numpy as np

    analyzer = TribeAdAnalyzer()
    print("health:", json.dumps(analyzer.health.remote(), indent=2))

    # The clip must exist on the *worker* filesystem, so it is fetched there.
    sample = analyzer.fetch_sample_clip.remote(clip_seconds)
    print(
        f"sample clip on worker: {sample['path']} "
        f"({sample['bytes'] / 1e6:.1f} MB, trimmed={sample['trimmed']})"
    )

    report = analyzer.analyze_video.remote(
        sample["path"], "selftest", "Sintel trailer", use_llm=use_llm
    )

    diag = report["diagnostics"]
    verdict = report["interpretation"]["verdict"]
    print("\n=== TRIBE results ===")
    print("prediction shape:", report["predictions"])
    print("dynamic range z:", diag["dynamic_range"], "| laterality:", diag["laterality_ratio"])
    print("peak windows:", [(w["stimulus_start_s"], w["stimulus_end_s"]) for w in diag["peak_windows"]])
    print("flat segments:", [(s["prediction_start_s"], s["duration_s"]) for s in diag["low_engagement_segments"]])
    print("brain grid:", report["brain_grid"]["width"], "x", report["brain_grid"]["height"],
          "frames:", report["brain_grid"]["n_frames"])
    print("\n=== Muse verdict ===")
    print("verdict:", verdict["overall_verdict"], "| confidence:", verdict["verdict_confidence"])
    print("summary:", verdict["summary"])
    print("degraded:", report["interpretation"]["degraded"])
    print("timings:", json.dumps(report.get("timings", {}), indent=2))

    frames = np.array(report["brain_grid"]["frames"])
    assert frames.ndim == 2 and frames.shape[0] == report["brain_grid"]["n_frames"]
    print("\nselftest OK")


