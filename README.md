# tribe-ad-pretest

Pre-launch brain-response ranking for ad creatives. Upload up to N videos; get
back a ranked report with per-second predicted cortical activity, the moments
that peaked, the moments that went flat, and an LLM reading of those numbers
that **cites the exact figures it used**.

Built on [TRIBE v2](https://github.com/facebookresearch/tribev2) (Meta's fMRI
encoding model) for prediction and the Meta Model API (`muse-spark-1.3`,
`muse-image-1.0`) for interpretation. Runs on Modal with an L40S.

> **Read this before you use it for a decision.** TRIBE predicts *fMRI signal
> for one synthetic average viewer* watching your video. It is not a panel, not
> a survey, and not a sales forecast. See [Honest limits](#honest-limits).

---

## Why the numbers and the prose are separated

An LLM asked "did this ad perform well on the brain scan?" will invent a
plausible answer. So the LLM never computes anything:

```
video ─► TRIBE v2 ─► (n_trs, 20484) float array
                          │
                          ├─► numpy: ROI means, z-scores, peaks, flat segments,
                          │          laterality, dynamic range, ranking  ← ALL math
                          │
                          └─► brain grid 160×64 frames (lossy, for display)

numpy results ─► JSON evidence payload (+ SHA-256 digest)
                          │
                          └─► Muse Spark, json_schema strict:true
                                  │  may only cite numbers already in the payload;
                                  │  must give confidence + caveats
                                  ▼
                              interpretation
                          (on any failure → deterministic_summary, degraded: true)
```

Consequences you can rely on:

- Every number in the report is reproducible from the `.npy` predictions.
- The verdict keys are fixed by a JSON schema with `additionalProperties: false`,
  so the model cannot add fields or drop required ones.
- If Muse is down, slow, or returns unusable JSON, the report still renders with
  a templated summary and `interpretation.degraded = true`.
- Each interpretation carries an `evidence_digest` (SHA-256 of the evidence
  payload), so a stored report can be re-checked later.

## Quickstart

```bash
# 0. Local deps (only needed to run the client + tests, not the app)
uv venv && uv pip install modal 'fastapi[standard]' requests pytest numpy

# 1. Create the Modal secret. Keys must be exactly these two names.
export MODEL_API_KEY='LLM_...'      # Meta Model API key
export HF_TOKEN='hf_...'            # needs meta-llama/Llama-3.2-3B access
python scripts/create_secrets.py    # idempotent

# 2. Deploy
modal deploy app.py

# 3. Prove it works on the GPU (real clip, real inference, real Muse call)
modal run app.py::selftest --clip-seconds 8

# 4. Smoke-test the deployed HTTPS surface
python scripts/smoke_test.py https://<workspace>--tribe-ad-pretest-tribeadanalyzer-healthz.modal.run

# 5. Unit tests (no GPU, no network)
pytest tests -q
```

`scripts/create_secrets.py` is the only supported way to make the secret —
`modal.Secret.from_name(...)` is just a reference and **cannot** create or
override one; it resolves at deploy time and raises `NotFoundError` otherwise.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness + resolved config (GPU, cpu, memory, `scaledown_window_s`). Does not load the model. |
| `GET` | `/warmup` | Pays the cold start: scheduled **on the GPU class**, so the response only lands after `@modal.enter()` has loaded TRIBE into VRAM. No video, no inference, no model API spend. The frontend pings it silently on page load. |
| `POST` | `/analyze` | One creative. Body: `asset_id`, `filename`, `video_base64`, optional `label`, `creative_context`, `use_llm`. Returns the full report. |
| `POST` | `/campaign` | 2–8 creatives, ranked. Add `?format=html` for the shareable report. |
| `POST` | `/jobs` | Queue one creative, returns `202 {job_id, events_url}` in milliseconds. Same body as `/analyze`. |
| `GET` | `/jobs/{job_id}/events` | **Server-Sent Events** stream of real pipeline stages (`queued → predict → diagnostics → brain_grid → interpret → done`), then `end`. Query param, not a path segment, because each Modal endpoint is its own hostname. |
| `GET` | `/jobs/{job_id}` | Poll one job: `status`, stage list, and the finished `studio` block. |

`POST /analyze` returns the report body directly, so a 40 MB JSON is possible
with the brain grid included (see below).

### Why `/jobs` exists

`/analyze` is synchronous, and a 25 s clip takes ~95 s of GPU time (a cold
container adds ~3 min). A browser cannot show honest progress for that with a
single request, so `/jobs` splits it in two:

```
POST /jobs ──► 202 {job_id}            (returns in ms; GPU work continues)
                    │
                    └─► GET /jobs/{id}/events   (SSE)
                             queued → predict → diagnostics → brain_grid
                                    → interpret → done → end
```

Every event carries the **real** intermediate result for its stage (TR count,
diagnostic scalars, verdict) — nothing is a simulated progress bar. The events
are appended to a `modal.Dict` by the worker as each stage actually completes
(`agent/pipeline.py::analyse_stimulus(progress=...)`), so a late subscriber
replays the same log instead of missing it.

A note on the hostnames: Modal publishes **one hostname per function**, shaped

```
<workspace>--<app>-<class>-<slug>.<region>.modal.run
```

so the frontend derives each endpoint from `NEXT_PUBLIC_TRIBE_API` by inserting
the slug before the region (`frontend/src/lib/api.ts::endpointUrl`).

### Why the page pings `/warmup`

`/healthz` is answered by the **web** container and deliberately never touches
the model, so it proves routing but leaves the L40S scaled to zero. The first
upload would then pay the whole cold start — image pull plus ~30 GB of weights
from the Volume, minutes — exactly while the user waits.

`page.tsx` therefore fires `pingWarmup()` once on mount (a `useEffect` with an
empty dependency list). Because `warmup` is a method on `TribeAdAnalyzer`, the
request cannot be answered until a GPU container exists and `@modal.enter()` has
finished, so the model is resident by the time a file is picked.

The ping is fire-and-forget by design:

- nothing is rendered and nothing is awaited, so the UI is unaffected;
- every failure (offline, DNS, HTTP error) is swallowed — a ping must never put
  an error banner on a page that otherwise works from cache;
- the client abandons the socket after 10 min, but the container keeps booting,
  which is the only side effect relied upon;
- no video is sent and no inference runs, so it costs a container wake-up and
  nothing else.

Responses report `warm`, `device`, `model_load_s`, `container_ready_s` and
`task_id`. Note that Modal sets `MODAL_IS_REMOTE=1` on **every** container it
starts, so `warm: true` here means "the GPU worker answered the request" — not
"this call was free". The honest cold-start numbers are the `timings` block in a
report.


### Campaign request

```json
{
  "campaign": {"name": "Q4 Launch", "brand": "Acme", "objective": "awareness"},
  "creatives": [
    {"asset_id": "a", "filename": "a.mp4", "video_base64": "...", "label": "Cut A"},
    {"asset_id": "b", "filename": "b.mp4", "video_base64": "...", "label": "Cut B"}
  ]
}
```

## What the report contains

| Field | Meaning |
|---|---|
| `predictions` | Shape, TR, mesh, subject, hemodynamic lag. The array itself is cached as `.npy`. |
| `regions` | Per-ROI z-scored time series + summary stats. |
| `diagnostics.peak_windows` | Top non-overlapping windows, with **stimulus time** (5 s lag subtracted) so you can seek the video. |
| `diagnostics.low_engagement_segments` | Contiguous stretches in the clip's own bottom quantile. |
| `diagnostics.laterality_ratio` | Left vs right hemisphere balance (0.5 = balanced). |
| `diagnostics.dynamic_range` | Max z − min z of the global signal: how much the ad moves the model *relative to itself*. |
| `brain_grid` | `160×64` per-TR frames, z×10 rounded, for drawing a heatmap. |
| `interpretation` | Muse verdict + confidence + caveats, or `degraded: true`. |
| `ranking` (campaign) | 0–100 score from dynamic range, laterality, and flat-time fraction. |

### Reading `peak_windows` correctly

The lag is real: TRIBE's output at second *t* reflects the video at *t − 5 s*.
The report exposes both, so a peak at 12 s in the predictions maps to 7 s of
actual video. Do not compare a peak window against the video without applying
that offset — the report already does it for you in `stimulus_start_s`.

## Honest limits

These are not disclaimers bolted on; they are the boundary of what the pipeline
computes.

1. **It is one synthetic subject.** TRIBE's output is an encoding-model
   prediction for a single average viewer, not a distribution over people.
   There is no per-viewer variance, so the report cannot distinguish "most
   viewers" from "a minority of viewers". Ranking two creatives compares two
   model responses, not two audiences.
2. **fMRI signal is not attention, liking, or purchase intent.** Predictions are
   BOLD-response magnitudes on `fsaverage5`. A strong visual-motion response and
   a strong emotional response both look like "large amplitude". Any semantic
   claim ("this moment lands emotionally") comes from the LLM reading the
   numbers and is labelled interpretation, not measurement.
3. **Region labels are geometric unless you opt in.** By default ROIs are coarse
   fsaverage5 zones from vertex coordinates. `ROI_ATLAS=schaefer` uses the
   Schaefer 100-parcel atlas via nilearn — better, but still a parcellation of a
   predicted signal. Check which one produced a report before quoting a parcel.
4. **Downstream metrics are heuristic.** `ranking` weights are hand-picked
   trapezoid rules, not fitted to campaign outcomes. Nothing here is trained on
   whether an ad actually sold anything. Treat rank order as a discussion aid.
5. **Clip length.** TRIBE runs at TR = 1 s over the clip duration, so a 30 s ad
   yields ~30 TRs. Short clips give few samples per window; flat segments under
   3 s are not reported at all.
6. **The evidence digest proves consistency, not correctness.** It shows the
   interpretation was generated from that exact evidence payload. It does not
   validate the model.
7. **Cost and cold start are real.** Loading TRIBE is ~30 GB of weights; the
   container stays warm for 420 s (`SCALEDOWN_WINDOW_S`) after the last call.
   Working through creatives one at a time can pay repeated cold starts — use
   `/campaign` to batch them into one warm container.
8. **No authentication by default.** The FastAPI endpoints are public. Put them
   behind `requires_proxy_auth=True` or a Modal proxy token before sharing a URL
   beyond your team.
9. **`low_engagement_segments` is relative to the clip.** The threshold is the
   clip's own 25th percentile, so it means "flattest stretch *for this ad*", not
   "viewers disengaged". A uniformly strong ad can still contain a low segment.


## Verified facts appendix

Everything below was observed live during development, not inferred from docs.
Where a claim cost real debugging time, the observed failure is included.

### TRIBE v2

- Package is `tribev2`, **not on PyPI** — install from git. Requires
  Python ≥ 3.11, `torch>=2.5.1,<2.7`, `numpy==2.2.6` (pinned).
- `TribeModel.from_pretrained("facebook/tribev2", cache_folder=...)`;
  `model.get_events_dataframe(video_path=...)`;
  `preds, segments = model.predict(events=df)`.
- `preds.shape == (n_trs, 20484)`; left = `[:, :10242]`, right = `[:, 10242:]`;
  fsaverage5; 5 s hemodynamic lag.
- Encoders pulled at runtime: `meta-llama/Llama-3.2-3B` (gated → `HF_TOKEN`
  required), `facebook/dinov2-large`, `facebook/vjepa2-vitg-fpc64-256`,
  `facebook/w2v-bert-2.0`, plus whisperx `large-v3` via `uvx`.
  `best.ckpt` is 709 MB.
- **`uvx` must be on PATH.** TRIBE shells out to `uvx whisperx` for word
  timings; the image ships `/root/.local/bin` and `_predict_path` injects it.
- **`HF_HUB_ENABLE_HF_TRANSFER=1` breaks WhisperX.** The `uvx` environment does
  not contain `hf_transfer` and `huggingface_hub` *raises* instead of falling
  back: `ValueError: Fast download using 'hf_transfer' is enabled ...
  but 'hf_transfer' package is not available in your environment.` We keep it on
  for our own multi-GB weight downloads and pop it just before event building.
- Observed timings on L40S for an **8-second** clip: event build 131.9 s,
  diagnostics 0.23 s, brain grid 0.06 s, Muse 22.0 s → 154.2 s total
  (`degraded: False`, verdict `mixed`, confidence 0.45).
- `get_events_dataframe` warns about an `Index` column in the events dataframe;
  harmless, TRIBE drops it.

### Meta Model API

- Base `https://api.meta.ai/v1`, auth `Authorization: Bearer LLM_...`. The
  `LLM_*` string is the **API key**, not an endpoint id.
- Chat: `POST /chat/completions`. `response_format: {"type": "json_schema",
  "json_schema": {"strict": true, ...}}` is validated server-side.
- `reasoning_effort` ∈ {`minimal`, `low`, `medium`, `high`, `xhigh`, `max`};
  `none` → HTTP 400.
- **Reasoning tokens count against `max_tokens`.** Observed failure mode: HTTP
  200 with `message.content == null` and `finish_reason: "length"` because the
  budget was consumed by reasoning. The client raises an error naming
  `max_tokens` rather than returning empty content.
- Images: `POST /images/generations`, `size: "WxH"`, `output_format` ∈
  png|jpeg|webp, `response_format` ∈ url|b64_json, `n <= 10`, returns
  `data[].b64_json`.
- Models confirmed available: `muse-spark-1.3`, `muse-spark-1.3-contributor`,
  `muse-image-1.0`, `muse-wonder-1.0`, `sam-3.1`.

### Modal (SDK 1.6.3)

- `app.cls(cpu=, memory=, gpu=, timeout<=86400, startup_timeout=, volumes=,
  secrets=, scaledown_window=)`. `cpu=` counts **physical** cores (6 → 17 reported).
- `@modal.enter()` runs once per container — used to load TRIBE into VRAM.
- `@modal.fastapi_endpoint(method=, docs=, requires_proxy_auth=)`.
- `modal.Volume.from_name(name, create_if_missing=True)`.
- **`Secret.from_name()` cannot create or override a secret.** It resolves at
  deploy time and raises `NotFoundError` otherwise — hence
  `scripts/create_secrets.py` using
  `Secret.objects.create(..., allow_existing=True)`.
- L40S: 46 GB VRAM, ≈100 MB/s Hugging Face download.
- **`.remote()` paths are remote.** A file written to `/tmp` on the local client
  is invisible to the worker; `selftest` downloads its sample clip *inside* the
  container via `fetch_sample_clip`.
- `ast.unparse` renders a decorator as `app.cls(...)`, not `@app.cls(...)`.


## Layout

```
app.py                  Modal app: image, @app.cls GPU worker, endpoints, selftest
agent/config.py         constants, env parsing, brain-grid spec
agent/muse.py           MuseSparkClient (strict JSON) + MuseImageClient
agent/roi.py            fsaverage5 geometric zones or Schaefer atlas → RegionMap
agent/analysis.py       z-scoring, windows, peaks, flat segments, laterality
agent/brain_grid.py     (20484,) → 160×64 nearest-centroid vertex picks
agent/interpret.py      evidence payload, digest, schema-constrained call, fallback
agent/pipeline.py       per-stimulus orchestration, ranking, report payload, stage hook
agent/studio.py         the single studio-payload builder (cache + live API)
agent/render.py         standalone HTML report
agent/schemas.py        JSON schemas (verdict, campaign)
scripts/create_secrets.py   idempotent Modal secret creation
scripts/smoke_test.py       health + optional analyze against a deployed URL
scripts/make_sample_4.sh    rebuilds sample_4.mp4 (8-12s of sample_2, +contrast/+saturation)
scripts/generate_sample_cache.py  pre-computes frontend/public/cache/*.json
tests/                  unit tests — no GPU, no network
```

### The brain grid is lossy by construction

`20484 = 2 × 3 × 3 × 569`, so 20484 vertices cannot tile a rectangle exactly.
The grid takes the nearest vertex to each of 160×64 cell centroids per
hemisphere (80 columns each). It exists to be *looked at* — do the arithmetic on
the predictions array. At `decimals=1` it costs ~4 KB per TR, which is why
campaign reports drop it (`include_brain_grid=False`).

Only 5 of the 8 coarse zones survive that downsampling. Measured shares of the
10,240 cells: `posterolateral` 75.2%, `medial_posterior` 21.3%,
`dorsolateral` 1.9%, `inferior_lateral` 0.9%, `medial_central` 0.6%.
`medial_anterior`, `ventrolateral_anterior` and `dorsolateral_anterior` sample
**zero** cells.

## Neuro Studio static cache

`frontend/` reads pre-computed JSON by default, and can switch to a live GPU
run from the Inspector. Both paths return the **same** `studio` shape because
they call the same builder (`agent/studio.py::build_studio_payload`), so the UI
has one code path:

| Path | When | Badge |
|---|---|---|
| `public/cache/sample_N.json` | on load, and when a sample card is clicked | `CACHED RUN` |
| `POST /jobs` + SSE | when a file is uploaded in the Inspector | `LIVE GPU RUN` |

Build the cache with:

```bash
# real run — needs a deployed endpoint (modal deploy app.py)
export TRIBE_ENDPOINT=https://<workspace>--tribe-ad-pretest-analyze.modal.run
python scripts/generate_sample_cache.py

# offline: reuse reports you already exported
python scripts/generate_sample_cache.py --from-report out/sample_2.json

# UI placeholder only — every file is stamped "synthetic": true
python scripts/generate_sample_cache.py --fake
```

It writes one file per clip to `frontend/public/cache/`:
`sample_2.json`, `sample_3.json`, `sample_4.json`. Each contains the backend
report verbatim under `report`, plus a `studio` block with the four shapes the
UI needs:

| Key | Contents |
|---|---|
| `fmri_curves` | `n_vertices: 20484`, `tr_s: 1.0`, and `per_second[]` with `second`, `timestamp`, `global_z`, `hemisphere_split` |
| `networks` | per-second 0-100% for the 5 telemetry networks, plus `coverage`, `grid_cell_share`, `zone_map`, and `static_region_mean_z` for unsampled ones |
| `diagnostics_by_timestamp` | peaks / weak moments / peak windows with `MM:SS` labels |
| `rewritten_voiceover_hooks` | Muse Spark rewrite suggestions (`target_time`, `hook`, `rationale`) |
| `muse_image` | `before_after` (base64) — `available: false` unless `--images` |

Two honest caveats, both recorded inside the JSON:

* **`Language / Broca` has no per-second series.** Its zone
  (`ventrolateral_anterior`) samples zero grid cells, so it appears under
  `networks.static_region_mean_z` as a clip-level mean. The UI must not animate
  it as a time series.
* **`Visual V1-V4` is broad posterior cortex** (`posterolateral` = 75% of the
  grid), not a V1-V4 label. `grid_cell_share` publishes the real coverage.

Muse Image 1.0 is **not** called by the deployed `/analyze` route
(`MuseImageClient` exists in `agent/muse.py` but is not wired into
`agent/pipeline.py`). `--images` therefore makes a separate `muse-image-1.0`
call; without it the Before/After panel has no generated "after" frame.

### Status: cache generated from real inference

`frontend/public/cache/sample_{2,3,4}.json` now hold **real TRIBE v2 predictions**
(`synthetic: false`, 20,484 vertices, `tr_s: 1.0`, Muse Spark 1.3 verdicts).
They were produced by the deployed L40S app, not synthesised.

Two backend bugs had to be fixed in `app.py` to make this reproducible:

1. **Uncommitted volume write.** `/analyze` wrote the upload to `/workspace`
   and then dispatched `analyze_video` to a *different* container, but only the
   campaign path ever called `workspace_vol.commit()`. The worker therefore
   intermittently raised
   `FileNotFoundError: video_path does not exist: /workspace/<id>/<id>.mp4`.
   `sample_2` passed only by landing on the same container.
2. **Stale volume reads.** Even after committing, the worker could see a stale
   snapshot. `analyze_video` now calls `workspace_vol.reload()` and polls for
   up to 30 s before failing.

Both were intermittent, so they look like flaky infrastructure rather than bugs.

### Measured result: clip *length* drove the response, not contrast

The first version of `sample_4.mp4` was a 4 s cut (8-12 s of `sample_2`) with
+26% contrast and +59% saturation. On the real model it scored **lower** than
its source, so the cut was rebuilt at **15 s** (3-18 s of `sample_2`), keeping
the same boost. Length turned out to be the dominant factor: TRIBE runs at
`tr_s = 1.0`, so a 4 s clip yields only 5 TRs and most of the clip is consumed
by the 5 s hemodynamic lag.

| Clip | TRs | dynamic_range | top peak score |
|---|---|---|---|
| `sample_2` (source, 30 s) | 31 | 2.129 | 0.1747 |
| `sample_4`, 4 s (superseded) | 5 | 1.705 | 0.1365 |
| **`sample_4`, 15 s (current)** | **15** | **2.559** | **0.3356** |

The 15 s cut beats the source by **+92% on peak score** and the old 4 s cut by
**+146%**. Treat "more contrast = more brain response" as **unsupported**: the
contrast boost was identical in both cuts, and only the length changed the
outcome.

Window selection is measured, not guessed. Per-second `scene_score` on
`sample_2.mp4` shows the deadest stretch at ~18-28 s (`26-27s` = 0.0006,
`27-28s` = 0.0004, `29-30s` = 0.0000). Scoring every 15-20 s window, `3-18 s`
retained the most motion (mean 0.0129) with only 3 dead seconds. The finished
15 s clip has mean motion 0.0175 vs 0.0093 for the old 8-12 s window (**+88%**),
with 2 dead seconds instead of 9.

Note `global_z` is normalised per clip (mean forced to 0), so it is **not**
comparable across clips. Use `dynamic_range` and `peak_windows[].score`, which
are computed on raw predictions.

### Muse Image 1.0

`studio.muse_image.available` is now `true` and `before_after.after_base64`
holds a real 1024x1024 PNG. It is produced by the **script**, not by
`/analyze` -- `MuseImageClient` is still not called by the deployed route, so
this only appears when running with `--images` and a valid `MODEL_API_KEY`.
Muse Image has no effect on the TRIBE predictions; it only supplies the
"after" poster frame for the Before/After panel.


## Environment variables

| Var | Default | Notes |
|---|---|---|
| `MODEL_API_KEY` | — | Required. Delivered via the `muse-model-api` secret. |
| `HF_TOKEN` | — | Required. Gated Llama repo. |
| `TRIBE_GPU` | `L40S` | |
| `TRIBE_USE_LLM` | `true` | `false` → always use `deterministic_summary`. |
| `ROI_ATLAS` | `mesh` | `schaefer` for the 100-parcel atlas. |
| `TRIBE_TR_S` | `1.0` | Seconds per TR. |
| `TRIBE_APP_NAME` | `tribe-ad-pretest` | |

Frontend only (`frontend/.env.local`, never a secret — these endpoints are public):

| Var | Default | Notes |
|---|---|---|
| `NEXT_PUBLIC_TRIBE_API` | unset | Deployed endpoint prefix **including the region**, e.g. `https://<ws>--tribe-ad-pretest-tribeadanalyzer.us-east.modal.run`. Unset → the page runs entirely from `public/cache/*.json` and the badge reads `CACHED RUN`. |

## Tests

`pytest tests -q` → **57 passed** in ~6 s, no GPU or network required. Coverage
focuses on the parts that can silently lie:

- diagnostics find an injected burst and compute laterality with the right sign
- window offsets equal `prediction_start_s − 5 s` (the hemodynamic lag)
- `low_engagement` stays relative to the clip (a flat-but-strong tail is not flagged)
- brain-grid dimensions and the z×10 quantisation bounds
- vertex picks are unique/in-range and `canonical_vertices` is a permutation
- an LLM outage degrades to `deterministic_summary` while still satisfying every
  required schema key
- diagnostics are byte-identical across runs (determinism)
- ranking is monotonic and scores stay in 0–100
- Muse client: strict-schema request shape, no retry on 4xx, retry on 503,
  `content: null` + `finish_reason: length` raises a `max_tokens` error,
  image-parameter validation, b64 PNG decoding
- a static AST check that `app.py` still carries `app.cls` / `modal.enter()` /
  `modal.method()` / `fastapi_endpoint`, the documented worker limits, and
  `scaledown_window=SCALEDOWN_WINDOW_S`
- the studio contract (`tests/test_studio_contract.py`): the offline cache and
  the live API produce byte-identical `studio` blocks, every field the UI reads
  is present, network zones are disjoint, unsampled networks are reported as
  static rather than animated, and the progress hook emits
  `predict → diagnostics → brain_grid → interpret → done` in order with real
  payloads

The AST test is what caught the decorator-format assumption above.

## Deployment status

Live at `https://modal.com/apps/<workspace>/main/deployed/tribe-ad-pretest`
(app `tribe-ad-pretest`, GPU `L40S`, 6 cores, 32 GiB,
`scaledown_window_s: 420`). `scripts/smoke_test.py` against the deployed
`/healthz` returns `ok: true` with that config echoed back, and
`modal run app.py::selftest` completed a real TRIBE + Muse round trip on GPU.

