/**
 * Client for the TRIBE backend.
 *
 * Two paths, deliberately:
 *
 *  1. **Cache** — `public/cache/sample_N.json` holds a real backend report,
 *     pre-computed offline. Instant, costs nothing, and is what the demo loads
 *     so the page is never empty.
 *  2. **Live** — `POST /jobs` queues a GPU run and `GET /jobs/{id}/events`
 *     streams real stage events (Server-Sent Events). The URL comes from
 *     `NEXT_PUBLIC_TRIBE_API`, so no secret is ever shipped to the browser:
 *     the Modal endpoints are public and the API key lives only in the
 *     `muse-model-api` secret on the worker.
 *
 * Both paths return the same `Studio` shape (`agent/studio.py` is the single
 * builder), so the UI never branches on where the data came from.
 */

import type { JobEvent, JobStage, JobStatus, SampleCacheFile, Studio } from './types'

/**
 * Modal publishes *one hostname per function*:
 *
 *   https://<workspace>--<app>-<class>-jobs.<region>.modal.run
 *   https://<workspace>--<app>-<class>-healthz.<region>.modal.run
 *
 * so a single base URL cannot address both. `NEXT_PUBLIC_TRIBE_API` is the
 * prefix up to and including the class name; the endpoint slug is appended
 * from `ENDPOINT_SLUGS`.
 */
const RAW_API_PREFIX = (process.env.NEXT_PUBLIC_TRIBE_API ?? '').replace(/\/+$/, '')

type EndpointName = 'jobs' | 'jobEvents' | 'jobStatus' | 'health' | 'warmup'

/** Suffix each endpoint appends to the prefix. Keep in sync with app.py. */
const ENDPOINT_SLUGS: Record<EndpointName, string> = {
  jobs: 'jobs',
  jobEvents: 'job-events',
  jobStatus: 'job-status',
  health: 'healthz',
  warmup: 'warmup',
}

/**
 * Build the absolute URL for one endpoint.
 *
 * Accepts any of these and normalises to `<prefix>-<slug>.<region>.modal.run`:
 *
 *   ...--tribe-ad-pretest-tribeadanalyzer                     (bare prefix)
 *   ...--tribe-ad-pretest-tribeadanalyzer.us-east.modal.run   (prefix + suffix)
 *   ...--tribe-ad-pretest-tribeadanalyzer-jobs.us-east.modal.run  (one endpoint)
 */
/**
 * Build the absolute URL for one endpoint.
 *
 * Verified against the live deployment, whose hostname shape is:
 *
 *   <workspace>--<app>-<class>-<slug>.<region>.modal.run
 *
 * `NEXT_PUBLIC_TRIBE_API` is therefore the prefix **including the region**, e.g.
 *
 *   https://r7ronaldocris7--tribe-ad-pretest-tribeadanalyzer.us-east.modal.run
 *   └──────────────────── prefix ───────────────────┘└─ region ─┘└ domain ┘
 *
 * and each endpoint inserts its slug *before* the region. A pasted full
 * endpoint URL is also accepted: its slug is stripped first.
 */
export function endpointUrl(name: EndpointName): string {
  if (!RAW_API_PREFIX) return ''

  const suffix = /(\.[a-z]{2}(?:-[a-z]+)+(?:-\d+)?)?\.modal\.run$/i.exec(RAW_API_PREFIX)
  const region = suffix?.[1] ?? ''
  const head = suffix
    ? RAW_API_PREFIX.slice(0, RAW_API_PREFIX.length - suffix[0].length)
    : RAW_API_PREFIX

  // Drop a slug left over from a pasted single-endpoint URL.
  const prefix = head.replace(/-(jobs|job-events|job-status|healthz|warmup)$/i, '')
  return `${prefix}-${ENDPOINT_SLUGS[name]}${region}.modal.run`
}

export const hasLiveBackend = RAW_API_PREFIX.length > 0


/**
 * Cached reports store the video as a repo-relative path
 * (`frontend/public/videos/sample_3.mp4`, useful when generating the cache).
 * The browser needs the *web* path (`/videos/sample_3.mp4`), so rewrite it
 * here rather than trusting the file's own field.
 */
export function publicVideoPath(video: string): string {
  const match = /videos\/([^/]+)$/.exec(video)
  if (match) return `/videos/${match[1]}`
  return video.startsWith('/') ? video : `/${video}`
}

/** Cached samples staged in `public/cache/`. */
export const CACHED_SAMPLES = ['sample_3', 'sample_2', 'sample_4'] as const
export type CachedSample = (typeof CACHED_SAMPLES)[number]

/**
 * Cards are labelled by what each clip *is*, matching `assets` below and the
 * numbers in README (30 s / 25 s / 15 s). `sample_4` is a cut of `sample_2`
 * with boosted contrast, which is why it is called "AI optimized".
 */
export const SAMPLE_LABELS: Record<CachedSample, string> = {
  sample_3: 'Cinematic',
  sample_2: 'Edu speech',
  sample_4: 'AI optimized',
}

function cacheUrl(sample: CachedSample): string {
  return `/cache/${sample}.json`
}

/**
 * Load a pre-computed sample. Returns the studio block only; the raw report
 * stays on the server because the UI never renders it directly.
 */
export async function loadCachedSample(
  sample: CachedSample,
  signal?: AbortSignal,
): Promise<Studio> {
  const response = await fetch(cacheUrl(sample), { signal, cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`cache ${sample} unavailable (HTTP ${response.status})`)
  }
  const payload = (await response.json()) as SampleCacheFile
  return payload.studio
}

/** Result of queueing a live analysis. */
export interface QueuedJob {
  job_id: string
  events_url: string
}

/**
 * Silently wake the GPU worker.
 *
 * `GET /healthz` is answered by the web container and never touches the model,
 * so a page load alone leaves the L40S asleep and the first upload pays the
 * whole cold start. `GET /warmup` is a method on the GPU class, so its response
 * does not arrive until `@modal.enter()` has loaded TRIBE into VRAM.
 *
 * The ping is fire-and-forget by design: no state, no error surfaced to the
 * user, every failure swallowed. A cold container takes minutes to answer, so
 * the connection is simply abandoned after `timeoutMs` — the container finishes
 * booting regardless, which is all that matters for warmth. `keepalive` lets the
 * request outlive a fast navigation away from the page.
 */
export function pingWarmup(options: { timeoutMs?: number } = {}): Promise<void> {
  if (!hasLiveBackend || typeof window === 'undefined') return Promise.resolve()

  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), options.timeoutMs ?? 10 * 60 * 1000)

  return fetch(endpointUrl('warmup'), {
    method: 'GET',
    mode: 'cors',
    cache: 'no-store',
    keepalive: true,
    signal: controller.signal,
  })
    .then(() => undefined)
    .catch(() => undefined)
    .finally(() => window.clearTimeout(timer))
}

/**
 * Queue a live analysis. Resolves as soon as the backend has stored the upload
 * (HTTP 202) — the GPU run continues independently.
 */
export async function queueJob(options: {
  videoBase64: string
  filename: string
  assetId?: string
  label?: string
  creativeContext?: string
  signal?: AbortSignal
}): Promise<QueuedJob> {
  if (!hasLiveBackend) {
    throw new Error(
      'NEXT_PUBLIC_TRIBE_API is not set, so live analysis is unavailable',
    )
  }
  const response = await fetch(endpointUrl('jobs'), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    signal: options.signal,
    body: JSON.stringify({
      asset_id: options.assetId,
      filename: options.filename,
      label: options.label,
      creative_context: options.creativeContext,
      video_base64: options.videoBase64,
    }),
  })
  const body = (await response.json().catch(() => ({}))) as Partial<QueuedJob> & {
    error?: string
  }
  if (!response.ok || !body.job_id) {
    throw new Error(body.error ?? `POST /jobs failed (HTTP ${response.status})`)
  }
  return { job_id: body.job_id, events_url: body.events_url ?? '' }
}

/**
 * Follow a job's Server-Sent Events stream.
 *
 * `onEvent` fires per stage with the real payload; `onEnd` fires once on the
 * terminal event. Returns a disposer. `EventSource` is used rather than a
 * manual fetch loop because the backend already emits named SSE frames, and
 * the browser reconnects on transient drops by itself.
 */
export function streamJob(
  jobId: string,
  handlers: {
    onEvent?: (event: JobEvent) => void
    onEnd?: (status: JobStatus, detail: Record<string, unknown>) => void
    onError?: (message: string) => void
  },
): () => void {
  if (!hasLiveBackend) {
    handlers.onError?.('NEXT_PUBLIC_TRIBE_API is not set')
    return () => {}
  }

  // Modal's GET endpoints take query parameters, not path segments.
  const source = new EventSource(`${endpointUrl('jobEvents')}?job_id=${encodeURIComponent(jobId)}`)
  const stages: (JobStage | string)[] = [
    'queued',
    'predict',
    'diagnostics',
    'brain_grid',
    'interpret',
    'done',
  ]

  let closed = false
  const dispose = () => {
    if (!closed) {
      closed = true
      source.close()
    }
  }

  for (const stage of stages) {
    source.addEventListener(stage, event => {
      try {
        handlers.onEvent?.(JSON.parse((event as MessageEvent).data) as JobEvent)
      } catch {
        handlers.onError?.(`malformed ${stage} event`)
      }
    })
  }

  source.addEventListener('end', event => {
    try {
      const payload = JSON.parse((event as MessageEvent).data) as {
        status?: JobStatus
      }
      handlers.onEnd?.(payload.status ?? 'done', payload)
    } catch {
      handlers.onEnd?.('done', {})
    }
    dispose()
  })

  source.addEventListener('error', () => {
    // A network drop also surfaces here; only report it if we never finished.
    if (!closed) handlers.onError?.('live stream interrupted')
  })

  return dispose
}

/**
 * Read a `File` as bare base64 (no data-URL prefix), which is what the backend
 * expects in `video_base64`.
 */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error('could not read the selected file'))
    reader.onload = () => {
      const result = String(reader.result ?? '')
      const comma = result.indexOf(',')
      resolve(comma >= 0 ? result.slice(comma + 1) : result)
    }
    reader.readAsDataURL(file)
  })
}
