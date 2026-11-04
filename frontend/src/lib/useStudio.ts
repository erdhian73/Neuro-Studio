'use client'

/**
 * `useStudio` — one source of truth for everything the page draws.
 *
 * Loads the cached sample on mount, then (optionally) replaces it with a live
 * GPU run. Because the cache and the live path emit the same `Studio` shape,
 * the components below never branch on origin; only the badge does.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  CACHED_SAMPLES,
  fileToBase64,
  loadCachedSample,
  queueJob,
  streamJob,
  type CachedSample,
} from '@/lib/api'
import type { DataSource, Studio } from '@/lib/types'

const DEFAULT_SAMPLE: CachedSample = 'sample_3'

/** Human labels for the streaming stages, in the order the backend runs them. */
export const STAGE_LABELS: Record<string, string> = {
  queued: 'Queued',
  predict: 'TRIBE v2 inference',
  diagnostics: 'Diagnostics',
  brain_grid: 'Brain grid',
  interpret: 'Muse Spark 1.3',
  done: 'Complete',
}

export function useStudio() {
  const [sample, setSample] = useState<CachedSample>(DEFAULT_SAMPLE)
  const [studio, setStudio] = useState<Studio | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stages, setStages] = useState<string[]>([])
  const [source, setSource] = useState<DataSource>({
    mode: 'cache',
    label: 'Cached run',
    detail: 'Pre-computed TRIBE v2 report',
  })

  const disposeRef = useRef<(() => void) | null>(null)

  const loadSample = useCallback(
    async (next: CachedSample, signal?: AbortSignal) => {
      try {
        const payload = await loadCachedSample(next, signal)
        setStudio(payload)
        setError(null)
        setStages([])
        setSource({
          mode: 'cache',
          label: 'Cached run',
          detail: payload.synthetic
            ? 'SYNTHETIC placeholder — not real TRIBE output'
            : 'Pre-computed TRIBE v2 report',
        })
      } catch (cause) {
        if ((cause as Error).name === 'AbortError') return
        setError((cause as Error).message)
      } finally {
        setLoading(false)
      }
    },
    [],
  )

  // Load the selected sample. This effect only *starts* the request: every
  // state update happens in a promise callback, never synchronously in the
  // effect body (which is what `react-hooks/set-state-in-effect` guards).
  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false

    loadCachedSample(sample, controller.signal)
      .then(payload => {
        if (cancelled) return
        setStudio(payload)
        setError(null)
        setStages([])
        setSource({
          mode: 'cache',
          label: 'Cached run',
          detail: payload.synthetic
            ? 'SYNTHETIC placeholder — not real TRIBE output'
            : 'Pre-computed TRIBE v2 report',
        })
      })
      .catch((cause: Error) => {
        if (cancelled || cause.name === 'AbortError') return
        setError(cause.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [sample])

  useEffect(() => () => disposeRef.current?.(), [])

  /** Select one of the staged clips (instant, from cache). */
  const selectSample = useCallback(
    (next: CachedSample) => {
      disposeRef.current?.()
      disposeRef.current = null
      if (next === sample) {
        // Same clip re-picked: force a reload of the cached file.
        setSample(next)
        void loadCachedSample(next)
          .then(payload => {
            setStudio(payload)
            setError(null)
          })
          .catch((cause: Error) => setError(cause.message))
        return
      }
      setLoading(true)
      setSample(next)
    },
    [sample],
  )

  const runLive = useCallback(async (file: File) => {
    disposeRef.current?.()
    disposeRef.current = null

    setLoading(true)
    setError(null)
    setStages([])
    setSource({ mode: 'pending', label: 'Uploading', detail: file.name })

    try {
      const videoBase64 = await fileToBase64(file)
      const job = await queueJob({
        videoBase64,
        filename: file.name,
        label: file.name,
        assetId: `live-${Date.now().toString(36)}`,
        creativeContext: 'Neuro Studio live upload',
      })

      setSource({
        mode: 'running',
        label: 'Live GPU run',
        detail: 'Streaming real stage events',
        jobId: job.job_id,
      })
      setLoading(false)

      disposeRef.current = streamJob(job.job_id, {
        onEvent: event => {
          setStages(previous =>
            previous.includes(event.stage) ? previous : [...previous, event.stage],
          )
        },
        onEnd: (status, detail) => {
          const cached = (detail.studio as string | undefined) ?? null
          if (status === 'done' && cached) {
            try {
              setStudio(JSON.parse(cached) as Studio)
              setSource({
                mode: 'done',
                label: 'Live GPU run',
                detail: 'Completed on this upload',
                jobId: job.job_id,
              })
            } catch {
              setError('the job finished but its payload was unreadable')
            }
          } else if (status === 'error') {
            setError(String(detail.error ?? 'the GPU run failed'))
          }
          setLoading(false)
        },
        onError: message => setError(message),
      })
    } catch (cause) {
      setError((cause as Error).message)
      setLoading(false)
      setSource({
        mode: 'error',
        label: 'Live run unavailable',
        detail: (cause as Error).message,
      })
    }
  }, [])

  return {
    studio,
    source,
    error,
    loading,
    stages,
    sample,
    samples: CACHED_SAMPLES,
    seconds: studio?.fmri_curves.per_second ?? [],
    selectSample,
    runLive,
    reload: () => void loadSample(sample),
  }
}

/**
 * Sample the per-second telemetry at a playhead position.
 *
 * TRIBE runs at 1 TR/s and the UI plays continuously, so the nearest second is
 * the correct granularity — interpolating would imply a resolution the model
 * does not have.
 */
export function usePlayhead(studio: Studio | null, currentTime: number) {
  return useMemo(() => {
    if (!studio) return null
    const series = studio.networks.networks
    const perSecond = studio.fmri_curves.per_second
    if (perSecond.length === 0) return null

    const tr = studio.fmri_curves.tr_s || 1
    const index = Math.min(
      Math.max(Math.round(currentTime / tr), 0),
      perSecond.length - 1,
    )

    const levels: Record<string, number> = {}
    for (const [name, values] of Object.entries(series)) {
      if (values.length > 0) {
        levels[name] = values[Math.min(index, values.length - 1)]
      }
    }

    return {
      index,
      second: perSecond[index],
      levels,
      // Networks the 64x160 grid never samples only have a clip-level mean.
      staticLevels: studio.networks.static_region_mean_z ?? {},
    }
  }, [studio, currentTime])
}

/** Parse a `MM:SS` label, or a `12.0-16.0` / `12.0` numeric label, to seconds. */
export function timestampToSeconds(label: string): number | null {
  const trimmed = label.trim()
  const mmss = /^(\d{1,2}):(\d{2})$/.exec(trimmed)
  if (mmss) return Number(mmss[1]) * 60 + Number(mmss[2])
  const numeric = /^(\d+(?:\.\d+)?)/.exec(trimmed)
  if (numeric) return Number(numeric[1])
  return null
}

/**
 * Build SVG path data for the per-second cortical curve.
 *
 * Values are `global_z`, which is z-scored over the whole clip (mean forced to
 * 0), so the chart is a shape, not an absolute scale — the README makes the
 * same point and the axis is deliberately unlabelled.
 */
export function buildCurvePaths(studio: Studio | null, width = 1000, height = 70) {
  const seconds = studio?.fmri_curves.per_second ?? []
  if (seconds.length === 0) return null

  const values = seconds.map(item => item.global_z)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  // 4px of headroom top and bottom so the line never touches the frame.
  const pad = 4
  const usable = height - pad * 2

  const points = values.map((value, index) => {
    const x = seconds.length === 1 ? width / 2 : (index / (seconds.length - 1)) * width
    const y = pad + (1 - (value - min) / span) * usable
    return [Number(x.toFixed(1)), Number(y.toFixed(1))] as const
  })

  const line = points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x} ${y}`).join(' ')
  const area = `${line} V${height} H0 Z`

  return { line, area, points, min, max }
}

/**
 * Low-engagement spans as percentages of the clip, for the warning overlay.
 *
 * `stimulus_*` is already lag-corrected upstream, so these line up with what
 * the viewer sees (the README is explicit that comparing a peak against
 * prediction time instead of stimulus time is wrong).
 */
export function lowEngagementSpans(studio: Studio | null) {
  if (!studio) return []
  const perSecond = studio.fmri_curves.per_second
  if (perSecond.length === 0) return []

  const tr = studio.fmri_curves.tr_s || 1
  const total = perSecond.length * tr
  const segments = studio.low_engagement_segments ?? []
  if (segments.length === 0) return []

  return segments.map(segment => ({
    left: Math.max((segment.stimulus_start_s / total) * 100, 0),
    width: Math.max((segment.duration_s / total) * 100, 0.5),
    durationS: segment.duration_s,
  }))
}
