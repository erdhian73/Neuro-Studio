'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  CACHED_SAMPLES,
  SAMPLE_LABELS,
  pingWarmup,
  publicVideoPath,
  type CachedSample,
} from '@/lib/api'
import type { DataSource, Studio } from '@/lib/types'
import {
  STAGE_LABELS,
  buildCurvePaths,
  lowEngagementSpans,
  timestampToSeconds,
  usePlayhead,
  useStudio,
} from '@/lib/useStudio'
import {
  Activity,
  AudioLines,
  ChevronDown,
  CircleHelp,
  Clock3,
  Film,
  FolderOpen,
  Grid2X2,
  Layers3,
  LayoutGrid,
  Lock,
  Maximize2,
  MessageSquareText,
  MoreHorizontal,
  Move,
  Play,
  Plus,
  Search,
  Settings2,
  Sparkles,
  SplitSquareHorizontal,
  SunMedium,
  Type,
  Upload,
  WandSparkles,
  Zap,
  BrainCircuit,
  Image as ImageIcon,
  CheckCircle2,
} from 'lucide-react'

const tools = [
  { icon: Move, label: 'Select' },
  { icon: SplitSquareHorizontal, label: 'Cut' },
  { icon: Type, label: 'Text' },
  { icon: AudioLines, label: 'Audio' },
  { icon: WandSparkles, label: 'Effects' },
]

/**
 * Canvas preview source. sample_3.mp4 is a 25 s clip and has matching TRIBE v2
 * predictions in frontend/public/cache/sample_3.json, so the picture and the
 * fMRI track below it stay in sync.
 */
const PREVIEW_VIDEO = '/videos/sample_3.mp4'

/**
 * Frame rate of PREVIEW_VIDEO. Browsers do not expose fps on <video>, so this
 * cannot be read from the element and has to be declared. Verified with
 * ffprobe rather than assumed:
 *
 *   sample_2.mp4 : 25/1        = 25 fps
 *   sample_3.mp4 : 24000/1001  = 23.976 fps  <-- PREVIEW_VIDEO
 *   sample_4.mp4 : 25/1        = 25 fps
 *
 * Update this if PREVIEW_VIDEO changes. Displayed as a rounded integer (`24`),
 * which is the conventional label for 24000/1001.
 */
const PREVIEW_FPS = 24

/**
 * `mm:ss` for the timeline readout. Deliberately no frame field: the preview is
 * 23.976 fps, so an `hh:mm:ss:ff` display would need drop-frame handling to be
 * honest and the seconds resolution is all this timeline needs.
 */
function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0
  const whole = Math.floor(seconds)
  const mm = String(Math.floor(whole / 60)).padStart(2, '0')
  const ss = String(whole % 60).padStart(2, '0')
  return `${mm}:${ss}`
}

/** Euclid: used to reduce 1280x720 to "16:9". */
function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b)
}

/**
 * Asset browser contents. These mirror the real files in frontend/public/videos
 * (duration and 720p resolution verified with ffprobe), so no card claims a
 * spec the project does not actually contain. The "Ambient drift" entry is the
 * audio track embedded in the clips rather than a separate file.
 *
 * Durations are the real ones from each cached report (31.0 / 25.2 / 15.0 s),
 * floored to whole seconds for display. They are *not* rounded up: sample_3 is
 * 25.2 s, so it reads 00:25.
 */
const assets = [
  { title: 'Sample 02 · source', seconds: 31.0, tone: 'violet', kind: 'VIDEO' },
  { title: 'Sample 03 · cinematic', seconds: 25.2, tone: 'blue', kind: 'VIDEO' },
  { title: 'Sample 04 · AI optimized', seconds: 15.0, tone: 'peach', kind: 'VIDEO' },
  { title: 'Ambient drift', seconds: null, tone: 'green', kind: 'AUDIO' },
]

function assetMeta(asset: (typeof assets)[number]): string {
  return asset.seconds === null
    ? 'AAC · 44.1 kHz'
    : `${formatTime(asset.seconds)} · 720p`
}

/**
 * Network colours, keyed by the backend's network names.
 *
 * The bar *values* come from the live/cached studio payload; only the palette
 * is declared here, because colour is presentation, not data.
 */
const NETWORK_TONES: Record<string, string> = {
  'Visual V1-V4': 'lime',
  Auditory: 'cyan',
  'Language / Broca': 'violet',
  'Dorsal Attention': 'amber',
  'Default Mode': 'pink',
}

/** Display order for the telemetry bars (matches the backend's zone_map). */
const NETWORK_ORDER = Object.keys(NETWORK_TONES)

function Telemetry({
  levels,
  staticLevels,
  vertices,
}: {
  levels: Record<string, number>
  staticLevels: Record<string, number>
  vertices: number
}) {
  // `Language / Broca` samples zero grid cells, so it has a clip-level mean
  // only. It is rendered greyed and labelled, never animated (see the JSON's
  // `static_note`).
  const rows = NETWORK_ORDER.map(label => {
    const animated = levels[label]
    const isStatic = animated === undefined
    const raw = isStatic ? staticLevels[label] : animated
    return {
      label,
      tone: NETWORK_TONES[label],
      value: typeof raw === 'number' ? Math.round(raw) : null,
      isStatic,
    }
  })

  return (
    <div className="telemetry-strip">
      <div className="telemetry-heading">
        <span>
          <BrainCircuit size={13} /> LIVE NEURAL TELEMETRY
        </span>
        <b>
          <i /> {vertices.toLocaleString('en-US')} active vertices
        </b>
      </div>
      <div className="network-bars">
        {rows.map(row => (
          <div className={`network${row.isStatic ? ' network-static' : ''}`} key={row.label}>
            <div>
              <span>
                {row.label}
                {row.isStatic ? <em> clip mean</em> : null}
              </span>
              <strong>{row.value === null ? '—' : `${row.value}%`}</strong>
            </div>
            <div className="progress">
              <span
                className={row.tone}
                style={{ width: `${row.value ?? 0}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function NeuroInspector({
  studio,
  source,
  stages,
  loading,
  error,
  onSelectSample,
  onUpload,
  onSeek,
  sample,
}: {
  studio: Studio | null
  source: DataSource
  stages: string[]
  loading: boolean
  error: string | null
  onSelectSample: (sample: CachedSample) => void
  onUpload: (file: File) => void
  onSeek: (seconds: number) => void
  sample: CachedSample
}) {
  const fileRef = useRef<HTMLInputElement>(null)

  // Real verdict text. `null` fields render as an explicit dash rather than a
  // plausible-looking placeholder, which is the whole point of this rewrite.
  const verdict = studio?.verdict
  const confidence =
    typeof verdict?.confidence === 'number'
      ? `${Math.round(verdict.confidence * 100)}% confidence`
      : '— confidence'

  const diagnostics = studio?.diagnostics_by_timestamp ?? []
  const hooks = studio?.rewritten_voiceover_hooks ?? []
  const muse = studio?.muse_image

  return (
    <div className="neuro-inspector">
      <div className="inspector-head">
        <div>
          <p className="eyebrow">NEURO AI PIPELINE</p>
          <h2>Inspector</h2>
        </div>
        <span className={`live-pill mode-${source.mode}`} title={source.detail}>
          <i /> {source.label.toUpperCase()}
        </span>
      </div>

      <section className="sample-section">
        <div className="section-label">
          DEMO SAMPLES <span>REAL TRIBE RUNS</span>
        </div>
        <div className="sample-grid">
          {CACHED_SAMPLES.map((name, i) => (
            <button
              key={name}
              className={name === sample ? 'sample active' : 'sample'}
              onClick={() => onSelectSample(name)}
            >
              <span>0{i + 1}</span>
              {SAMPLE_LABELS[name]}
            </button>
          ))}
        </div>
        <button
          className="upload-custom"
          onClick={() => fileRef.current?.click()}
          disabled={loading}
        >
          <Upload size={13} /> {loading ? 'Working…' : 'Upload Custom MP4'}
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="video/mp4,video/*"
          hidden
          onChange={event => {
            const file = event.target.files?.[0]
            if (file) onUpload(file)
            // Reset so re-picking the same file fires `change` again.
            event.target.value = ''
          }}
        />

        {stages.length > 0 ? (
          <ol className="stage-list">
            {stages.map(stage => (
              <li key={stage} className={stage === 'done' ? 'stage done' : 'stage'}>
                {STAGE_LABELS[stage] ?? stage}
              </li>
            ))}
          </ol>
        ) : null}

        {error ? <p className="inspector-error">{error}</p> : null}
      </section>

      <section className="diagnosis-card">
        <div className="card-title">
          <span className="spark-icon">
            <Zap size={13} fill="currentColor" />
          </span>
          <div>
            <strong>Muse Spark 1.3</strong>
            <small>Cognitive Diagnosis</small>
          </div>
          <span className="confidence">{confidence}</span>
        </div>

        {verdict?.summary ? <p className="verdict-summary">{verdict.summary}</p> : null}

        {diagnostics.length === 0 ? (
          <p className="inspector-empty">No timestamped moments in this report.</p>
        ) : (
          diagnostics.slice(0, 3).map((item, index) => {
            const seconds = timestampToSeconds(item.timestamp)
            return (
              <div className={`insight insight-${item.kind}`} key={`${item.timestamp}-${index}`}>
                <button
                  className="insight-seek"
                  onClick={() => seconds !== null && onSeek(seconds)}
                  disabled={seconds === null}
                  title={seconds === null ? 'no seek target' : `Seek to ${item.timestamp}`}
                >
                  {item.timestamp || '—'}
                </button>
                <p>{item.interpretation || item.evidence}</p>
              </div>
            )
          })
        )}

        {hooks.length > 0 ? (
          <div className="rewrite">
            <span>REWRITTEN VOICEOVER HOOK · {hooks[0].target_time}</span>
            <p>&quot;{hooks[0].hook}&quot;</p>
          </div>
        ) : null}

        {verdict?.lateralisation_note ? (
          <p className="lateralisation-note">{verdict.lateralisation_note}</p>
        ) : null}
      </section>

      <section className="scene-card">
        <div className="card-title">
          <span className="image-icon">
            <ImageIcon size={13} />
          </span>
          <div>
            <strong>Muse Image 1.0</strong>
            <small>Scene Replacement</small>
          </div>
        </div>
        <div className="before-after">
          <div className="scene-thumb before">
            <span>BEFORE</span>
          </div>
          <div className="scene-thumb after">
            {muse?.available && muse.before_after?.after_base64 ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={`data:image/png;base64,${muse.before_after.after_base64}`}
                alt="Muse Image replacement frame"
              />
            ) : (
              <span>NOT GENERATED</span>
            )}
          </div>
        </div>
        <p className="muse-note">
          {muse?.available
            ? 'Generated by muse-image-1.0 for this clip.'
            : 'Muse Image is not called by the analysis route, so no replacement frame exists.'}
        </p>
        <button className="apply-fix" disabled={!muse?.available}>
          <CheckCircle2 size={14} /> Apply AI Fix to Timeline
        </button>
      </section>
    </div>
  )
}

export default function Page() {
  const [activeTool, setActiveTool] = useState('Select')
  const [playing, setPlaying] = useState(false)
  const [activeTab, setActiveTab] = useState('Inspector')

  // --- studio data (cached sample, optionally replaced by a live GPU run) ---
  const {
    studio,
    source,
    stages,
    loading,
    error,
    sample,
    selectSample,
    runLive,
  } = useStudio()

  // Warm the L40S as soon as the page opens. `GET /healthz` answers from the
  // web container without touching the model, so without this ping the first
  // upload pays the whole cold start (~30 GB of weights, minutes) while the
  // user watches a spinner. The request is deliberately silent: progress is not
  // rendered, nothing is awaited, and a failure (offline, cold container still
  // booting past the client timeout) must never surface as a UI error — the
  // container keeps booting either way, which is the only part we rely on.
  useEffect(() => {
    void pingWarmup()
  }, [])

  // --- video preview ----------------------------------------------------
  // `playing` mirrors the <video> element rather than driving it directly, so
  // the button and the element can never disagree (e.g. when playback ends on
  // its own, or a scrub seeks).
  const videoRef = useRef<HTMLVideoElement>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  // Real source attributes, read from the element rather than hardcoded, so the
  // header can never drift out of sync with the file again. NOTE: browsers do
  // not expose frame rate, so fps stays a declared constant (see below).
  const [resolution, setResolution] = useState<{ width: number; height: number } | null>(null)

  // The preview always plays the clip that has a matching TRIBE report, so the
  // picture and the telemetry strip below can never disagree. The cache stores
  // a repo-relative path, so it is mapped to the served `/videos/...` URL.
  const previewVideo = studio?.video ? publicVideoPath(studio.video) : PREVIEW_VIDEO

  // Telemetry sampled at the playhead: this is what makes the strip respond to
  // playback instead of showing one static set of numbers.
  const playhead = usePlayhead(studio, currentTime)

  // Real cortical curve for the fMRI track (replaces the old hand-drawn path).
  const curve = buildCurvePaths(studio)
  const seconds = studio?.fmri_curves.per_second ?? []
  const deadSpans = lowEngagementSpans(studio)

  /**
   * Timeline ruler ticks every 5 s up to the real duration. Integer labels mean
   * a 25.212 s clip ends on "00:25" — floors, never rounds up past the end.
   */
  const rulerTicks = (() => {
    const total = duration > 0 ? duration : 0
    const ticks: number[] = []
    for (let value = 0; value <= total + 0.001; value += 5) {
      ticks.push(Math.floor(value))
    }
    return ticks
  })()

  /** Fraction of the clip already played, 0..1. Drives scrub bar + playhead. */
  const progress = duration > 0 ? Math.min(currentTime / duration, 1) : 0

  /**
   * Short resolution label ("720p" / "1080p") derived from the real video, and
   * the aspect ratio as a simplified "16:9". Both fall back to null until
   * metadata loads, so the header shows nothing rather than a wrong value.
   */
  const resolutionLabel = resolution ? `${resolution.height}p` : null
  const aspectLabel = resolution
    ? (() => {
        const divisor = gcd(resolution.width, resolution.height)
        return `${resolution.width / divisor}:${resolution.height / divisor}`
      })()
    : null

  const togglePlay = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      // play() rejects if the browser blocks autoplay before a gesture; the
      // click is a gesture, but a failed promise must not become an unhandled
      // rejection that trips the Next dev overlay.
      void video.play().catch(() => setPlaying(false))
    } else {
      video.pause()
    }
  }, [])

  // Lock the playhead to the element's own clock via rAF. `timeupdate` only
  // fires ~4x/s, which makes the playhead visibly stutter; rAF tracks every
  // painted frame.
  useEffect(() => {
    if (!playing) return
    let frame = 0
    const tick = () => {
      const video = videoRef.current
      if (video) setCurrentTime(video.currentTime)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing])

  // Read the real source attributes. The `onLoadedMetadata` prop alone is not
  // enough here: when the clip is already in the HTTP cache the event can fire
  // before React attaches its listener, leaving duration at 0 and the scrub bar
  // dead (worse after a sample switch, which remounts the <video> via `key`).
  // Re-reading on mount plus on `loadedmetadata` covers both orders.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const read = () => {
      if (Number.isFinite(video.duration) && video.duration > 0) {
        setDuration(video.duration)
      }
      if (video.videoWidth > 0) {
        setResolution({ width: video.videoWidth, height: video.videoHeight })
      }
    }

    read()
    video.addEventListener('loadedmetadata', read)
    video.addEventListener('durationchange', read)
    return () => {
      video.removeEventListener('loadedmetadata', read)
      video.removeEventListener('durationchange', read)
    }
  }, [previewVideo])

  /** Seek to an absolute time; used by the Inspector's timestamp buttons. */
  const seekTo = useCallback((seconds: number) => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = Math.min(Math.max(seconds, 0), video.duration || seconds)
    setCurrentTime(video.currentTime)
  }, [])

  const handleScrub = (event: React.MouseEvent<HTMLDivElement>) => {
    const video = videoRef.current
    if (!video || duration <= 0) return
    const rect = event.currentTarget.getBoundingClientRect()
    const ratio = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1)
    video.currentTime = ratio * duration
    setCurrentTime(video.currentTime)
  }

  // Spacebar toggles playback, but not while a form control or the video
  // itself has focus (the video has native key handling when focused).
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code !== 'Space') return
      const target = event.target as HTMLElement | null
      if (target && target.closest('input, textarea, select, button, video')) return
      event.preventDefault()
      togglePlay()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [togglePlay])

  return (
    <main className="studio-shell">
      <header className="studio-header">
        <div className="brand-lockup">
          <div className="brand-mark">
            <Sparkles size={17} />
          </div>
          <span>NEURO</span>
          <span className="brand-divider">/</span>
          <span className="brand-muted">STUDIO</span>
        </div>
        <div className="project-name">
          <span className="status-dot" /> TRIBE v2 / Untitled ritual <ChevronDown size={14} />
        </div>
        <div className="header-actions">
          <button className="icon-button" aria-label="Help">
            <CircleHelp size={17} />
          </button>
          <button className="icon-button" aria-label="Settings">
            <Settings2 size={17} />
          </button>
          <button className="share-button">
            <Upload size={14} /> Export
          </button>
          <div className="avatar">AK</div>
        </div>
      </header>

      <div className="studio-body">
        <aside className="left-rail">
          <div className="rail-group">
            <button className="rail-button rail-active" aria-label="Edit">
              <LayoutGrid size={18} />
              <span>Edit</span>
            </button>
            <button className="rail-button" aria-label="Assets">
              <FolderOpen size={18} />
              <span>Assets</span>
            </button>
            <button className="rail-button" aria-label="Templates">
              <Grid2X2 size={18} />
              <span>Templates</span>
            </button>
            <button className="rail-button" aria-label="Brand">
              <Layers3 size={18} />
              <span>Brand</span>
            </button>
          </div>
          <div className="rail-bottom">
            <button className="rail-button" aria-label="Activity">
              <Activity size={18} />
              <span>Activity</span>
            </button>
            <button className="rail-button" aria-label="Preferences">
              <SunMedium size={18} />
              <span>Theme</span>
            </button>
          </div>
        </aside>

        <section className="workspace">
          <div className="workspace-toolbar">
            <div className="tool-tabs">
              {tools.map(({ icon: Icon, label }) => (
                <button
                  key={label}
                  onClick={() => setActiveTool(label)}
                  className={`tool-tab ${activeTool === label ? 'selected' : ''}`}
                >
                  <Icon size={15} />
                  <span>{label}</span>
                </button>
              ))}
            </div>
            <div className="toolbar-right">
              <span className="saved">
                <span className="save-dot" /> Saved just now
              </span>
              <button className="icon-button">
                <MoreHorizontal size={17} />
              </button>
            </div>
          </div>

          <div className="canvas-area">
            <div className="canvas-topline">
              <span>Scene 01 · Neural Hook</span>
              <span className="canvas-meta">
                {aspectLabel ?? '—'}&nbsp; · &nbsp;{PREVIEW_FPS} fps&nbsp; · &nbsp;
                {resolutionLabel ?? '—'}
              </span>
            </div>
            <div className="preview-frame">
              {/* Real HTML5 video replaces the old aurora/text mock. The <video>
                  keeps the aspect-ratio of the frame via object-fit: contain so
                  a 16:9 source is never cropped. */}
              <video
                ref={videoRef}
                className="preview-video"
                src={previewVideo}
                key={previewVideo}
                playsInline
                preload="metadata"
                onClick={togglePlay}
                onLoadedMetadata={(event) => {
                  const video = event.currentTarget
                  setDuration(video.duration)
                  setResolution({ width: video.videoWidth, height: video.videoHeight })
                }}
                onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => {
                  setPlaying(false)
                  setCurrentTime(0)
                }}
              />
              <div className="preview-overlay">
                <span>{formatTime(currentTime)}</span>
                <Maximize2 size={14} />
              </div>
            </div>
            <div className="player-controls">
              <span className="timecode">{formatTime(currentTime)}</span>
              <button
                className="play-button"
                onClick={togglePlay}
                aria-label={playing ? 'Pause' : 'Play'}
              >
                {playing ? <span className="pause-bars" /> : <Play size={15} fill="currentColor" />}
              </button>
              <span className="duration">{formatTime(duration)}</span>
              <div
                className="scrub"
                onClick={handleScrub}
                role="slider"
                aria-label="Seek"
                aria-valuemin={0}
                aria-valuemax={Math.round(duration)}
                aria-valuenow={Math.round(currentTime)}
                aria-valuetext={`${formatTime(currentTime)} of ${formatTime(duration)}`}
                tabIndex={0}
              >
                <span style={{ width: `${progress * 100}%` }} />
              </div>
              <span className="zoom">100%</span>
            </div>
            <Telemetry
              levels={playhead?.levels ?? {}}
              staticLevels={playhead?.staticLevels ?? {}}
              vertices={studio?.fmri_curves.n_vertices ?? 20484}
            />
          </div>

          <div className="timeline-panel">
            <div className="timeline-header">
              <div className="timeline-title">
                <span>Timeline</span>
                <span className="timeline-chip">TRIBE v2 · Main composition</span>
              </div>
              <div className="timeline-actions">
                <button className="small-button">
                  <Plus size={14} /> Add track
                </button>
                <button className="icon-button">
                  <Lock size={15} />
                </button>
                <button className="icon-button">
                  <MoreHorizontal size={16} />
                </button>
              </div>
            </div>
            <div className="timeline-ruler">
              {/* Derived from the element's real duration (25.212 s), so the
                  ruler can never disagree with the clip it describes. */}
              {rulerTicks.map(secondsValue => (
                <span key={secondsValue}>{formatTime(secondsValue)}</span>
              ))}
            </div>
            <div className="tracks">
              <div className="track-label">
                <Film size={14} /> V1
              </div>
              <div className="track-row">
                <div className="clip clip-main">
                  <span className="clip-stripe" />
                  Find your frequency <small>{formatTime(duration)}</small>
                </div>
                {/* Playhead position is driven by the <video> clock, so it
                    tracks playback and seeks instead of sitting at 26%. */}
                <div
                  className="playhead"
                  style={{ left: `${progress * 100}%` }}
                  aria-hidden="true"
                />
              </div>

              <div className="track-label">
                <AudioLines size={14} /> A1
              </div>
              <div className="track-row audio-row">
                <div className="audio-clip">
                  <span className="waveform" />
                  Ambient drift <small>00:26</small>
                </div>
              </div>

              <div className="track-label fmri-label">
                <BrainCircuit size={14} /> fMRI
              </div>
              <div className="track-row fmri-row">
                <div className="fmri-name">
                  TRIBE v2 Cortical Activation{' '}
                  <small>
                    {studio?.fmri_curves.n_vertices.toLocaleString('en-US') ?? '—'} vertices
                    {curve ? ` · ${seconds.length} TRs` : ''}
                  </small>
                </div>
                {curve ? (
                  <svg
                    className="activation-chart"
                    viewBox="0 0 1000 70"
                    preserveAspectRatio="none"
                    role="img"
                    aria-label={`Predicted cortical response over ${seconds.length} seconds`}
                  >
                    <defs>
                      <linearGradient id="neuroFill" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0" stopColor="#c8ef72" stopOpacity=".5" />
                        <stop offset="1" stopColor="#39d5c0" stopOpacity=".02" />
                      </linearGradient>
                    </defs>
                    {/* Dead zones, already lag-corrected to stimulus time. */}
                    {deadSpans.map((span, index) => (
                      <rect
                        key={`dead-${index}`}
                        className="warning-zone"
                        x={`${span.left}%`}
                        y="0"
                        width={`${span.width}%`}
                        height="70"
                      />
                    ))}
                    <path className="chart-area" d={curve.area} fill="url(#neuroFill)" />
                    <path className="chart-line" d={curve.line} />
                    {/* Playhead marker so the chart tracks the video. */}
                    <line
                      className="chart-playhead"
                      x1={progress * 1000}
                      x2={progress * 1000}
                      y1="0"
                      y2="70"
                    />
                  </svg>
                ) : (
                  <span className="chart-empty">No TRIBE curve loaded</span>
                )}
                {deadSpans.length > 0 ? (
                  <span className="warning-label">
                    Low engagement{' '}
                    <b>
                      {deadSpans
                        .map(span => `${span.durationS.toFixed(1)}s`)
                        .join(' · ')}
                    </b>
                  </span>
                ) : null}
              </div>
            </div>
          </div>
        </section>

        <aside className="right-panel">
          <div className="panel-tabs">
            {['Media', 'Inspector', 'Effects'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={activeTab === tab ? 'active' : ''}
              >
                {tab === 'Inspector' ? 'Neuro AI' : tab}
              </button>
            ))}
          </div>
          {activeTab === 'Inspector' ? (
            <NeuroInspector
              studio={studio}
              source={source}
              stages={stages}
              loading={loading}
              error={error}
              onSelectSample={selectSample}
              onUpload={runLive}
              onSeek={seekTo}
              sample={sample}
            />
          ) : activeTab === 'Media' ? (
            <div className="media-content">
              <div className="media-heading">
                <div>
                  <p className="eyebrow">PROJECT LIBRARY</p>
                  <h2>Media</h2>
                </div>
                <button className="add-media">
                  <Plus size={17} />
                </button>
              </div>
              <div className="search-box">
                <Search size={14} />
                <span>Search assets...</span>
                <span className="shortcut">⌘ K</span>
              </div>
              <div className="asset-grid">
                {assets.map(asset => (
                  <div className="asset-card" key={asset.title}>
                    <div className={`asset-thumb ${asset.tone}`}>
                      <span className="asset-kind">{asset.kind}</span>
                    </div>
                    <div className="asset-info">
                      <span>{asset.title}</span>
                      <small>{assetMeta(asset)}</small>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="effects-list">
              <p className="eyebrow">MOTION LIBRARY</p>
              <h2>Effects</h2>
              {['Chromatic bloom', 'Slow drift', 'Film grain', 'Soft focus'].map(effect => (
                <button className="effect-row" key={effect}>
                  <span>{effect}</span>
                  <Plus size={14} />
                </button>
              ))}
            </div>
          )}
        </aside>
      </div>

      <footer className="status-bar">
        <span>
          <span className="green-dot" /> All systems calm
        </span>
        <span className="status-center">
          <MessageSquareText size={13} /> Feedback
        </span>
        <span>
          <Clock3 size={13} /> {PREVIEW_FPS} fps&nbsp; · &nbsp;GPU active
        </span>
      </footer>
    </main>
  )
}
