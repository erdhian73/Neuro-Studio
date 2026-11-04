/**
 * Types mirroring the backend `studio` block.
 *
 * The same shapes are produced by `agent/studio.py` for both the offline cache
 * (`public/cache/*.json`) and the live `/jobs` endpoint, so one type covers
 * both paths. Keep this in sync with that module if you change a field.
 */

export type NetworkSeries = Record<string, number[]>

export interface NetworkCoverage {
  src_zone_count: number
  has_series: boolean
  seconds: number
}

export interface StudioNetworks {
  n_seconds: number
  networks: NetworkSeries
  coverage: Record<string, NetworkCoverage>
  derivation?: string
  zone_map?: Record<string, string[]>
  unassigned_zones?: string[]
  grid_cell_share?: Record<string, number>
  /** Clip-level mean for networks the 64x160 grid never samples. */
  static_region_mean_z?: Record<string, number>
  static_note?: string
  unavailable_reason?: string
}

export interface FmriSecond {
  second: number
  timestamp: string
  global_z: number
  hemisphere_split: Record<string, number>
}

export interface StudioFmriCurves {
  n_vertices: number
  summary?: string
  mesh?: string
  subject?: string
  tr_s: number
  hemodynamic_lag_s: number
  duration_s: number | null
  per_second: FmriSecond[]
}

export type DiagnosticKind = 'moment' | 'weak_moment' | 'peak_window'

export interface StudioDiagnostic {
  timestamp: string
  kind: DiagnosticKind
  evidence: string
  interpretation: string
  confidence: number | null
}

export interface StudioHook {
  target_time: string
  hook: string
  rationale: string
}

export interface StudioVerdict {
  overall: string | null
  confidence: number | null
  summary: string | null
  lateralisation_note: string | null
}

export interface StudioMuseImage {
  model: string
  available: boolean
  before_after: { after_base64?: string; note?: string } | null
  note?: string
}

/** A stretch the clip spends in its own bottom quantile of predicted response. */
export interface LowEngagementSegment {
  stimulus_start_s: number
  stimulus_end_s: number
  duration_s: number
  mean_z_in_segment?: number
  threshold_z?: number
}

export interface Studio {
  sample: string
  video: string
  schema_version: number
  synthetic: boolean
  fmri_curves: StudioFmriCurves
  networks: StudioNetworks
  diagnostics_by_timestamp: StudioDiagnostic[]
  rewritten_voiceover_hooks: StudioHook[]
  verdict: StudioVerdict
  muse_image: StudioMuseImage
  /** Lag-corrected dead zones, when the report carries them. */
  low_engagement_segments?: LowEngagementSegment[]
  note?: string
}

/** One cached sample file (`public/cache/sample_N.json`). */
export interface SampleCacheFile {
  sample: string
  video: string
  duration_s: number | null
  generated_at: string
  studio: Studio
}

/** Stage names emitted by `agent/pipeline.py::STAGES` (plus job lifecycle). */
export type JobStage =
  | 'queued'
  | 'predict'
  | 'diagnostics'
  | 'brain_grid'
  | 'interpret'
  | 'done'

export type JobStatus = 'pending' | 'running' | 'done' | 'error' | 'cache'

export interface JobEvent {
  job_id: string
  stage: JobStage | string
  seq: number
  ts: number
  data: Record<string, unknown>
}

/** Where the studio's numbers came from — surfaced in the UI. */
export interface DataSource {
  mode: JobStatus
  label: string
  detail: string
  jobId?: string
}
