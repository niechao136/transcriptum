/** 后端契约类型（与 backend/transcriptum/api.py 对应） */

export type ReviewStatus = 'pending' | 'accepted' | 'rejected' | 'edited'
export type RecStatus =
  | 'uploaded' | 'transcribing' | 'transcribed'
  | 'proofreading' | 'proofread' | 'parsing' | 'done' | 'failed'

export interface Capabilities {
  has_word_ts: boolean
  has_confidence: boolean
  has_speaker: boolean
  json_mode: boolean
}

export interface Page<T> {
  items: T[]
  total: number
}

export interface Health {
  ok: boolean
  asr: { adapter: string; capabilities: Capabilities }
  llm: { adapter: string; model: string }
  diarizer: string
  proofread?: { low_confidence_threshold: number }
}

export interface Edit {
  type: string
  from: string
  to: string
  reason: string
  confidence: number
}

export interface Segment {
  id: number
  seq: number
  start_ms: number
  end_ms: number
  speaker_label: string | null
  asr_text: string
  ai_text: string | null
  final_text: string | null
  ai_edits: Edit[] | null
  review_status: ReviewStatus
  needs_review_speaker: boolean
  needs_review_speaker_reason: string | null
  low_confidence: boolean
  asr_confidence: number | null
}

export interface Recording {
  id: string
  title: string
  kind: 'meeting' | 'assessment'
  source: 'upload' | 'live'
  status: RecStatus
  error_msg: string | null
  duration_ms: number
  created_at: string
  has_audio: boolean
  warnings: string[] | null
  asr_capabilities: Capabilities | null
}

export interface RecordingDetail extends Recording {
  segments: Segment[]
  speakers: { id: number; label: string; person_id: number | null; needs_review: number }[]
  review_progress: { total: number; reviewed: number; needs_speaker: number }
}

export interface FieldDef {
  key: string
  label: string
  type: 'text' | 'number' | 'enum' | 'list'
  required: boolean
  options: string[]
  guidance: string
}

export interface Framework {
  id: number
  code: string
  name: string
  version: number
  fields: FieldDef[]
}

export interface FieldValue {
  id: number
  field_key: string
  content: string | null
  citations: number[]
  confidence: number | null
  source: 'ai' | 'human_edited'
  status: 'ok' | 'not_extracted' | 'missing_citation' | 'invalid_value'
}

export interface Assessment {
  id: number
  recording_id: string
  framework_id: number
  framework_version: number
  framework: Framework
  status: 'draft' | 'submitted'
  values: FieldValue[]
  submittable: boolean
  blockers: string[]
}
