/** 极简 API 客户端：统一错误处理，不吞异常。 */
import type {
  Assessment, Framework, Health, Page, Recording, RecordingDetail, Segment,
} from './types'

const BASE = '/api'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
    this.name = 'ApiError'
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body?.detail ?? detail
    } catch {
      /* 非 JSON 错误体，保留 statusText */
    }
    throw new ApiError(res.status, typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  health: () => req<Health>('/health'),

  // 后端列表接口统一返回 {items,total}，这里在客户端展开，调用方只拿数组
  listRecordings: async () => (await req<Page<Recording>>('/recordings')).items,

  getRecording: (id: string) => req<RecordingDetail>(`/recordings/${id}`),

  createRecording: (file: File, title: string, kind: string, consent: boolean) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('title', title)
    fd.append('kind', kind)
    fd.append('source', 'upload')
    fd.append('consent', String(consent))
    return req<Recording>('/recordings', { method: 'POST', body: fd })
  },

  startTranscribe: (id: string) =>
    req<{ job_id: string }>(`/recordings/${id}/transcribe`, { method: 'POST' }),

  startProofread: (id: string) =>
    req<{ job_id: string }>(`/recordings/${id}/proofread`, { method: 'POST' }),

  updateSegment: (
    id: string,
    segId: number,
    patch: { final_text?: string; review_status?: string; speaker_label?: string },
  ) => req<Segment>(`/recordings/${id}/segments/${segId}`, {
    method: 'PATCH', body: JSON.stringify(patch),
  }),

  bulkAccept: (id: string) =>
    req<{ accepted: number }>(`/recordings/${id}/segments/bulk-accept`, { method: 'POST' }),

  audioUrl: (id: string) => `${BASE}/recordings/${id}/audio`,

  exportUrl: (id: string, fmt: string, redact: boolean) =>
    `${BASE}/recordings/${id}/export?fmt=${fmt}&redact=${redact}`,

  listFrameworks: async () => (await req<Page<Framework>>('/frameworks')).items,

  listAssessments: async (recId: string) =>
    (await req<Page<Assessment>>(`/recordings/${recId}/assessments`)).items,

  createAssessment: (recId: string, frameworkId: number) =>
    req<Assessment>(`/recordings/${recId}/assessments`, {
      method: 'POST', body: JSON.stringify({ framework_id: frameworkId }),
    }),

  getAssessment: (id: number) => req<Assessment>(`/assessments/${id}`),

  updateField: (aid: number, key: string, content: string, citations: number[]) =>
    req<Assessment>(`/assessments/${aid}/fields/${key}`, {
      method: 'PATCH', body: JSON.stringify({ content, citations }),
    }),

  submitAssessment: (aid: number) =>
    req<Assessment>(`/assessments/${aid}/submit`, { method: 'POST' }),
}
