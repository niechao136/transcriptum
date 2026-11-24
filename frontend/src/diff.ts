/** 字符级 diff 高亮：把 ai_edits 落到原文上。
 *
 * MOSS 无字级时间戳，所以 diff 只能做到字符级（不做到词级对齐）。
 * 这里不用通用 diff 库：我们已有结构化的 from→to，
 * 直接按 from 定位更准，也避免把「模型改写整句」误显示为多处小改。
 */
import type { Edit } from './types'

export interface DiffPart {
  text: string
  kind: 'same' | 'changed'
  reason?: string
}

export function buildDiff(source: string, edits: Edit[] | null): DiffPart[] {
  if (!edits || edits.length === 0) return [{ text: source, kind: 'same' }]

  interface Hit { start: number; end: number; to: string; reason: string }
  const hits: Hit[] = []
  for (const e of edits) {
    if (!e.from) continue
    let idx = source.indexOf(e.from)
    // 同一 from 可能出现多次：全部标出，但避免重叠
    while (idx !== -1) {
      const end = idx + e.from.length
      if (!hits.some((h) => idx < h.end && end > h.start)) {
        hits.push({ start: idx, end, to: e.to, reason: e.reason })
      }
      idx = source.indexOf(e.from, idx + 1)
    }
  }
  if (hits.length === 0) return [{ text: source, kind: 'same' }]
  hits.sort((a, b) => a.start - b.start)

  const parts: DiffPart[] = []
  let cursor = 0
  for (const h of hits) {
    if (h.start > cursor) parts.push({ text: source.slice(cursor, h.start), kind: 'same' })
    parts.push({ text: h.to, kind: 'changed', reason: h.reason })
    cursor = h.end
  }
  if (cursor < source.length) parts.push({ text: source.slice(cursor), kind: 'same' })
  return parts
}

export function msToClock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  const m = String(Math.floor(total / 60)).padStart(2, '0')
  const s = String(total % 60).padStart(2, '0')
  return `${m}:${s}`
}
