/** 校对台：逐句校对 + 键盘操作 + diff 高亮。
 *
 * 设计文档 §5 的验收点：
 * - 逐句跳播、编辑、确认、驳回；
 * - 键盘全流程（j/k 切句、Enter 确认、Backspace 驳回、e 编辑）；
 * - 说话人待确认必须显式标出，不能默认成已确认；
 * - 无字级时间戳时 UI 不能假装能精确到字。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError } from './api'
import { AudioPlayer } from './AudioPlayer'
import { buildDiff, msToClock } from './diff'
import type { Edit, RecordingDetail, Segment } from './types'

interface Props {
  recordingId: string
  onBack: () => void
  lowConfidenceThreshold: number
}

export function ProofreadDesk({ recordingId, onBack, lowConfidenceThreshold }: Props) {
  const [rec, setRec] = useState<RecordingDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [activeIdx, setActiveIdx] = useState(0)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const listRef = useRef<HTMLDivElement>(null)

  const load = useCallback(async () => {
    try {
      setRec(await api.getRecording(recordingId))
      setError(null)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }, [recordingId])

  useEffect(() => { void load() }, [load])

  const segments = rec?.segments ?? []
  const active = segments[activeIdx] as Segment | undefined

  const runJob = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try {
      await fn()
      // worker 是异步的：轮询到状态稳定为止
      for (let i = 0; i < 180; i++) {
        await new Promise((r) => setTimeout(r, 1000))
        const fresh = await api.getRecording(recordingId)
        setRec(fresh)
        if (fresh.status === 'failed' || fresh.status === 'done' ||
            fresh.status === 'transcribed' || fresh.status === 'proofread') break
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const patch = useCallback(async (
    seg: Segment,
    body: { final_text?: string; review_status?: string; speaker_label?: string },
  ) => {
    try {
      await api.updateSegment(recordingId, seg.id, body)
      await load()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }, [recordingId, load])

  const accept = useCallback(async () => {
    if (!active) return
    await patch(active, { review_status: 'accepted' })
    setEditing(false)
    setActiveIdx((i) => Math.min(i + 1, segments.length - 1))
  }, [active, patch, segments.length])

  const reject = useCallback(async () => {
    if (!active) return
    await patch(active, { review_status: 'rejected' })
    setActiveIdx((i) => Math.min(i + 1, segments.length - 1))
  }, [active, patch, segments.length])

  const startEdit = useCallback(() => {
    if (!active) return
    setDraft(active.final_text ?? active.ai_text ?? active.asr_text)
    setEditing(true)
  }, [active])

  const commitEdit = useCallback(async () => {
    if (!active) return
    await patch(active, { final_text: draft, review_status: 'edited' })
    setEditing(false)
    setActiveIdx((i) => Math.min(i + 1, segments.length - 1))
  }, [active, draft, patch, segments.length])

  // 键盘全流程
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName
      const inField = tag === 'TEXTAREA' || tag === 'INPUT'
      if (inField) {
        if (e.key === 'Escape') { e.preventDefault(); setEditing(false) }
        return
      }
      if (editing) {
        if (e.key === 'Escape') { e.preventDefault(); setEditing(false) }
        return
      }
      if (e.key === 'ArrowDown' || e.key === 'j') {
        e.preventDefault(); setActiveIdx((i) => Math.min(i + 1, segments.length - 1))
      } else if (e.key === 'ArrowUp' || e.key === 'k') {
        e.preventDefault(); setActiveIdx((i) => Math.max(i - 1, 0))
      } else if (e.key === 'Enter') {
        e.preventDefault(); void accept()
      } else if (e.key === 'Backspace') {
        e.preventDefault(); void reject()
      } else if (e.key === 'e') {
        e.preventDefault(); startEdit()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [editing, segments.length, accept, reject, startEdit])

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-idx="${activeIdx}"]`)
    el?.scrollIntoView({ block: 'nearest' })
  }, [activeIdx])

  const progress = rec?.review_progress
  const capabilities = rec?.asr_capabilities
  const noWordTs = capabilities ? !capabilities.has_word_ts : true

  const activeRange = useMemo(
    () => (active ? { start_ms: active.start_ms, end_ms: active.end_ms } : null),
    [active],
  )

  if (error && !rec) {
    return (
      <div className="pane">
        <button onClick={onBack}>← 返回</button>
        <p className="err">加载失败：{error}</p>
      </div>
    )
  }
  if (!rec) return <div className="pane">加载中…</div>

  return (
    <div className="desk">
      <header className="desk-head">
        <button onClick={onBack}>← 返回</button>
        <h2>{rec.title}</h2>
        <span className={`badge status-${rec.status}`}>{rec.status}</span>
        {progress && (
          <span className="muted">
            已确认 {progress.reviewed}/{progress.total}
            {progress.needs_speaker > 0 && (
              <b className="warn"> · {progress.needs_speaker} 句说话人待确认</b>
            )}
          </span>
        )}
        <div className="spacer" />
        <button disabled={busy} onClick={() => void runJob(() => api.startTranscribe(recordingId))}>
          转写
        </button>
        <button disabled={busy} onClick={() => void runJob(() => api.startProofread(recordingId))}>
          AI 校对
        </button>
      </header>

      {rec.warnings && rec.warnings.length > 0 && (
        <div className="banner warn">
          {rec.warnings.map((w, i) => <div key={i}>{w}</div>)}
        </div>
      )}
      {noWordTs && (
        <div className="banner info">
          当前 ASR 不提供字级时间戳：跳播精确到句，diff 以字符级展示。
        </div>
      )}
      {error && <div className="banner err">{error}</div>}

      <div className="desk-body">
        <div className="col-audio">
          {rec.has_audio
            ? <AudioPlayer src={api.audioUrl(recordingId)} activeRange={activeRange} />
            : <p className="muted">音频尚未就绪（先转写）</p>}
          <div className="hint">
            <div>↑/↓ 或 j/k 切换句子</div>
            <div>Enter 确认 · Backspace 驳回 · e 编辑</div>
            <div>Esc 取消编辑 · Ctrl+Enter 保存</div>
          </div>
        </div>

        <div className="col-list" ref={listRef}>
          {segments.length === 0 && (
            <p className="muted">还没有句子。先点「转写」跑一遍 ASR。</p>
          )}
          {segments.map((s, idx) => {
            const edits: Edit[] = s.ai_edits ?? []
            const parts = buildDiff(s.asr_text, edits)
            const isActive = idx === activeIdx
            const display = s.final_text ?? s.ai_text ?? s.asr_text
            const lowConf = s.low_confidence ||
              (s.asr_confidence != null && s.asr_confidence < lowConfidenceThreshold)
            return (
              <div
                key={s.id}
                data-idx={idx}
                className={`seg ${isActive ? 'active' : ''} rs-${s.review_status}`}
                onClick={() => { setActiveIdx(idx); setEditing(false) }}
              >
                <div className="seg-meta">
                  <span className="ts">{msToClock(s.start_ms)}</span>
                  <span className={`spk ${s.needs_review_speaker ? 'spk-warn' : ''}`}>
                    {s.speaker_label ?? '?'}
                    {s.needs_review_speaker && (
                      <span title={s.needs_review_speaker_reason ?? '说话人待确认'}> ⚠</span>
                    )}
                  </span>
                  {lowConf && <span className="chip warn">低置信</span>}
                  {s.review_status === 'accepted' && <span className="chip ok">已确认</span>}
                  {s.review_status === 'rejected' && <span className="chip bad">已驳回</span>}
                  {s.review_status === 'edited' && <span className="chip edit">已修改</span>}
                </div>

                {isActive && editing ? (
                  <div className="seg-edit">
                    <textarea
                      autoFocus
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                          e.preventDefault(); void commitEdit()
                        }
                      }}
                    />
                    <div className="row">
                      <button onClick={() => void commitEdit()}>保存</button>
                      <button onClick={() => setEditing(false)}>取消</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="seg-text">
                      {s.review_status === 'accepted' || s.final_text
                        ? <span>{display}</span>
                        : parts.map((p, i) =>
                            p.kind === 'changed' ? (
                              <mark key={i} title={p.reason}>{p.text}</mark>
                            ) : (
                              <span key={i}>{p.text}</span>
                            ),
                          )}
                    </div>
                    {edits.length > 0 && !s.final_text && (
                      <ul className="edits">
                        {edits.map((e, i) => (
                          <li key={i}>
                            <code>{e.from}</code> → <code>{e.to}</code>
                            <span className="muted">（{e.reason}）</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </>
                )}

                {isActive && !editing && (
                  <div className="row seg-actions">
                    <button onClick={(e) => { e.stopPropagation(); void accept() }}>确认</button>
                    <button onClick={(e) => { e.stopPropagation(); void reject() }}>驳回</button>
                    <button onClick={(e) => { e.stopPropagation(); startEdit() }}>编辑</button>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
