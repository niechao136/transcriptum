/** 评估页：框架选择、字段编辑、引用展示、提交校验。
 *
 * §6 硬约束在 UI 上必须可见：
 * - 每个字段旁显示引用句号，点击可定位；
 * - 无引用的字段标红且不可提交；
 * - 提交按钮被 blockers 阻塞时给出具体原因。
 */
import { useCallback, useEffect, useState } from 'react'
import { api, ApiError } from './api'
import type { Assessment, Framework, RecordingDetail } from './types'

interface Props {
  recordingId: string
  onBack: () => void
  onJumpToSegment: (seq: number) => void
}

export function AssessmentView({ recordingId, onBack, onJumpToSegment }: Props) {
  const [frameworks, setFrameworks] = useState<Framework[]>([])
  const [rec, setRec] = useState<RecordingDetail | null>(null)
  const [list, setList] = useState<Assessment[]>([])
  const [current, setCurrent] = useState<Assessment | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, string>>({})

  const loadAll = useCallback(async () => {
    try {
      const [fw, r, as] = await Promise.all([
        api.listFrameworks(), api.getRecording(recordingId),
        api.listAssessments(recordingId),
      ])
      setFrameworks(fw)
      setRec(r)
      setList(as)
      setCurrent(as[0] ?? null)
      setError(null)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }, [recordingId])

  useEffect(() => { void loadAll() }, [loadAll])

  const create = async (fwId: number) => {
    try {
      const a = await api.createAssessment(recordingId, fwId)
      await loadAll()
      setCurrent(a)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }

  const saveField = async (key: string) => {
    if (!current) return
    const content = drafts[key]
    if (content === undefined) return
    try {
      const a = await api.updateField(current.id, key, content, current.values
        .find((v) => v.field_key === key)?.citations ?? [])
      setCurrent(a)
      setDrafts((d) => { const n = { ...d }; delete n[key]; return n })
      await loadAll()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }

  const submit = async () => {
    if (!current) return
    try {
      const a = await api.submitAssessment(current.id)
      setCurrent(a)
      await loadAll()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }

  return (
    <div className="pane">
      <header className="row">
        <button onClick={onBack}>← 返回</button>
        <h2>{rec?.title ?? ''} · 考核评估</h2>
      </header>

      {error && <div className="banner err">{error}</div>}

      <section className="card">
        <h3>选择框架</h3>
        <div className="row">
          {frameworks.map((f) => (
            <button key={f.id} onClick={() => void create(f.id)}>
              {f.name} v{f.version}
            </button>
          ))}
        </div>
        {list.length > 0 && (
          <p className="muted">
            已有 {list.length} 份评估：
            {list.map((a) => (
              <button key={a.id} className="link" onClick={() => setCurrent(a)}>
                #{a.id} {a.framework.name}（{a.status === 'draft' ? '草稿' : '已提交'}）
              </button>
            ))}
          </p>
        )}
      </section>

      {current && (
        <section className="card">
          <header className="row">
            <h3>{current.framework.name} v{current.framework_version}</h3>
            <span className={`badge ${current.status === 'submitted' ? 'ok' : ''}`}>
              {current.status === 'submitted' ? '已提交' : '草稿'}
            </span>
            <div className="spacer" />
            <button
              disabled={!current.submittable || current.status === 'submitted'}
              onClick={() => void submit()}
            >提交</button>
          </header>

          {current.blockers.length > 0 && (
            <ul className="blockers">
              {current.blockers.map((b, i) => <li key={i}>{b}</li>)}
            </ul>
          )}

          {current.values.map((v) => {
            const def = current.framework.fields.find((f) => f.key === v.field_key)
            const draft = drafts[v.field_key]
            const editing = draft !== undefined
            const bad = v.status === 'missing_citation' || v.status === 'invalid_value'
            return (
              <div key={v.id} className={`field ${bad ? 'field-bad' : ''}`}>
                <div className="field-head">
                  <b>{def?.label ?? v.field_key}</b>
                  {def?.required && <span className="chip">必填</span>}
                  <span className={`chip status-${v.status}`}>
                    {v.status === 'ok' ? '已提取'
                      : v.status === 'missing_citation' ? '缺引用'
                      : v.status === 'invalid_value' ? '类型不符'
                      : '未提取'}
                  </span>
                  {v.source === 'human_edited' && <span className="chip edit">人工修正</span>}
                  {v.confidence != null && (
                    <span className="muted">置信度 {v.confidence.toFixed(2)}</span>
                  )}
                </div>

                <div className="cites">
                  引用：
                  {v.citations.length === 0
                    ? <b className="warn">无（不可提交）</b>
                    : v.citations.map((seq) => (
                        <button key={seq} className="link" onClick={() => onJumpToSegment(seq)}>
                          [{seq}]
                        </button>
                      ))}
                </div>

                {editing ? (
                  <div className="row">
                    <textarea
                      value={draft}
                      onChange={(e) => setDrafts((d) => ({ ...d, [v.field_key]: e.target.value }))}
                    />
                    <button onClick={() => void saveField(v.field_key)}>保存</button>
                    <button onClick={() => setDrafts((d) => {
                      const n = { ...d }; delete n[v.field_key]; return n
                    })}>取消</button>
                  </div>
                ) : (
                  <div className="row">
                    <p className="field-content">{v.content || <i className="muted">（空）</i>}</p>
                    <button onClick={() => setDrafts((d) => ({ ...d, [v.field_key]: v.content ?? '' }))}>
                      修正
                    </button>
                  </div>
                )}
                {def?.guidance && <p className="hint">{def.guidance}</p>}
              </div>
            )
          })}
        </section>
      )}
    </div>
  )
}
