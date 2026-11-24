/** 录音列表 + 上传 + 导出。 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from './api'
import type { Recording } from './types'

interface Props {
  onOpen: (id: string) => void
  onAssessment: (id: string) => void
}

export function RecordingList({ onOpen, onAssessment }: Props) {
  const [items, setItems] = useState<Recording[]>([])
  const [error, setError] = useState<string | null>(null)
  const [title, setTitle] = useState('')
  const [kind, setKind] = useState<'meeting' | 'assessment'>('meeting')
  const [consent, setConsent] = useState(false)
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const load = useCallback(async () => {
    try {
      setItems(await api.listRecordings())
      setError(null)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const upload = async () => {
    const file = fileRef.current?.files?.[0]
    if (!file) { setError('请选择音频文件'); return }
    if (!title.trim()) { setError('请填写标题'); return }
    if (!consent) { setError('请先确认已获得参会人录音同意'); return }
    setUploading(true)
    try {
      await api.createRecording(file, title.trim(), kind, consent)
      setTitle('')
      if (fileRef.current) fileRef.current.value = ''
      setConsent(false)
      await load()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="pane">
      <h1>会议与评估记录</h1>
      <p className="muted">
        本地部署 · 音频不出内网 · 转写在 CPU/GPU 侧完成，文本仅在本机模型间流转。
      </p>

      <section className="card">
        <h3>新建录音</h3>
        <div className="row">
          <input placeholder="标题" value={title} onChange={(e) => setTitle(e.target.value)} />
          <select value={kind} onChange={(e) => setKind(e.target.value as 'meeting' | 'assessment')}>
            <option value="meeting">会议</option>
            <option value="assessment">考核评估</option>
          </select>
          <input ref={fileRef} type="file" accept="audio/*" />
          <button disabled={uploading} onClick={() => void upload()}>
            {uploading ? '上传中…' : '上传'}
          </button>
        </div>
        <label className="consent">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          我已按规程告知参会人并取得录音同意
        </label>
      </section>

      {error && <div className="banner err">{error}</div>}

      <section className="card">
        <h3>录音列表</h3>
        {items.length === 0 && <p className="muted">还没有录音。</p>}
        <table className="tbl">
          <thead>
            <tr><th>标题</th><th>类型</th><th>状态</th><th>时长</th><th>操作</th></tr>
          </thead>
          <tbody>
            {items.map((r) => (
              <tr key={r.id}>
                <td>{r.title}</td>
                <td>{r.kind === 'meeting' ? '会议' : '考核'}</td>
                <td><span className={`badge status-${r.status}`}>{r.status}</span></td>
                <td>{Math.round(r.duration_ms / 1000)}s</td>
                <td className="actions">
                  <button onClick={() => onOpen(r.id)}>校对</button>
                  <button onClick={() => onAssessment(r.id)}>评估</button>
                  <a href={api.exportUrl(r.id, 'md', true)} download>MD</a>
                  <a href={api.exportUrl(r.id, 'docx', true)} download>DOCX</a>
                  <a href={api.exportUrl(r.id, 'xlsx', true)} download>XLSX</a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
