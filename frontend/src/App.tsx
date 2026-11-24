import { useEffect, useState } from 'react'
import { api, ApiError } from './api'
import { AssessmentView } from './AssessmentView'
import { ProofreadDesk } from './ProofreadDesk'
import { RecordingList } from './RecordingList'
import type { Health } from './types'
import './styles.css'

type Route = { name: 'list' } | { name: 'desk'; id: string } | { name: 'assess'; id: string }

export default function App() {
  const [route, setRoute] = useState<Route>({ name: 'list' })
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.health().then(setHealth).catch((e) =>
      setError(e instanceof ApiError ? e.message : String(e)))
  }, [])

  const lowConf = health?.proofread?.low_confidence_threshold ?? 0.6

  return (
    <div className="app">
      <nav className="topbar">
        <b>Transcriptum</b>
        <span className="muted">本地会议记录 / 考核评估</span>
        <div className="spacer" />
        {health && (
          <span className="caps">
            ASR <code>{health.asr.adapter}</code>
            {!health.asr.capabilities.has_word_ts && <i className="muted">（无字级时间戳）</i>}
            {!health.asr.capabilities.has_confidence && <i className="muted">（无置信度）</i>}
            {' · '}LLM <code>{health.llm.adapter}</code>
          </span>
        )}
      </nav>

      {error && <div className="banner err">{error}</div>}

      {route.name === 'list' && (
        <RecordingList
          onOpen={(id) => setRoute({ name: 'desk', id })}
          onAssessment={(id) => setRoute({ name: 'assess', id })}
        />
      )}
      {route.name === 'desk' && (
        <ProofreadDesk
          recordingId={route.id}
          onBack={() => setRoute({ name: 'list' })}
          lowConfidenceThreshold={lowConf}
        />
      )}
      {route.name === 'assess' && (
        <AssessmentView
          recordingId={route.id}
          onBack={() => setRoute({ name: 'list' })}
          onJumpToSegment={() => setRoute({ name: 'desk', id: route.id })}
        />
      )}
    </div>
  )
}
