/** 音频播放器：支持逐句跳播与「播完本句自动停」。
 *
 * 用原生 <audio> + timeupdate 而非 wavesurfer 做跳播，
 * 因为跳播只需要精确 seek；waveform 用 wavesurfer 单独渲染。
 */
import { useEffect, useRef } from 'react'

interface Props {
  src: string
  /** 当前句子的起止（ms），点击列表时设置 */
  activeRange: { start_ms: number; end_ms: number } | null
  onEnded?: () => void
}

export function AudioPlayer({ src, activeRange, onEnded }: Props) {
  const ref = useRef<HTMLAudioElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el || !activeRange) return
    el.currentTime = activeRange.start_ms / 1000
    void el.play().catch(() => {
      // 浏览器自动播放策略拒绝时静默：用户手动点播放即可
    })
  }, [activeRange])

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const handler = () => {
      if (!activeRange) return
      if (el.currentTime * 1000 >= activeRange.end_ms) {
        el.pause()
        onEnded?.()
      }
    }
    el.addEventListener('timeupdate', handler)
    return () => el.removeEventListener('timeupdate', handler)
  }, [activeRange, onEnded])

  return (
    <audio
      ref={ref}
      src={src}
      controls
      preload="metadata"
      style={{ width: '100%' }}
    />
  )
}
