"""音频归一化：统一转 16kHz 单声道 WAV（§4 上传链路）。

用 ffmpeg；不可用时抛错而不是带病继续 —— 采样率不对会污染后续所有时间戳。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

TARGET_SR = 16000


class AudioError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def probe_duration_ms(audio_path: Path) -> int:
    """取音频时长（毫秒）；失败返回 0（不阻断主流程）。

    优先 ffprobe；没有 ffprobe 时对 WAV 直接读头（标准库 wave，精确到采样），
    因为归一化产物一定是 16k 单声道 PCM WAV。这样目标机缺 ffprobe 时也能
    得到真实时长，不必依赖 ASR 时间戳（后者在短音频/纯音上会幻觉出
    远超文件长度的值）。
    """
    audio_path = Path(audio_path)
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return _wav_duration_ms(audio_path)
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "error", "-print_format", "json",
                "-show_format", str(audio_path),
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )
        data = json.loads(out.stdout)
        return int(float(data["format"]["duration"]) * 1000)
    except (subprocess.SubprocessError, KeyError, ValueError, json.JSONDecodeError):
        return _wav_duration_ms(audio_path)


def _wav_duration_ms(path: Path) -> int:
    """从 WAV 头算时长；非 WAV 或损坏文件返回 0。"""
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            return int(w.getnframes() / w.getframerate() * 1000)
    except (OSError, EOFError, ValueError, wave.Error):
        return 0


def normalize_to_wav(src: Path, dst: Path, *, sample_rate: int = TARGET_SR) -> Path:
    """转 16k 单声道 WAV。已是目标格式则直接复制。"""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioError("ffmpeg 不可用，无法归一化音频")

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
        str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.SubprocessError as exc:
        raise AudioError(f"ffmpeg 执行失败：{exc}") from exc
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg 归一化失败：{proc.stderr[-800:]}")
    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioError("归一化后输出为空")
    return dst
