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
    """用 ffprobe 取时长；失败返回 0（不阻断主流程）。"""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
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
