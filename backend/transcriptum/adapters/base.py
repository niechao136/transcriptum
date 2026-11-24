"""适配器契约（设计文档 §7）。

三条硬规则（违反即设计事故）：
1. 能力必须显式声明，UI 按能力降级 —— 不假装有字级时间戳/置信度。
2. 不提供上传 Python 代码、不动态 import、不写 entry_points。
3. 新增模型 = 加一个 adapter 类 + 注册一行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

JobStatus = Literal["queued", "running", "done", "failed"]


class AdapterError(RuntimeError):
    """外部服务调用失败 —— 交由队列 retry，不静默降级。"""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class Capabilities:
    """§7.1 必须显式声明能力。缺失能力决定 UI 退化行为。"""

    has_word_ts: bool = False
    has_confidence: bool = False
    has_speaker: bool = False
    json_mode: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "has_word_ts": self.has_word_ts,
            "has_confidence": self.has_confidence,
            "has_speaker": self.has_speaker,
            "json_mode": self.json_mode,
        }


@dataclass
class RawSegment:
    """ASR/Diarizer 的中间表示。

    speaker_label 是会话内相对标签（§7.5），跨录音不复用。
    """

    start_s: float
    end_s: float
    text: str
    words: list[dict] | None = None
    conf: float | None = None
    speaker_label: str | None = None

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError(f"段结束早于开始：{self.start_s} > {self.end_s}")


@dataclass
class JobState:
    job_id: str
    status: JobStatus
    progress: float = 0.0
    error: str | None = None


@runtime_checkable
class AsrAdapter(Protocol):
    """① 语音识别。

    短音频走同步接口，长音频走 submit/poll/fetch 三段式；
    两者对外统一为本协议，上层完全不感知差异。
    """

    name: str
    capabilities: Capabilities

    def submit(self, audio_path: Path, hotwords: list[str]) -> str: ...
    def poll(self, job_id: str) -> JobState: ...
    def fetch(self, job_id: str) -> list[RawSegment]: ...


@runtime_checkable
class DiarizerAdapter(Protocol):
    """② 说话人分离（独立槽位，MOSS 用 no-op 适配器）。"""

    name: str

    def diarize(self, audio_path: Path, segments: list[RawSegment]) -> list[RawSegment]: ...


@runtime_checkable
class LlmAdapter(Protocol):
    """③ 文本模型：校对与评估解析。"""

    name: str
    capabilities: Capabilities

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str: ...
