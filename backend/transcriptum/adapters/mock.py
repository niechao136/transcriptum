"""Mock 适配器 —— M0 阶段验证「上层完全不感知适配器差异」的工具。

设计文档 §11 M0：先用两个假适配器跑通 submit/poll/fetch 与能力降级。
这两个 mock 故意能力不同：
- MockMossAsr: 有说话人、无字级时间戳、无置信度（等于真实 MOSS）
- MockOpenAiAsr: 无说话人、有置信度（等于纯 OpenAI ASR）
同一个上层代码必须都能跑。
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from .base import AdapterError, Capabilities, JobState, RawSegment

_MOCK_TRANSCRIPT = (
    "[0.00][S01]今天我们过一下三号机的检修情况。"
    "[3.20][S02]好的，先看压缩机油压，[5.80][S01]油压正常就可以合闸。"
    "[8.40][S02]明白，那我把合闸前的检查项过一遍。"
    "[12.10][S01]对，安全阀校验记录也要带上。"
)

_MOCK_NO_SPEAKER = (
    "今天我们过一下三号机的检修情况。"
    "好的，先看压缩机油压，油压正常就可以合闸。"
    "明白，那我把合闸前的检查项过一遍。"
)


class MockMossAsr:
    """模拟 MOSS：有说话人、无字级时间戳、无置信度。"""

    name = "mock_moss"
    capabilities = Capabilities(
        has_word_ts=False, has_confidence=False, has_speaker=True, json_mode=False
    )

    def __init__(self, *, delay_s: float = 0.0, fail_times: int = 0) -> None:
        self._jobs: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self._delay = delay_s
        self._fail_times = fail_times
        self._counter = itertools.count(1)

    def submit(self, audio_path: Path, hotwords: list[str]) -> str:
        if not Path(audio_path).exists():
            raise AdapterError(f"音频不存在：{audio_path}", retryable=False)
        job_id = f"moss-{next(self._ids)}"
        self._jobs[job_id] = {"created": time.monotonic(), "hotwords": list(hotwords)}
        return job_id

    def poll(self, job_id: str) -> JobState:
        job = self._jobs.get(job_id)
        if job is None:
            return JobState(job_id=job_id, status="failed", error="unknown job")
        if self._delay and time.monotonic() - job["created"] < self._delay:
            return JobState(job_id=job_id, status="running", progress=0.5)
        return JobState(job_id=job_id, status="done", progress=1.0)

    def fetch(self, job_id: str) -> list[RawSegment]:
        from .moss_parser import parse_moss_transcript, to_raw_segments

        n = next(self._counter)
        if n <= self._fail_times:
            raise AdapterError(f"mock 第 {n} 次 fetch 故意失败")
        return to_raw_segments(parse_moss_transcript(_MOCK_TRANSCRIPT), has_speaker=True)


class MockOpenAiAsr:
    """模拟纯 OpenAI audio/transcriptions：无说话人、有置信度。"""

    name = "mock_openai_audio"
    capabilities = Capabilities(
        has_word_ts=False, has_confidence=True, has_speaker=False, json_mode=False
    )

    def __init__(self, *, segments: list[RawSegment] | None = None) -> None:
        self._segments = segments
        self._jobs: dict[str, list[RawSegment]] = {}
        self._ids = itertools.count(1)

    def submit(self, audio_path: Path, hotwords: list[str]) -> str:
        job_id = f"oai-{next(self._ids)}"
        if self._segments is not None:
            self._jobs[job_id] = list(self._segments)
        else:
            base = [
                (0.0, 3.0, "今天我们过一下三号机的检修情况。"),
                (3.2, 5.6, "好的，先看压缩机油压，油压正常就可以合闸。"),
                (5.8, 8.4, "明白，那我把合闸前的检查项过一遍。"),
                (8.5, 12.0, "对，安全阀校验记录也要带上。"),
            ]
            self._jobs[job_id] = [
                RawSegment(start_s=s, end_s=e, text=t, conf=0.92 if i % 2 == 0 else 0.58)
                for i, (s, e, t) in enumerate(base)
            ]
        return job_id

    def poll(self, job_id: str) -> JobState:
        if job_id not in self._jobs:
            return JobState(job_id=job_id, status="failed", error="unknown job")
        return JobState(job_id=job_id, status="done", progress=1.0)

    def fetch(self, job_id: str) -> list[RawSegment]:
        if job_id not in self._jobs:
            raise AdapterError(f"unknown job {job_id}", retryable=False)
        return list(self._jobs[job_id])


class NoopDiarizer:
    """MOSS 自带说话人时的空适配器；也用于「无分离」降级。"""

    name = "none"

    def diarize(self, audio_path: Path, segments: list[RawSegment]) -> list[RawSegment]:
        return segments


class MockLlm:
    """模拟校对/解析 LLM。

    返回结构化 JSON（edits），用于验证 ai_edits 落库与前端 diff。
    刻意偏保守：只改明显的术语错误，不编造内容（§4.3）。
    """

    name = "mock_llm"
    capabilities = Capabilities(json_mode=True)

    # 简单术语纠正表，模拟「热词表 + 上下文」的保守校对
    _FIXES = [
        ("机油压", "机油压力", "术语补全"),
        ("合砸", "合闸", "同音字错写"),
        ("三号鸡", "三号机", "同音字错写"),
        ("安全法", "安全阀", "同音字错写"),
    ]

    def __init__(self, *, json_mode: bool = True, fail_times: int = 0) -> None:
        self._json_mode = json_mode
        self._fail = fail_times
        self.calls: list[tuple[str, str, bool]] = []

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls.append((system, user, json_mode))
        if self._fail > 0:
            self._fail -= 1
            raise AdapterError("mock llm 故意失败")

        # 评估解析模式：system 里带 ASSESSMENT 标记
        if "ASSESSMENT" in system:
            return json.dumps(self._mock_assessment(user), ensure_ascii=False)

        source = _extract_source(user)
        edits = []
        cur = source
        for wrong, right, reason in self._FIXES:
            if wrong in cur:
                edits.append(
                    {"type": "replace", "from": wrong, "to": right,
                     "reason": reason, "confidence": 0.9}
                )
                cur = cur.replace(wrong, right)
        return json.dumps(
            {"corrected_text": cur, "edits": edits}, ensure_ascii=False
        )

    @staticmethod
    def _mock_assessment(user: str) -> dict:
        return {
            "fields": [
                {
                    "key": "summary",
                    "content": "本次交班确认三号机压缩机油压正常，可执行合闸，"
                               "并需补齐安全阀校验记录。",
                    "citations": [1, 2, 4],
                    "confidence": 0.88,
                },
                {
                    "key": "risk_points",
                    "content": "安全阀校验记录尚未归档，属未闭环风险项。",
                    "citations": [4],
                    "confidence": 0.81,
                },
                {
                    "key": "todo",
                    "content": "合闸前完成检查项逐条核对；补齐安全阀校验记录。",
                    "citations": [3, 4],
                    "confidence": 0.79,
                },
            ]
        }


def _extract_source(user: str) -> str:
    """mock 从 prompt 里取 <source>...</source> 内容。"""
    start = user.find("<source>")
    end = user.find("</source>")
    if start == -1 or end == -1:
        return ""
    return user[start + len("<source>") : end]
