"""MOSS 输出解析器（设计文档 §4 / §7）。

MOSS-Transcribe-Diarize 的输出形态是「带时间戳与说话人的全文」：

    [0.00][S01]启动压缩机需要先确认油压，[2.32][S02]油压正常就可以合闸。[5.10][S01]好。

解析目标：把这段全文拆成 RawSegment 列表，每段一句一人。
解析失败必须抛错，不许"尽力而为"地吞掉内容 —— 否则会静默丢数据。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import RawSegment

# [12.34] 或 [01:02.34]
_TS = r"\[(\d+(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\]"
_TOKEN_RE = re.compile(rf"{_TS}\s*(\[(S\d{{2,3}}|[A-Z]{{1,3}})\])?")
_END_PUNCT = "。！？!?；;…"
_SPLIT_PUNCT = "。！？!?；;"


class MossParseError(ValueError):
    pass


@dataclass
class ParsedSegment:
    start_s: float
    end_s: float
    text: str
    speaker_label: str | None


def _to_seconds(raw: str) -> float:
    if ":" in raw:
        mm, ss = raw.split(":", 1)
        return int(mm) * 60 + float(ss)
    return float(raw)


def parse_moss_transcript(text: str, *, default_end_pad_s: float = 0.0) -> list[ParsedSegment]:
    """把 MOSS 全文解析为句子列表。

    规则：
    - 每个 [时间][说话人] 标记开始一个新候选句；
    - 遇到句末标点则切句（一句一人，符合 §3.2 落库约束）；
    - 末尾没有时间戳的片段，用最后一个时间戳 + 文本长度估算，或按 default_end_pad_s 补齐。
    """
    if not text or not text.strip():
        return []

    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        raise MossParseError("MOSS 输出中未找到任何时间戳标记，无法对齐时间轴")

    events: list[tuple[float, str | None, str]] = []  # (开始时间, 说话人, 文本)
    for idx, m in enumerate(matches):
        start = _to_seconds(m.group(1))
        speaker = m.group(3)
        body_start = m.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        # 说话人切换但中间还有无标记文本时，也作为独立片段
        events.append((start, speaker, body))

    if not events:
        raise MossParseError("解析后没有任何片段")

    # 合并「同一说话人 + 无句末标点」的相邻片段；不同说话人绝不合并（一句一人）
    merged: list[tuple[float, str | None, str]] = []
    for start, speaker, body in events:
        if not body:
            if merged and merged[-1][1] == speaker:
                continue
            continue
        if (
            merged
            and merged[-1][1] == speaker
            and not merged[-1][2].endswith(tuple(_END_PUNCT))
            and not body.startswith(tuple(_SPLIT_PUNCT))
        ):
            prev_start, prev_spk, prev_text = merged[-1]
            merged[-1] = (prev_start, prev_spk, prev_text + body)
        else:
            merged.append((start, speaker, body))

    out: list[ParsedSegment] = []
    for i, (start, speaker, body) in enumerate(merged):
        if i + 1 < len(merged):
            end = merged[i + 1][0]
        else:
            # 最后一段：按语速估算（中文约 5 字/秒），下限 1 秒
            end = start + max(1.0, len(body) / 5.0) + default_end_pad_s
        if end <= start:
            end = start + 0.5
        body = body.strip()
        if not body:
            continue
        out.append(
            ParsedSegment(
                start_s=round(start, 3), end_s=round(end, 3), text=body, speaker_label=speaker
            )
        )
    return out


def to_raw_segments(parsed: list[ParsedSegment], *, has_speaker: bool) -> list[RawSegment]:
    return [
        RawSegment(
            start_s=p.start_s,
            end_s=p.end_s,
            text=p.text,
            speaker_label=p.speaker_label if has_speaker else None,
        )
        for p in parsed
    ]
