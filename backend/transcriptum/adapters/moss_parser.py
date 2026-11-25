"""MOSS-Transcribe-Diarize 输出解析器。

官方 canonical 格式（README / SGLang cookbook）：
    [start][Sxx]text[end] ...
例如：
    [0.48][S01]Welcome everyone[1.66][12.26][S02]The new pipeline is ready[13.81]

关键点：结束时间戳紧跟在文本之后，不是下一个片段的 start。
旧版（错误）实现把 end 当成下一个 start，会让句子时长翻倍 —— 已修复。

同时兼容旧的「无显式 end」格式，按下一个 start 推断 end。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import RawSegment

# [12.34] 或 [01:02.34]，秒为单位
_TS = r"\[(\d+(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\]"
# 完整片段：[start][Sxx] text [end]
_PAIR_RE = re.compile(
    rf"{_TS}\s*\[(S\d{{2,3}}|[A-Z]{{1,3}})\]\s*(.*?)\s*{_TS}",
    re.DOTALL,
)
# 旧格式：只有 [start][Sxx]，end 靠下一个 start 推断
_TOK_RE = re.compile(rf"{_TS}\s*(\[(S\d{{2,3}}|[A-Z]{{1,3}})\])?")


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


def parse_moss_transcript(
    text: str, *, default_end_pad_s: float = 0.0
) -> list[ParsedSegment]:
    """把 MOSS 全文解析为句子列表。

    优先按官方 canonical [start][Sxx]text[end] 解析；若该格式不成立，
    回退到旧格式 [start][Sxx]text ... [next_start][Syy] ...。
    """
    if not text or not text.strip():
        return []

    pairs = list(_PAIR_RE.finditer(text))
    if pairs:
        out = _parse_canonical(pairs)
        if out:
            return out
    return _parse_legacy(text, default_end_pad_s=default_end_pad_s)


def _parse_canonical(matches: list[re.Match]) -> list[ParsedSegment]:
    """解析 [start][Sxx]text[end]，严格校验 start < end。

    歧义处理：canonical 的 [end] 后面不应紧跟另一个 [Sxx] 标记
    （那是旧格式的下一个 start）。若紧跟 speaker 标记，则该 [end] 实为
    下一个片段的 start，本片段按旧格式交给 _parse_legacy 处理。
    """
    out: list[ParsedSegment] = []
    for m in matches:
        start = _to_seconds(m.group(1))
        speaker = m.group(2)
        body = m.group(3).strip()
        end = _to_seconds(m.group(4))

        # 关键消歧：紧跟 speaker 标记 ⇒ 这是旧格式的「下一个 start」
        after = text_after(m, body)
        if after is not None and re.match(r"\s*\[(S\d{2,3}|[A-Z]{1,3})\]", after):
            return []

        if not body:
            continue
        if end <= start:
            raise MossParseError(
                f"MOSS 片段结束时间不晚于开始时间：{start} -> {end}，文本 {body[:30]!r}"
            )
        out.append(
            ParsedSegment(
                start_s=round(start, 3),
                end_s=round(end, 3),
                text=body,
                speaker_label=speaker,
            )
        )
    return out


def text_after(m: re.Match, body: str) -> str | None:
    """返回 body 之后到下一个 [start] 之前的残留文本（用于消歧）。"""
    end = m.end()
    s = m.string
    if end >= len(s):
        return ""
    return s[end:]


def _parse_legacy(text: str, *, default_end_pad_s: float) -> list[ParsedSegment]:
    """兼容旧格式：[start][Sxx]text 反复出现，没有显式 end。"""
    matches = list(_TOK_RE.finditer(text))
    if not matches:
        raise MossParseError("MOSS 输出中未找到任何时间戳标记，无法对齐时间轴")

    events: list[tuple[float, str | None, str]] = []
    for idx, m in enumerate(matches):
        start = _to_seconds(m.group(1))
        speaker = m.group(3)
        body_start = m.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        events.append((start, speaker, text[body_start:body_end].strip()))

    merged: list[tuple[float, str | None, str]] = []
    for start, speaker, body in events:
        if not body:
            continue
        if (
            merged
            and merged[-1][1] == speaker
            and not merged[-1][2].endswith(tuple("。！？!?；;…"))
            and not body.startswith(tuple("。！？!?；;"))
        ):
            ps, spk, pt = merged[-1]
            merged[-1] = (ps, spk, pt + body)
        else:
            merged.append((start, speaker, body))

    out: list[ParsedSegment] = []
    for i, (start, speaker, body) in enumerate(merged):
        end = merged[i + 1][0] if i + 1 < len(merged) else start + max(1.0, len(body) / 5.0)
        end += default_end_pad_s
        if end <= start:
            end = start + 0.5
        out.append(
            ParsedSegment(
                start_s=round(start, 3), end_s=round(end, 3),
                text=body.strip(), speaker_label=speaker,
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
