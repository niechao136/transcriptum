"""§7.5 外接组合下的说话人区分 —— 时间戳对齐。

这是"ASR 外接 + Diarizer 独立"后唯一真正困难的地方。

第一步（零算力）：按重叠时长占比归属
    overlap(句, k) = Σ max(0, min(句.end, 块.end) - max(句.start, 块.start))
    speaker = argmax_k overlap(句, k)
    因为它只取一个标签，天然保证"一句一人"的落库约束。

第二步：三种失败情形
    跨段单句  → 不猜，打 needs_review_speaker，强制人工确认
    句内切换  → 按时间中点切分成两句（音频不切，共享原始时间轴）
    无重叠    → 回退"时间上最近的块"，并打 needs_review_speaker
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .adapters.base import RawSegment
from .config import DiarizationConfig


@dataclass
class SpeakerBlock:
    """Diarizer 输出的说话人时间块。"""

    start_s: float
    end_s: float
    label: str

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError(f"说话人块结束早于开始：{self.start_s} > {self.end_s}")
        if not self.label:
            raise ValueError("说话人块缺少 label")

    def overlaps(self, start: float, end: float) -> float:
        return max(0.0, min(end, self.end_s) - max(start, self.start_s))

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class AssignedSegment:
    start_s: float
    end_s: float
    text: str
    speaker_label: str | None
    needs_review: bool = False
    review_reason: str | None = None
    # 句内切分时标记来源，便于前端识别"这是同一原始句拆出来的"
    split_from: int | None = None
    conf: float | None = None


def assign_speakers(
    segments: list[RawSegment],
    blocks: list[SpeakerBlock],
    cfg: DiarizationConfig,
) -> list[AssignedSegment]:
    """把说话人时间块对齐到 ASR 段上。

    blocks 为空（ASR 自带说话人，或没有 Diarizer）时，直接沿用段上的 speaker_label。
    """
    if not blocks:
        return [
            AssignedSegment(
                start_s=s.start_s,
                end_s=s.end_s,
                text=s.text,
                speaker_label=s.speaker_label,
                conf=s.conf,
            )
            for s in segments
        ]

    out: list[AssignedSegment] = []
    for idx, seg in enumerate(segments):
        out.extend(_assign_one(seg, blocks, cfg, idx))
    return out


def _assign_one(
    seg: RawSegment, blocks: list[SpeakerBlock], cfg: DiarizationConfig, idx: int
) -> list[AssignedSegment]:
    dur = max(1e-6, seg.end_s - seg.start_s)
    per_speaker: dict[str, float] = {}
    for b in blocks:
        ov = b.overlaps(seg.start_s, seg.end_s)
        if ov > 0:
            per_speaker[b.label] = per_speaker.get(b.label, 0.0) + ov

    if not per_speaker:
        # 失败情形：无重叠 → 回退最近块，并强制人工确认
        nearest = _nearest_block(seg, blocks)
        return [
            AssignedSegment(
                start_s=seg.start_s,
                end_s=seg.end_s,
                text=seg.text,
                speaker_label=nearest.label if nearest else seg.speaker_label,
                needs_review=True,
                review_reason="no_overlap",
                conf=seg.conf,
            )
        ]

    ranked = sorted(per_speaker.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_overlap = ranked[0]
    top_ratio = top_overlap / dur
    second = ranked[1] if len(ranked) > 1 else None
    second_ratio = (second[1] / dur) if second else 0.0

    # 覆盖比例过低：归属不可靠
    if top_ratio < cfg.min_overlap_ratio:
        nearest = _nearest_block(seg, blocks)
        return [
            AssignedSegment(
                start_s=seg.start_s,
                end_s=seg.end_s,
                text=seg.text,
                speaker_label=nearest.label if nearest else top_label,
                needs_review=True,
                review_reason="low_overlap_ratio",
                conf=seg.conf,
            )
        ]

    # --- 句内切换：两人重叠都超阈值 → 按时间中点切分 ---
    # 触发条件（两种来源，语义一致）：
    #   1) assign_mode == midpoint_split：遇到双人重叠即切
    #   2) on_ambiguous == auto_split：本应标待复核，但配置要求直接切
    both_overlap = (
        second is not None
        and top_ratio >= cfg.min_overlap_ratio
        and second_ratio >= cfg.switch_overlap_threshold
    )
    want_split = cfg.assign_mode == "midpoint_split" or cfg.on_ambiguous == "auto_split"
    if both_overlap and want_split:
        return _split_at_midpoint(seg, ranked, blocks, cfg, idx)

    # --- 歧义：前两名占比接近 → 标待确认 ---
    ambiguous = second is not None and (top_ratio - second_ratio) < cfg.ambiguous_margin
    return [
        AssignedSegment(
            start_s=seg.start_s,
            end_s=seg.end_s,
            text=seg.text,
            speaker_label=top_label,
            needs_review=ambiguous,
            review_reason="ambiguous_overlap" if ambiguous else None,
            conf=seg.conf,
        )
    ]


def _nearest_block(seg: RawSegment, blocks: list[SpeakerBlock]) -> SpeakerBlock | None:
    if not blocks:
        return None

    def dist(b: SpeakerBlock) -> float:
        if b.end_s < seg.start_s:
            return seg.start_s - b.end_s
        if b.start_s > seg.end_s:
            return b.start_s - seg.end_s
        return 0.0

    return min(blocks, key=dist)


def _split_at_midpoint(
    seg: RawSegment,
    ranked: list[tuple[str, float]],
    blocks: list[SpeakerBlock],
    cfg: DiarizationConfig,
    idx: int,
) -> list[AssignedSegment]:
    """按两人重叠的时间分界点切分（不是简单算术中点，用实际切换点更准）。"""
    top_label = ranked[0][0]
    second_label = ranked[1][0]

    # 找两位说话人重叠区之间的分界：取两人块的交界时刻
    a_blocks = [b for b in blocks if b.label == top_label]
    b_blocks = [b for b in blocks if b.label == second_label]
    boundary = _find_boundary(seg, a_blocks, b_blocks)

    if boundary is None or not (seg.start_s < boundary < seg.end_s):
        return [
            AssignedSegment(
                start_s=seg.start_s,
                end_s=seg.end_s,
                text=seg.text,
                speaker_label=top_label,
                needs_review=True,
                review_reason="switch_boundary_unresolved",
                conf=seg.conf,
            )
        ]

    # 按分界点把文本切成两段：优先在标点处切，退化为按长度比例切
    left_text, right_text = _split_text(seg.text, seg.start_s, seg.end_s, boundary)
    if not left_text.strip() or not right_text.strip():
        return [
            AssignedSegment(
                start_s=seg.start_s,
                end_s=seg.end_s,
                text=seg.text,
                speaker_label=top_label,
                needs_review=True,
                review_reason="switch_text_unsplittable",
                conf=seg.conf,
            )
        ]

    first_label, second_out = (
        (top_label, second_label)
        if _overlap_pref(a_blocks, seg.start_s, boundary) >= _overlap_pref(
            b_blocks, seg.start_s, boundary
        )
        else (second_label, top_label)
    )

    return [
        AssignedSegment(
            start_s=seg.start_s,
            end_s=boundary,
            text=left_text,
            speaker_label=first_label,
            split_from=idx,
            conf=seg.conf,
        ),
        AssignedSegment(
            start_s=boundary,
            end_s=seg.end_s,
            text=right_text,
            speaker_label=second_out,
            split_from=idx,
            conf=seg.conf,
        ),
    ]


def _overlap_pref(blocks: list[SpeakerBlock], start: float, end: float) -> float:
    return sum(b.overlaps(start, end) for b in blocks)


def _find_boundary(
    seg: RawSegment, a_blocks: list[SpeakerBlock], b_blocks: list[SpeakerBlock]
) -> float | None:
    """两人块交界时刻：取离句子中点最近的跨度交界。"""
    mid = (seg.start_s + seg.end_s) / 2
    candidates: list[float] = []
    for a in a_blocks:
        for b in b_blocks:
            candidates.append((a.end_s + b.start_s) / 2)
            candidates.append((b.end_s + a.start_s) / 2)
    inside = [c for c in candidates if seg.start_s < c < seg.end_s]
    if not inside:
        return None
    return min(inside, key=lambda c: abs(c - mid))


def _split_text(text: str, start: float, end: float, boundary: float) -> tuple[str, str]:
    """按时间比例定位文本切点，并尽量吸附到标点。"""
    if not text:
        return "", ""
    dur = max(1e-6, end - start)
    ratio = min(1.0, max(0.0, (boundary - start) / dur))
    pos = int(round(len(text) * ratio))
    pos = max(1, min(len(text) - 1, pos))

    # 在 ±20% 范围内找标点更自然的切点
    window = max(1, len(text) // 5)
    best = pos
    best_score = -1.0
    for cand in range(max(1, pos - window), min(len(text) - 1, pos + window) + 1):
        ch = text[cand - 1]
        score = 1.0 if ch in "，。！？；：、,.!?;:" else 0.0
        # 越靠近比例切点越优先
        score -= abs(cand - pos) / (window * 4)
        if score > best_score:
            best_score = score
            best = cand
    return text[:best], text[best:]


def blocks_from_labels(segments: list[RawSegment]) -> list[SpeakerBlock]:
    """ASR 自带说话人时，把带标签的段转成说话人块。"""
    blocks: list[SpeakerBlock] = []
    for s in segments:
        if s.speaker_label:
            blocks.append(
                SpeakerBlock(start_s=s.start_s, end_s=s.end_s, label=s.speaker_label)
            )
    return _merge_same_label(blocks)


def _merge_same_label(blocks: list[SpeakerBlock]) -> list[SpeakerBlock]:
    if not blocks:
        return []
    ordered = sorted(blocks, key=lambda b: (b.start_s, b.end_s))
    merged: list[SpeakerBlock] = [ordered[0]]
    for b in ordered[1:]:
        last = merged[-1]
        if b.label == last.label and b.start_s <= last.end_s + 1e-6:
            merged[-1] = SpeakerBlock(
                start_s=last.start_s, end_s=max(last.end_s, b.end_s), label=last.label
            )
        else:
            merged.append(b)
    return merged
