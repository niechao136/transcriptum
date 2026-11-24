"""§7.5 时间重叠对齐 —— 三种失败情形逐条覆盖。

文档点名：
1. 说话人分离把重叠语音直接分给一个标签；
2. 字幕与说话人时间不完全对齐（差 0.5s 就能放错人）；
3. 说话人切换发生在句中。
"""

from __future__ import annotations

import pytest

from transcriptum.adapters.base import RawSegment
from transcriptum.align import AssignedSegment, SpeakerBlock, assign_speakers, blocks_from_labels
from transcriptum.config import DiarizationConfig


def seg(start, end, text="句子", speaker=None, conf=None):
    return RawSegment(start_s=start, end_s=end, text=text,
                      speaker_label=speaker, conf=conf)


def blk(label, start, end):
    return SpeakerBlock(label=label, start_s=start, end_s=end)


CFG = DiarizationConfig()


# ---------- 正常情形 ----------
def test_single_speaker_full_overlap():
    out = assign_speakers([seg(0.0, 3.0)], [blk("A", 0.0, 3.0)], CFG)
    assert len(out) == 1
    assert out[0].speaker_label == "A"
    assert out[0].needs_review is False


def test_no_diarization_degrades_gracefully():
    """无分离能力：沿用 ASR 自带标签，无标签则待确认。"""
    out = assign_speakers([seg(0.0, 2.0)], [], CFG)
    assert out[0].speaker_label is None
    out2 = assign_speakers([seg(0.0, 2.0, speaker="S01")], [], CFG)
    assert out2[0].speaker_label == "S01"
    assert out2[0].needs_review is False


# ---------- 失败情形 1：重叠段被整段分给一个标签 ----------
def test_overlap_picks_max_overlap_speaker():
    """重叠语音只分给一个标签时，按重叠最长者判定。"""
    s = seg(10.0, 14.0)
    out = assign_speakers([s], [blk("A", 10.0, 10.5), blk("B", 10.5, 14.0)], CFG)
    assert out[0].speaker_label == "B"


def test_ambiguous_margin_flags_review():
    """前两名重叠接近（差值 < margin）→ 必须标待确认，不能猜。"""
    s = seg(0.0, 10.0)
    out = assign_speakers([s], [blk("A", 0.0, 5.0), blk("B", 5.0, 10.0)], CFG)
    assert out[0].needs_review is True
    assert out[0].speaker_label in ("A", "B")


def test_clear_margin_not_flagged():
    s = seg(0.0, 10.0)
    out = assign_speakers([s], [blk("A", 0.0, 9.5), blk("B", 9.5, 10.0)], CFG)
    assert out[0].speaker_label == "A"
    assert out[0].needs_review is False


# ---------- 失败情形 2：时间不完全对齐 ----------
def test_offset_tolerance():
    """字幕差 0.3s 仍在容差内，不应放错人。"""
    s = seg(1.0, 4.0)
    out = assign_speakers([s], [blk("A", 1.3, 4.3)], CFG)
    assert out[0].speaker_label == "A"
    assert out[0].needs_review is False


def test_no_overlap_uses_nearest_block():
    """完全无重叠（分离结果偏了 1s 以上）→ 兜底最近块，而不是丢人。"""
    s = seg(5.0, 6.0)
    out = assign_speakers([s], [blk("A", 7.0, 9.0)], CFG)
    assert out[0].speaker_label == "A"
    assert out[0].needs_review is True  # 无重叠属推断，必须标待确认


def test_below_min_overlap_ratio_falls_back():
    s = seg(0.0, 10.0)
    out = assign_speakers([s], [blk("A", 0.0, 2.0), blk("B", 20.0, 30.0)], CFG)
    assert out[0].speaker_label == "A"
    assert out[0].needs_review is True


# ---------- 失败情形 3：句内切换 ----------
def test_mid_sentence_switch_splits():
    """两人各覆盖 50%，midpoint_split 模式下按中点切分。"""
    cfg = DiarizationConfig(assign_mode="midpoint_split")
    s = seg(0.0, 10.0, text="前半句，后半句。")
    out = assign_speakers([s], [blk("A", 0.0, 5.0), blk("B", 5.0, 10.0)], cfg)
    assert len(out) == 2
    assert {o.speaker_label for o in out} == {"A", "B"}
    assert out[0].end_s == out[1].start_s
    assert out[0].start_s == 0.0 and out[1].end_s == 10.0


def test_mid_sentence_switch_flags_when_not_splitting():
    """默认 flag_review 模式下不擅自切分，只标待确认。"""
    s = seg(0.0, 10.0)
    out = assign_speakers([s], [blk("A", 0.0, 5.0), blk("B", 5.0, 10.0)], CFG)
    assert len(out) == 1
    assert out[0].needs_review is True


def test_split_keeps_text_nonempty():
    cfg = DiarizationConfig(assign_mode="midpoint_split")
    s = seg(0.0, 10.0, text="甲说一句，乙接一句。")
    out = assign_speakers([s], [blk("A", 0.0, 5.0), blk("B", 5.0, 10.0)], cfg)
    for o in out:
        assert o.text.strip() != ""


# ---------- 不变量 ----------
def test_output_sorted_and_non_overlapping():
    segs = [seg(0.0, 2.0), seg(2.0, 4.0), seg(4.0, 6.0)]
    blocks = [blk("A", 0.0, 3.0), blk("B", 3.0, 6.0)]
    out = assign_speakers(segs, blocks, DiarizationConfig(assign_mode="midpoint_split"))
    for a, b in zip(out, out[1:]):
        assert a.end_s <= b.start_s + 1e-6


def test_empty_input():
    assert assign_speakers([], [], CFG) == []


def test_invalid_block_raises():
    with pytest.raises(ValueError):
        SpeakerBlock(label="A", start_s=5.0, end_s=1.0)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        DiarizationConfig(assign_mode="nope").validate()


def test_blocks_from_labels():
    segs = [seg(0.0, 2.0, speaker="A"), seg(2.0, 4.0, speaker="A"),
            seg(4.0, 6.0, speaker="B")]
    blocks = blocks_from_labels(segs)
    assert len(blocks) == 2  # 相邻同标签合并
    assert blocks[0].label == "A" and blocks[0].start_s == 0.0
    assert blocks[1].label == "B" and blocks[1].end_s == 6.0
