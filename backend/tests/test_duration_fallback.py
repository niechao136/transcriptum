"""ffprobe 缺失时录音时长的回退逻辑（报告 §6.1 声称已修复的问题）。

目标机（WSL / 上海服务器）没有 ffprobe，列表页会把所有录音显示成 0:00。
正确回退顺序：
  1. probe_duration_ms() 内部已改为直接读 WAV 头（标准库 wave），
     这是精确值，应优先采用；
  2. 只有当 WAV 也读不出时长（非 WAV / 损坏）时，才用 ASR 最后一句
     end_ms 兜底 —— 短音频上 ASR 时间戳会幻觉出远超文件长度的值。

必须覆盖「probe 返回 0」这条真实路径 —— 之前的测试全部用带 ffprobe 的
本地环境跑，回退分支根本没被执行到，所以修复里的 NameError 一直没被发现。
"""

from __future__ import annotations

import pytest

from transcriptum import audio
from transcriptum.service import RecordingService
from transcriptum.worker import Worker, enqueue


@pytest.fixture()
def svc(cfg, db, deps):
    return RecordingService(cfg, db, deps)


@pytest.fixture()
def rec(cfg, db, deps, wav, svc):
    return svc.create(title="三号机检修交班", kind="meeting", source="upload",
                      original_path=wav)


def test_duration_uses_wav_header_when_ffprobe_missing(svc, db, deps, rec, wav, monkeypatch):
    """ffprobe 不可用但 WAV 可读时，时长必须是 WAV 头的精确值。

    ASR 在 3 秒纯音上会幻觉出 14.9 秒的时间戳，直接拿它回退会写入
    完全错误的时长；WAV 头才是可靠来源。
    """
    real_which = audio.shutil.which
    monkeypatch.setattr(audio.shutil, "which",
                        lambda name: None if name == "ffprobe" else real_which(name))

    enqueue(db, rec["id"], "transcribe")
    Worker(deps).run_once()

    got = svc.get(rec["id"])
    assert got["status"] == "transcribed", f"转写应成功，实际 {got['status']}，error={got.get('error_msg')}"
    segs = svc.segments(rec["id"])
    assert segs, "应产出句子"
    end_ms = max(int(s["end_ms"]) for s in segs)
    wav_ms = audio._wav_duration_ms(wav)
    assert wav_ms > 0 and end_ms > wav_ms, "测试前提：ASR 时间戳大于文件真实时长（幻觉）"
    assert got["duration_ms"] == wav_ms, (
        f"duration_ms 应回退为 WAV 头时长 {wav_ms}，实际 {got['duration_ms']}"
    )


def test_duration_zero_does_not_break_transcribe(svc, db, deps, rec, monkeypatch):
    """回归：回退分支不能抛 NameError 把整条转写打挂。"""
    monkeypatch.setattr(audio, "probe_duration_ms", lambda _p: 0)

    enqueue(db, rec["id"], "transcribe")
    Worker(deps).run_once()

    got = svc.get(rec["id"])
    assert got["status"] == "transcribed"
    assert got.get("error_msg") in (None, ""), f"不应有错误：{got.get('error_msg')}"
