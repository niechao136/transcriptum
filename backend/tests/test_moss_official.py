"""按 MOSS 官方 README / SGLang cookbook 的输出格式回归。"""

from __future__ import annotations

import pytest

from transcriptum.adapters.moss_parser import MossParseError, parse_moss_transcript
from transcriptum.adapters.base import RawSegment

OFFICIAL = (
    "[0.48][S01]Welcome everyone[1.66]"
    "[12.26][S02]The new pipeline is ready[13.81]"
    "[14.36][S01]Great, include diarization[18.76]"
)


def test_official_format_explicit_end_times():
    segs = parse_moss_transcript(OFFICIAL)
    assert len(segs) == 3
    assert (segs[0].start_s, segs[0].end_s) == (0.48, 1.66)
    assert (segs[1].start_s, segs[1].end_s) == (12.26, 13.81)
    assert (segs[2].start_s, segs[2].end_s) == (14.36, 18.76)
    assert [s.speaker_label for s in segs] == ["S01", "S02", "S01"]
    assert segs[0].text == "Welcome everyone"


def test_official_format_mmss():
    segs = parse_moss_transcript("[00:01.20][S01]第一句话[00:03.50][02:00.10][S02]第二句[02:03.00]")
    assert segs[0].start_s == 1.2 and segs[0].end_s == 3.5
    assert segs[1].start_s == 120.1 and segs[1].end_s == 123.0


def test_legacy_format_still_supported():
    """旧格式（end=next start）不能破坏，兼容之前的数据。"""
    segs = parse_moss_transcript("[0.00][S01]甲说话，[2.32][S02]乙说话。")
    assert len(segs) == 2
    assert segs[0].speaker_label == "S01"
    assert segs[0].end_s == pytest.approx(2.32)


def test_bad_end_time_raises():
    with pytest.raises(MossParseError):
        parse_moss_transcript("[5.00][S01]时间倒流[2.00]")


def test_no_timestamp_raises():
    with pytest.raises(MossParseError):
        parse_moss_transcript("完全没有时间戳的一段话")


def test_empty():
    assert parse_moss_transcript("") == []


def test_verbose_json_speaker_prefix_stripped():
    """SGLang verbose_json: segment.text 带 [Sxx] 前缀，须拆出来。"""
    from transcriptum.adapters.http_asr import OpenAiAudioAsr

    ad = OpenAiAudioAsr(base_url="http://localhost:8000/v1", api_key="x")
    payload = {
        "segments": [
            {"start": 0.0, "end": 1.66, "text": "[S01]Welcome everyone", "avg_logprob": -0.2},
            {"start": 12.26, "end": 13.81, "text": "[S02]The pipeline is ready", "avg_logprob": -0.3},
        ]
    }

    class FakeResp:
        status_code = 200
        def json(self): return payload

    import transcriptum.adapters.http_asr as mod
    orig = mod.requests.post
    mod.requests.post = lambda *a, **k: FakeResp()
    try:
        import tempfile, wave, pathlib
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "a.wav"
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 1600)
            jid = ad.submit(p, [])
            segs = ad.fetch(jid)
    finally:
        mod.requests.post = orig

    assert segs[0].text == "Welcome everyone"
    assert segs[0].speaker_label == "S01"
    assert segs[1].speaker_label == "S02"
