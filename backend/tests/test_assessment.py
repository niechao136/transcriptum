"""评估模块契约测试（§6）。

核心约束：无引用不可提交；框架版本快照；类型校验；缺失不编造。
"""

from __future__ import annotations

import json

import pytest

from transcriptum.assessment import (
    BUILTIN_FRAMEWORKS,
    AssessmentError,
    FieldDef,
    FieldValue,
    FrameworkDef,
    can_submit,
    parse_assessment,
    validate_value,
)
from transcriptum.adapters.mock import MockLlm


class Scripted:
    name = "s"
    capabilities = type(MockLlm.capabilities)(json_mode=True)

    def __init__(self, payload):
        self.payload = payload

    def chat(self, system, user, *, json_mode=False):
        return self.payload if isinstance(self.payload, str) else json.dumps(
            self.payload, ensure_ascii=False)


SEGS = [
    {"seq": 1, "text": "三号机油压正常。", "speaker_label": "A"},
    {"seq": 2, "text": "可以合闸。", "speaker_label": "B"},
    {"seq": 3, "text": "补上安全阀校验记录。", "speaker_label": "A"},
]

FW = FrameworkDef(
    code="t", name="测试", version=1,
    fields=[
        FieldDef("summary", "摘要", "text", True),
        FieldDef("score", "评分", "number", False),
        FieldDef("level", "等级", "enum", False, options=["优", "良"]),
    ],
)


def test_builtin_frameworks_defined():
    codes = {f.code for f in BUILTIN_FRAMEWORKS}
    assert "shift_handover_assessment" in codes
    assert all(f.fields for f in BUILTIN_FRAMEWORKS)


def test_framework_roundtrip():
    d = FW.as_dict()
    back = FrameworkDef.from_dict(d)
    assert back.code == FW.code
    assert [f.key for f in back.fields] == [f.key for f in FW.fields]


def test_valid_assessment_parsed():
    llm = Scripted({"fields": [
        {"key": "summary", "content": "油压正常可合闸", "citations": [1, 2],
         "confidence": 0.9},
        {"key": "score", "content": "88", "citations": [1], "confidence": 0.8},
        {"key": "level", "content": "优", "citations": [1], "confidence": 0.8},
    ]})
    out = parse_assessment(FW, SEGS, llm=llm)
    assert all(v.status == "ok" for v in out)


def test_missing_citation_blocks_status():
    llm = Scripted({"fields": [
        {"key": "summary", "content": "有结论但无引用", "citations": []},
    ]})
    out = parse_assessment(FW, SEGS, llm=llm)
    summary = next(v for v in out if v.key == "summary")
    assert summary.status == "missing_citation"
    assert summary.citations == []


def test_invented_citation_dropped():
    """模型给了不存在的 seq → 过滤掉，结果变成无引用。"""
    llm = Scripted({"fields": [
        {"key": "summary", "content": "内容", "citations": [999]},
    ]})
    out = parse_assessment(FW, SEGS, llm=llm)
    summary = next(v for v in out if v.key == "summary")
    assert summary.citations == []
    assert summary.status == "missing_citation"


def test_missing_field_becomes_not_extracted():
    llm = Scripted({"fields": [
        {"key": "summary", "content": "x", "citations": [1]},
    ]})
    out = parse_assessment(FW, SEGS, llm=llm)
    score = next(v for v in out if v.key == "score")
    assert score.status == "not_extracted"
    assert score.content == ""


def test_invalid_number_rejected():
    llm = Scripted({"fields": [
        {"key": "score", "content": "八十八分", "citations": [1]},
    ]})
    out = parse_assessment(FW, SEGS, llm=llm)
    score = next(v for v in out if v.key == "score")
    assert score.status == "invalid_value"


def test_invalid_enum_rejected():
    llm = Scripted({"fields": [
        {"key": "level", "content": "超优", "citations": [1]},
    ]})
    out = parse_assessment(FW, SEGS, llm=llm)
    level = next(v for v in out if v.key == "level")
    assert level.status == "invalid_value"


def test_bad_json_raises():
    llm = Scripted("这不是 JSON")
    with pytest.raises(AssessmentError):
        parse_assessment(FW, SEGS, llm=llm)


def test_fenced_json_accepted():
    payload = json.dumps({"fields": [
        {"key": "summary", "content": "x", "citations": [1]}]}, ensure_ascii=False)
    llm = Scripted(f"```json\n{payload}\n```")
    out = parse_assessment(FW, SEGS, llm=llm)
    assert next(v for v in out if v.key == "summary").content == "x"


def test_missing_fields_array_raises():
    llm = Scripted({"result": []})
    with pytest.raises(AssessmentError):
        parse_assessment(FW, SEGS, llm=llm)


def test_can_submit_requires_citations():
    vals = [
        FieldValue("summary", "有内容没引用", [], status="missing_citation"),
        FieldValue("score", "88", [1]),
        FieldValue("level", "优", [1]),
    ]
    ok, reasons = can_submit(vals, FW)
    assert ok is False
    assert any("缺少引用" in r for r in reasons)


def test_can_submit_requires_required_field():
    vals = [
        FieldValue("summary", "", [], status="not_extracted"),
        FieldValue("score", "88", [1]),
    ]
    ok, reasons = can_submit(vals, FW)
    assert ok is False
    assert any("摘要" in r for r in reasons)


def test_can_submit_passes_when_all_good():
    vals = [
        FieldValue("summary", "结论", [1]),
        FieldValue("score", "88", [1]),
        FieldValue("level", "优", [1]),
    ]
    ok, reasons = can_submit(vals, FW)
    assert ok is True
    assert reasons == []


def test_validate_value_types():
    assert validate_value(FieldDef("n", "n", "number"), "12.5") is True
    assert validate_value(FieldDef("n", "n", "number"), "十二") is False
    assert validate_value(FieldDef("e", "e", "enum", options=["A"]), "A") is True
    assert validate_value(FieldDef("e", "e", "enum", options=["A"]), "B") is False
    assert validate_value(FieldDef("t", "t", "text"), "") is False


def test_mock_llm_assessment_end_to_end():
    """mock 的引用值覆盖 1..4，这里配 4 句完整数据做端到端。"""
    segs = SEGS + [{"seq": 4, "text": "安全阀校验记录待归档。", "speaker_label": "A"}]
    llm = MockLlm()
    out = parse_assessment(BUILTIN_FRAMEWORKS[0], segs, llm=llm)
    keys = {v.key for v in out}
    assert "summary" in keys
    summary = next(v for v in out if v.key == "summary")
    assert summary.status == "ok"
    assert summary.citations  # 引用被正确保留
    assert all(v.status != "invalid_value" for v in out)
