"""保守校对契约测试（§4.3）。

重点验证「不许编」的几条硬约束：
- 受保护 token（剂量/数值/缩写）不得被改；
- 模型改写整句时不采纳，回退逐处替换；
- 改动幅度过大不采纳；
- 模型/网络失败时保留原文并标待复核。
"""

from __future__ import annotations

import json

import pytest

from transcriptum.adapters.base import AdapterError
from transcriptum.adapters.mock import MockLlm
from transcriptum.proofread import (
    Edit,
    ProofResult,
    build_user_prompt,
    is_low_risk,
    proofread_segment,
    protected_tokens,
)


class ScriptedLlm:
    """按脚本返回固定 JSON，便于构造边界场景。"""

    name = "scripted"
    capabilities = type(MockLlm.capabilities)(json_mode=True)

    def __init__(self, payload):
        self.payload = payload

    def chat(self, system, user, *, json_mode=False):
        if isinstance(self.payload, Exception):
            raise self.payload
        return json.dumps(self.payload, ensure_ascii=False) if not isinstance(
            self.payload, str) else self.payload


def test_protected_tokens_extracted():
    toks = protected_tokens("给予 5mg 剂量，血压 120，复查 CT 与 HbA1c")
    assert "5mg" in toks
    assert "120" in toks
    assert "CT" in toks
    assert "HbA1c" in toks


def test_protected_dose_not_changed():
    """模型想把 5mg 改成 50mg —— 必须被丢弃。"""
    llm = ScriptedLlm({
        "corrected_text": "给予 50mg 剂量",
        "edits": [{"type": "replace", "from": "5mg", "to": "50mg",
                   "reason": "剂量调整", "confidence": 0.95}],
    })
    res = proofread_segment("给予 5mg 剂量", llm=llm)
    assert res.corrected_text == "给予 5mg 剂量"
    assert res.edits == []
    assert "protected" in (res.warning or "")


def test_rewrite_is_rejected():
    """模型把短句改写成一句长话 —— 以逐处替换为准，不采信 corrected_text。"""
    llm = ScriptedLlm({
        "corrected_text": "这是一段完全不存在的总结内容。",
        "edits": [{"type": "replace", "from": "合砸", "to": "合闸",
                   "reason": "同音字", "confidence": 0.92}],
    })
    res = proofread_segment("可以合砸了", llm=llm)
    assert res.corrected_text == "可以合闸了"
    assert "model_rewrote" in (res.warning or "")


def test_edit_ratio_too_high_not_accepted():
    src = "好"  # 1 字
    llm = ScriptedLlm({
        "corrected_text": "好的没问题",
        "edits": [{"type": "replace", "from": "好", "to": "好的没问题",
                   "reason": "补全", "confidence": 0.9}],
    })
    res = proofread_segment(src, llm=llm)
    assert res.accepted is False
    assert res.requires_review is True
    assert "edit_ratio_too_high" in (res.warning or "")


def test_no_edits_when_clean():
    llm = ScriptedLlm({"corrected_text": "", "edits": []})
    res = proofread_segment("这台机运行正常。", llm=llm)
    assert res.edits == []
    assert res.accepted is False
    assert res.corrected_text == "这台机运行正常。"


def test_llm_failure_keeps_source():
    llm = ScriptedLlm(AdapterError("boom"))
    res = proofread_segment("原文保持不变。", llm=llm)
    assert res.corrected_text == "原文保持不变。"
    assert res.requires_review is True
    assert "llm_unavailable" in (res.warning or "")


def test_malformed_json_keeps_source():
    llm = ScriptedLlm("这不是 JSON")
    res = proofread_segment("原文。", llm=llm)
    assert res.corrected_text == "原文。"
    assert res.requires_review is True


def test_unlocatable_edit_dropped():
    """模型给了原文里不存在的 from —— 不能凭空插入内容。"""
    llm = ScriptedLlm({
        "corrected_text": "油压正常。另外请补一条备注。",
        "edits": [{"type": "insert", "from": "不存在的片段", "to": "备注",
                   "reason": "补充", "confidence": 0.9}],
    })
    res = proofread_segment("油压正常。", llm=llm)
    assert res.edits == []
    assert res.corrected_text == "油压正常。"


def test_low_asr_confidence_marks_review():
    llm = ScriptedLlm({
        "corrected_text": "可以合闸了",
        "edits": [{"type": "replace", "from": "合砸", "to": "合闸",
                   "reason": "同音字", "confidence": 0.95}],
    })
    res = proofread_segment("可以合砸了", llm=llm, asr_confidence=0.3)
    assert res.requires_review is True


def test_low_risk_gate():
    ok = ProofResult(
        corrected_text="x", edits=[Edit("replace", "a", "b", "", 0.95)],
        accepted=True, requires_review=True, warning=None)
    assert is_low_risk(ok) is True
    low = ProofResult(
        corrected_text="x", edits=[Edit("replace", "a", "b", "", 0.60)],
        accepted=True, requires_review=True, warning=None)
    assert is_low_risk(low) is False
    warned = ProofResult(
        corrected_text="x", edits=[Edit("replace", "a", "b", "", 0.99)],
        accepted=True, requires_review=True, warning="any")
    assert is_low_risk(warned) is False


def test_glossary_prompt_contains_variants():
    prompt = build_user_prompt("油压正常", [], [("合闸", ["合砸", "合扎"])])
    assert "合砸" in prompt and "合闸" in prompt
    assert "<source>油压正常</source>" in prompt


def test_context_included_but_bounded():
    prompt = build_user_prompt("句子", ["上文一", "上文二"], [])
    assert "上文一" in prompt and "不要校对" in prompt


def test_mock_llm_end_to_end():
    llm = MockLlm()
    res = proofread_segment("先看机油压，油压正常就可以合砸。", llm=llm)
    assert "合闸" in res.corrected_text
    assert res.edits
    assert all(e.confidence >= 0.85 for e in res.edits)
    assert is_low_risk(res) is True
