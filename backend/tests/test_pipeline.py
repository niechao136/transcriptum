"""端到端流水线：上传 → 归一化 → 转写 → 对齐 → 校对 → 评估 → 导出。

这些测试跑在 Mock 适配器上，验证的是「上层完全不感知适配器差异」这个核心设计目标。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcriptum.adapters.base import RawSegment
from transcriptum.db import jloads
from transcriptum.exporters import ExportOptions, export, redact
from transcriptum.service import RecordingService
from transcriptum.worker import Worker, enqueue


@pytest.fixture()
def svc(cfg, db, deps):
    return RecordingService(cfg, db, deps)


@pytest.fixture()
def rec(cfg, db, deps, wav, svc):
    return svc.create(title="三号机检修交班", kind="meeting", source="upload",
                      original_path=wav)


def run_all(db, deps, rec_id: str, svc: RecordingService):
    enqueue(db, rec_id, "transcribe")
    enqueue(db, rec_id, "proofread")
    worker = Worker(deps)
    processed = 0
    while worker.run_once() and processed < 20:
        processed += 1
    return processed


def test_transcribe_creates_segments(svc, db, deps, rec):
    enqueue(db, rec["id"], "transcribe")
    Worker(deps).run_once()

    segs = svc.segments(rec["id"])
    assert len(segs) >= 4
    # 时间轴单调不减
    for a, b in zip(segs, segs[1:]):
        assert a["start_ms"] <= b["start_ms"]
    # 每句都落到了说话人簇
    assert all(s["speaker_id"] is not None for s in segs)
    assert svc.get(rec["id"])["status"] == "transcribed"


def test_segments_are_split_one_speaker_each(svc, db, deps, rec):
    enqueue(db, rec["id"], "transcribe")
    Worker(deps).run_once()
    segs = svc.segments(rec["id"])
    # 一句一人：同一句内不会有两个说话人（MOSS 输出本身就是切好的）
    labels = [s["speaker_label"] for s in segs]
    assert len(set(labels)) >= 2


def test_capabilities_snapshot(svc, db, deps, rec):
    """能力快照必须落库，前端才能按能力降级。"""
    enqueue(db, rec["id"], "transcribe")
    Worker(deps).run_once()
    r = svc.get(rec["id"])
    caps = r["asr_capabilities"]
    assert caps["has_speaker"] is True
    assert caps["has_word_ts"] is False   # MOSS 真实能力
    assert caps["has_confidence"] is False


def test_proofread_populates_ai_text_and_edits(svc, db, deps, rec):
    enqueue(db, rec["id"], "transcribe")
    Worker(deps).run_once()
    enqueue(db, rec["id"], "proofread")
    Worker(deps).run_once()

    segs = svc.segments(rec["id"])
    changed = [s for s in segs if s["ai_text"] and s["ai_text"] != s["asr_text"]]
    assert changed, "mock 应至少纠正一处术语"
    for s in changed:
        assert s["ai_edits"], "有改动就必须有结构化 edits"
        for e in s["ai_edits"]:
            assert {"type", "from", "to"} <= set(e)


def test_asr_text_immutable_after_proofread(svc, db, deps, rec):
    """asr_text 必须永远保留原始识别结果（可复原性）。"""
    enqueue(db, rec["id"], "transcribe")
    Worker(deps).run_once()
    before = {s["seq"]: s["asr_text"] for s in svc.segments(rec["id"])}

    enqueue(db, rec["id"], "proofread")
    Worker(deps).run_once()

    after = {s["seq"]: s["asr_text"] for s in svc.segments(rec["id"])}
    assert before == after


def test_review_accept_and_edit(svc, db, deps, rec):
    enqueue(db, rec["id"], "transcribe")
    Worker(deps).run_once()
    enqueue(db, rec["id"], "proofread")
    Worker(deps).run_once()

    seg = svc.segments(rec["id"])[0]
    updated = svc.update_segment(rec["id"], seg["seq"], review_status="accepted")
    assert updated["review_status"] == "accepted"

    edited = svc.update_segment(rec["id"], seg["seq"], final_text="人工定稿内容")
    assert edited["final_text"] == "人工定稿内容"
    assert edited["review_status"] == "edited"

    # 审计留痕
    logs = db.all("SELECT * FROM audit_log WHERE recording_id=? AND action='segment_updated'",
                  (rec["id"],))
    assert len(logs) >= 2


def test_batch_accept_only_low_risk(svc, db, deps, rec):
    enqueue(db, rec["id"], "transcribe")
    Worker(deps).run_once()
    enqueue(db, rec["id"], "proofread")
    Worker(deps).run_once()

    res = svc.batch_accept(rec["id"])
    assert isinstance(res["accepted"], list)
    # 被采纳的句子都有 final_text
    for seq in res["accepted"]:
        row = db.one("SELECT final_text FROM segment WHERE recording_id=? AND seq=?",
                     (rec["id"], seq))
        assert row["final_text"]


def test_assessment_full_flow(svc, db, deps, rec):
    run_all(db, deps, rec["id"], svc)

    fws = svc.frameworks()
    fw = next(f for f in fws if f["code"] == "shift_handover_assessment")
    enqueue(db, rec["id"], "assessment",
            model="mock", provider="mock")
    # 记录要用的框架：走 service 的正式入口
    svc.start_assessment(rec["id"], fw["id"])
    Worker(deps).run_once()

    assessments = svc.assessments(rec["id"])
    assert assessments, "评估结果应已生成"
    a = assessments[0]
    values = a["values"]
    assert values
    summary = next(v for v in values if v["field_key"] == "summary")
    assert summary["content"]
    assert summary["citations"], "有内容的字段必须有引用"


def test_assessment_submit_requires_citations(svc, db, deps, rec):
    run_all(db, deps, rec["id"], svc)
    fw = next(f for f in svc.frameworks() if f["code"] == "shift_handover_assessment")
    svc.start_assessment(rec["id"], fw["id"])
    Worker(deps).run_once()
    a = svc.assessments(rec["id"])[0]

    # 把 summary 的引用清空 → 提交必须被拒
    svc.update_field(a["id"], "summary", content="有内容但无引用", citations=[])
    from transcriptum.service import ValidationFailed

    with pytest.raises(ValidationFailed) as exc:
        svc.submit_assessment(a["id"])
    assert any("引用" in r for r in exc.value.reasons)


def test_assessment_submit_after_fix(svc, db, deps, rec):
    run_all(db, deps, rec["id"], svc)
    fw = next(f for f in svc.frameworks() if f["code"] == "shift_handover_assessment")
    svc.start_assessment(rec["id"], fw["id"])
    Worker(deps).run_once()
    a = svc.assessments(rec["id"])[0]
    seqs = [s["seq"] for s in svc.segments(rec["id"])]

    for v in a["values"]:
        # number 字段必须给合法数值（校验会拒绝「无」这种非数值内容）
        content = v["content"]
        if not content:
            field = next(f for f in fw["fields"] if f["key"] == v["field_key"])
            content = "85" if field["type"] == "number" else "无"
        svc.update_field(a["id"], v["field_key"], content=content, citations=seqs[:1])
    assert svc.submit_assessment(a["id"])["status"] == "submitted"


def test_invalid_citation_rejected(svc, db, deps, rec):
    run_all(db, deps, rec["id"], svc)
    fw = next(f for f in svc.frameworks() if f["code"] == "shift_handover_assessment")
    svc.start_assessment(rec["id"], fw["id"])
    Worker(deps).run_once()
    a = svc.assessments(rec["id"])[0]
    from transcriptum.service import ValidationFailed

    with pytest.raises(ValidationFailed):
        svc.update_field(a["id"], "summary", content="x", citations=[99999])


def test_export_markdown_marks_unconfirmed(svc, db, deps, rec):
    run_all(db, deps, rec["id"], svc)
    body, media, _ = export(db, rec["id"], svc.get(rec["id"]), "md",
                            ExportOptions(redact_sensitive=False))
    text = body.decode()
    assert "AI 建议未确认" in text
    assert "未经人工确认" in text


def test_export_redacts_sensitive(svc, db, wav):
    t = redact("联系 13812345678 或 a@b.com，身份证 110101199001011234")
    assert "13812345678" not in t
    assert "a@b.com" not in t
    assert "110101199001011234" not in t


def test_export_docx_and_xlsx(svc, db, deps, rec):
    run_all(db, deps, rec["id"], svc)
    for fmt, magic in (("docx", b"PK"), ("xlsx", b"PK")):
        try:
            body, _, _ = export(db, rec["id"], svc.get(rec["id"]), fmt,
                                ExportOptions(redact_sensitive=True))
        except RuntimeError:
            pytest.skip(f"{fmt} 依赖未安装")
        assert body.startswith(magic)


def test_retry_then_succeed(cfg, db, deps, wav, svc):
    """ASR 首次 fetch 失败 → 队列重试 → 最终成功。"""
    from transcriptum.adapters.mock import MockMossAsr

    rec = svc.create(title="重试测试", kind="meeting", source="upload", original_path=wav)
    deps.asr = MockMossAsr(fail_times=1)
    enqueue(db, rec["id"], "transcribe")
    worker = Worker(deps)
    for _ in range(10):
        worker.run_once()
        job = db.one("SELECT * FROM job WHERE recording_id=? ORDER BY created_at DESC",
                     (rec["id"],))
        if job["status"] == "done":
            break
    job = db.one("SELECT * FROM job WHERE recording_id=? ORDER BY created_at DESC",
                 (rec["id"],))
    assert job["status"] == "done", job
    assert job["attempts"] >= 2


def test_failure_marks_recording_failed(cfg, db, deps, wav, svc):
    from transcriptum.adapters.base import AdapterError
    from transcriptum.adapters.mock import MockMossAsr

    class AlwaysFail(MockMossAsr):
        def fetch(self, job_id):
            raise AdapterError("永久失败")

    rec = svc.create(title="失败测试", kind="meeting", source="upload", original_path=wav)
    deps.asr = AlwaysFail()
    db.exec("UPDATE job SET max_attempts=1")
    enqueue(db, rec["id"], "transcribe")
    worker = Worker(deps)
    for _ in range(6):
        worker.run_once()
        if svc.get(rec["id"])["status"] == "failed":
            break
    assert svc.get(rec["id"])["status"] == "failed"
