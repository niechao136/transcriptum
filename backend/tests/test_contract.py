"""接口契约回归测试：锁定真实返回形状，防止前后端脱节。

这些用例来自实际跑端到端时踩到的坑（详情不含 segments、创建评估异步、
PATCH 静默忽略 asr_text、无审计接口），每一条都对应一次真实故障。
"""

from __future__ import annotations

import io
import wave

import pytest
from fastapi.testclient import TestClient

from transcriptum.api import create_app


def _wav_bytes(seconds: float = 2.0, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * seconds))
    return buf.getvalue()


@pytest.fixture()
def client(cfg):
    return TestClient(create_app(cfg))


def _create(client) -> dict:
    r = client.post(
        "/api/recordings",
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"title": "契约测试", "kind": "meeting", "source": "upload", "consent": "true"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _transcribe(client, rec_id: str):
    """走真实队列入口：enqueue → claim → run，与生产路径一致。"""
    from transcriptum.worker import _run_transcribe, claim_next, enqueue

    db = client.app.state.db
    with db.tx():
        enqueue(db, rec_id, "transcribe")
    job = claim_next(db)
    assert job is not None and job["recording_id"] == rec_id
    _run_transcribe(client.app.state.worker_deps, job)
    return client.get(f"/api/recordings/{rec_id}").json()


def _run_queue_once(client) -> bool:
    """跑一个排队 job；返回是否有 job 被执行。"""
    from transcriptum.worker import claim_next, run_job

    db = client.app.state.db
    job = claim_next(db)
    if job is None:
        return False
    run_job(client.app.state.worker_deps, job)
    return True


def test_detail_includes_segments_after_transcribe(client):
    """详情必须自带 segments，否则校对台要发两次请求、且容易读到空列表。"""
    rec = _create(client)
    assert rec["segments"] == []
    detail = _transcribe(client, rec["id"])
    assert detail["status"] == "transcribed"
    assert len(detail["segments"]) > 0
    first = detail["segments"][0]
    for key in ("seq", "start_ms", "end_ms", "asr_text", "speaker_label"):
        assert key in first
    # MOSS 无字级时间戳：能力必须如实反映
    assert detail["asr_capabilities"]["has_word_ts"] is False


def test_lists_are_paginated_envelopes(client):
    """所有列表接口统一 {items,total}，前端不能按裸数组处理。"""
    for path in ("/api/recordings", "/api/frameworks"):
        body = client.get(path).json()
        assert "items" in body, path
    body = client.get("/api/recordings").json()
    assert "total" in body


def test_asr_text_immutable_rejected_explicitly(client):
    """PATCH asr_text 必须显式 400，绝不能静默成功。"""
    rec = _create(client)
    detail = _transcribe(client, rec["id"])
    seq = detail["segments"][0]["seq"]
    r = client.patch(f"/api/recordings/{rec['id']}/segments/{seq}",
                     json={"asr_text": "篡改内容"})
    # 422 Unprocessable Entity：字段不可改属于语义校验失败，不是语法错误
    assert r.status_code == 422
    assert "asr_text" in r.text
    # 原文未被改动
    after = client.get(f"/api/recordings/{rec['id']}").json()
    assert after["segments"][0]["asr_text"] == detail["segments"][0]["asr_text"]


def test_create_assessment_is_async_and_returns_full_payload(client):
    rec = _create(client)
    _transcribe(client, rec["id"])
    fws = client.get("/api/frameworks").json()["items"]
    fw = next(f for f in fws if f["code"] == "shift_handover_assessment")

    r = client.post(f"/api/recordings/{rec['id']}/assessments",
                    json={"framework_id": fw["id"]})
    assert r.status_code == 200
    assert "job_id" in r.json()  # 异步：先返回 job

    # 异步：手动驱动队列（测试环境不起 worker 线程）
    for _ in range(6):
        if _run_queue_once(client):
            break
    items = client.get(f"/api/recordings/{rec['id']}/assessments").json()["items"]
    assert items, "评估未生成"
    assert items, "评估未生成"
    a = items[0]
    for key in ("values", "framework", "submittable", "blockers"):
        assert key in a, key
    assert isinstance(a["submittable"], bool)
    # 引用全部有效时应当可提交
    assert a["submittable"] is True
    assert a["blockers"] == []


def test_audit_endpoint_exists_and_records(client):
    rec = _create(client)
    _transcribe(client, rec["id"])
    r = client.get(f"/api/recordings/{rec['id']}/audit")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert any(i["action"] == "recording_created" for i in body["items"])
