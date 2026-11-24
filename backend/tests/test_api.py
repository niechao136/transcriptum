"""HTTP 接口测试（§9），重点验证 Range 播放与各项 4xx 语义。"""

from __future__ import annotations

import io
import wave

import pytest
from fastapi.testclient import TestClient

from transcriptum.api import create_app
from transcriptum.worker import Worker


def _wav_bytes(seconds: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * seconds)
    return buf.getvalue()


@pytest.fixture()
def client(cfg):
    app = create_app(cfg)
    return TestClient(app)


def _worker_for(client) -> Worker:
    """复用 app 已构建的 worker 依赖，避免重建适配器导致 mock 状态不一致。"""
    return Worker(client.app.state.worker_deps)


def _run_transcribe(client, rec_id):
    """同步跑一遍转写阶段，让归一化音频真实落盘。"""
    w = _worker_for(client)
    w.load_capabilities(rec_id)
    rec = client.app.state.db.one("SELECT * FROM recording WHERE id=?", (rec_id,))
    w.do_transcribe(rec_id, {"recording": rec})
    return client.app.state.db.one("SELECT * FROM recording WHERE id=?", (rec_id,))


def _create(client, title="接口测试会议"):
    files = {"file": ("a.wav", _wav_bytes(), "audio/wav")}
    data = {"title": title, "kind": "meeting", "source": "upload", "consent": "true"}
    r = client.post("/api/recordings", files=files, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def test_health_reports_capabilities(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["asr"]["capabilities"]["has_speaker"] is True
    # MOSS 关键能力：无字级时间戳、无置信度 —— UI 必须据此降级
    assert body["asr"]["capabilities"]["has_word_ts"] is False
    assert body["asr"]["capabilities"]["has_confidence"] is False


def test_create_and_list_recording(client):
    rec = _create(client)
    assert rec["status"] == "uploaded"
    assert rec["consent_at"]
    r = client.get("/api/recordings")
    assert r.status_code == 200
    assert r.json()["total"] >= 1


def test_create_requires_file(client):
    r = client.post("/api/recordings",
                    data={"title": "t", "kind": "meeting", "source": "upload"})
    assert r.status_code == 422


def test_bad_kind_rejected(client):
    files = {"file": ("a.wav", _wav_bytes(), "audio/wav")}
    r = client.post("/api/recordings", files=files,
                    data={"title": "t", "kind": "nope", "source": "upload"})
    assert r.status_code == 422


def test_missing_recording_404(client):
    assert client.get("/api/recordings/nope").status_code == 404


def test_unknown_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_pipeline_via_api(client, cfg):
    rec = _create(client)
    r = client.post(f"/api/recordings/{rec['id']}/transcribe")
    assert r.status_code == 200
    Worker(WorkerDeps_for(cfg, client)).run_once()

    segs = client.get(f"/api/recordings/{rec['id']}/segments").json()["items"]
    assert segs, "转写后应有句子"

    r = client.post(f"/api/recordings/{rec['id']}/proofread")
    assert r.status_code == 200
    Worker(WorkerDeps_for(cfg, client)).run_once()

    segs = client.get(f"/api/recordings/{rec['id']}/segments").json()["items"]
    assert any(s["ai_text"] for s in segs)


def WorkerDeps_for(cfg, client):  # noqa: N802 - 测试内构造，复用 app 的 db
    from transcriptum.worker import WorkerDeps

    db = client.app.state.db
    return WorkerDeps.build(cfg, db)


# ---------------- Range 播放 ----------------
def _prepare_audio(client, cfg, seconds: int = 2) -> str:
    """跑一次转写让 worker 生成归一化音频，返回录音 id。"""
    rec = _create(client)
    client.post(f"/api/recordings/{rec['id']}/transcribe")
    Worker(WorkerDeps_for(cfg, client)).run_once()
    return rec["id"]


def test_audio_range_206(client, cfg):
    rec_id = _prepare_audio(client, cfg)
    r = client.get(f"/api/recordings/{rec_id}/audio",
                   headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206
    assert r.headers["content-range"].startswith("bytes 0-1023/")
    assert int(r.headers["content-length"]) == 1024
    assert r.headers["accept-ranges"] == "bytes"


def test_audio_full_request_200(client, cfg):
    rec_id = _prepare_audio(client, cfg, seconds=1)
    r = client.get(f"/api/recordings/{rec_id}/audio")
    assert r.status_code == 200
    assert r.headers["accept-ranges"] == "bytes"


def test_audio_range_suffix(client, cfg):
    rec_id = _prepare_audio(client, cfg, seconds=1)
    r = client.get(f"/api/recordings/{rec_id}/audio",
                   headers={"Range": "bytes=-100"})
    assert r.status_code == 206
    assert int(r.headers["content-length"]) == 100


def test_audio_range_invalid_416(client, cfg):
    rec_id = _prepare_audio(client, cfg, seconds=1)
    r = client.get(f"/api/recordings/{rec_id}/audio",
                   headers={"Range": "bytes=999999999-"})
    assert r.status_code == 416


def test_audio_missing_file_404(client):
    """未转写 → 没有归一化音频 → 404（而不是返回原始文件）。"""
    rec = _create(client)
    path = client.app.state.svc.get(rec["id"]).get("audio_path")
    assert not path, "创建阶段不应预写 audio_path"
    r = client.get(f"/api/recordings/{rec['id']}/audio")
    assert r.status_code == 404


# ---------------- 评估 ----------------
def test_frameworks_seeded(client):
    r = client.get("/api/frameworks")
    codes = {f["code"] for f in r.json()["items"]}
    assert "shift_handover_assessment" in codes
    assert "safety_briefing_assessment" in codes


def test_create_framework_validation(client):
    assert client.post("/api/frameworks", json={"code": "", "name": "x",
                                                "fields": []}).status_code == 422
    bad = {"code": "c1", "name": "n", "fields": [{"key": "a", "label": "A", "type": "enum"}]}
    assert client.post("/api/frameworks", json=bad).status_code == 422


def test_create_framework_versions(client):
    body = {"code": "custom1", "name": "自定义",
            "fields": [{"key": "a", "label": "A", "type": "text"}]}
    r1 = client.post("/api/frameworks", json=body)
    assert r1.status_code == 200
    r2 = client.post("/api/frameworks", json=body)
    assert r2.status_code == 200
    assert r2.json()["version"] == r1.json()["version"] + 1


def test_export_markdown(client):
    rec = _create(client)
    r = client.get(f"/api/recordings/{rec['id']}/export?fmt=md")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    text = r.content.decode("utf-8")
    assert "AI 建议未确认" in text  # 合规提示必须存在
