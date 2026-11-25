"""端到端冒烟：真实起 HTTP 服务 → 上传 → 转写 → 校对 → 评估 → 导出。

这是设计文档 §11 的 M0/M1 验收脚本，也是交付时给用户"自己再跑一遍"的入口。
用法：python scripts/smoke_e2e.py [--base http://127.0.0.1:8080]
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import wave
from pathlib import Path

import httpx


def make_wav(seconds: float = 3.0, sr: int = 16000) -> bytes:
    # 目标机验收：优先用真实语音（模拟纯音无法触发 mock LLM 校对/评估）。
    import os
    real = os.environ.get("TRANSCRIPTUM_SMOKE_WAV")
    if real and Path(real).exists():
        return Path(real).read_bytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * seconds))
    return buf.getvalue()


def _wait_assessment(c, rec_id: str, timeout_s: float = 120.0) -> dict | None:
    """评估是异步 job：轮询到出现带 values 的评估结果。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        items = c.get(f"/api/recordings/{rec_id}/assessments").json().get("items") or []
        for a in items:
            if a.get("values"):
                return a
        time.sleep(1.5)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    ok = 0
    fail = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fail += 1
            print(f"  FAIL  {name} {detail}")

    with httpx.Client(base_url=base, timeout=60) as c:
        print("[1] 健康检查")
        r = c.get("/api/health")
        check("health 200", r.status_code == 200, r.text[:200])
        h = r.json()
        check("能力已显式声明", "has_word_ts" in h["asr"]["capabilities"])
        caps = h["asr"]["capabilities"]
        print(f"      ASR={h['asr']['adapter']} has_speaker={caps['has_speaker']}"
              f" has_word_ts={caps['has_word_ts']}")

        print("[2] 上传录音")
        files = {"file": ("smoke.wav", make_wav(), "audio/wav")}
        data = {"title": "冒烟测试会议（中文标题）", "kind": "meeting",
                "source": "upload", "consent": "true"}
        r = c.post("/api/recordings", files=files, data=data)
        check("上传 200", r.status_code == 200, r.text[:300])
        rec = r.json()
        rec_id = rec["id"]
        print(f"      recording_id={rec_id}")

        print("[3] 转写（提交 job + 轮询）")
        r = c.post(f"/api/recordings/{rec_id}/transcribe")
        check("转写受理", r.status_code == 200, r.text[:300])
        rec = _wait(c, rec_id, {"transcribed", "proofread", "done"}, {"failed"})
        check("转写完成", rec["status"] in {"transcribed", "proofread", "done"},
              f"status={rec['status']} err={rec.get('error_msg')}")
        segs = rec.get("segments") or []
        check("产出句子", len(segs) > 0, f"len={len(segs)}")
        if not segs:
            return _report(ok, fail)
        check("句子按时间递增", all(
            segs[i]["start_ms"] <= segs[i + 1]["start_ms"] for i in range(len(segs) - 1)))
        check("一句一人（speaker 非空）", all(s.get("speaker_label") for s in segs))
        check("asr_text 已落库", all(s["asr_text"] for s in segs))

        print("[4] 音频 Range 播放")
        r = c.get(f"/api/recordings/{rec_id}/audio",
                  headers={"Range": "bytes=0-99"})
        check("Range 206", r.status_code == 206, f"got {r.status_code}")
        check("Content-Range 正确", r.headers.get("content-range", "").startswith("bytes 0-99/"),
              r.headers.get("content-range", ""))
        check("Accept-Ranges", r.headers.get("accept-ranges") == "bytes")

        print("[5] AI 校对")
        r = c.post(f"/api/recordings/{rec_id}/proofread")
        check("校对受理", r.status_code == 200, r.text[:300])
        rec = _wait(c, rec_id, {"proofread", "done"}, {"failed"})
        segs = rec.get("segments") or []
        check("校对完成", rec["status"] in {"proofread", "done"}, rec.get("error_msg"))
        with_edits = [s for s in segs if s.get("ai_edits")]
        check("产生了 ai_edits", len(with_edits) > 0, f"{len(with_edits)}/{len(segs)}")

        print("[6] 人类确认一句（按 seq 定位）")
        seg = segs[0]
        r = c.patch(f"/api/recordings/{rec_id}/segments/{seg['seq']}",
                    json={"review_status": "accepted"})
        check("确认成功", r.status_code == 200, r.text[:200])
        rec = c.get(f"/api/recordings/{rec_id}").json()
        prog = rec.get("review_progress") or {}
        check("进度已更新", prog.get("reviewed", 0) >= 1, str(prog))

        print("[7] asr_text 不可变")
        r = c.patch(f"/api/recordings/{rec_id}/segments/{seg['seq']}",
                    json={"asr_text": "篡改"})
        check("拒绝改 asr_text", r.status_code in (400, 422), f"got {r.status_code}")

        print("[8] 评估：引用强制")
        fws = c.get("/api/frameworks").json()["items"]
        shift = next(f for f in fws if f["code"] == "shift_handover_assessment")
        r = c.post(f"/api/recordings/{rec_id}/assessments", json={"framework_id": shift["id"]})
        check("创建评估受理（异步）", r.status_code == 200, r.text[:300])
        a = _wait_assessment(c, rec_id, timeout_s=120)
        check("评估解析完成", a is not None)
        if a:
            missing = [v for v in a["values"] if v["status"] == "missing_citation"]
            check("缺引用字段被标记", isinstance(missing, list), f"missing={len(missing)}")
            check("提交校验可用", "submittable" in a and "blockers" in a)

            target = next((v for v in a["values"] if v["content"]), None)
            if target:
                r = c.patch(f"/api/assessments/{a['id']}/fields/{target['field_key']}",
                            json={"content": target["content"], "citations": []})
                check("清空引用后不可提交",
                      r.status_code == 200 and r.json()["submittable"] is False,
                      r.text[:300])
                r = c.post(f"/api/assessments/{a['id']}/submit")
                check("无引用提交被拒", r.status_code == 422, f"got {r.status_code}")
                r = c.patch(f"/api/assessments/{a['id']}/fields/{target['field_key']}",
                            json={"content": "冒烟修正内容", "citations": [seg["seq"]]})
                if r.status_code == 200:
                    check("带引用可提交", r.json()["submittable"] is True,
                          str(r.json().get("blockers")))

        print("[9] 导出")
        for fmt in ("md", "docx", "xlsx"):
            r = c.get(f"/api/recordings/{rec_id}/export", params={"format": fmt})
            check(f"导出 {fmt}", r.status_code == 200, f"got {r.status_code} {r.text[:150]}")
            if r.status_code == 200:
                check(f"{fmt} 非空且文件名可解码",
                      len(r.content) > 100 and "filename*=" in r.headers.get(
                          "content-disposition", ""),
                      r.headers.get("content-disposition", ""))

        print("[10] 审计")
        r = c.get(f"/api/recordings/{rec_id}/audit")
        if r.status_code == 200:
            logs = r.json().get("items") or []
            check("有审计记录", len(logs) > 0, f"{len(logs)}")
            check("记录了人工确认", any(l["action"].startswith("segment")
                                       for l in logs))
        else:
            check("审计接口可用", False, f"got {r.status_code}")

    return _report(ok, fail)


def _wait(c: httpx.Client, rec_id: str, done: set[str], failed: set[str],
          timeout_s: float = 300) -> dict:
    deadline = time.time() + timeout_s
    rec = {}
    while time.time() < deadline:
        rec = c.get(f"/api/recordings/{rec_id}").json()
        if rec["status"] in done or rec["status"] in failed:
            return rec
        time.sleep(1)
    return rec


def _report(ok: int, fail: int) -> int:
    print(f"\n结果：{ok} 通过，{fail} 失败")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
