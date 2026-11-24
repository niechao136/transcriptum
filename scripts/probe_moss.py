"""在目标机（2060S）跑：探测 MOSS 服务是否可用、输出形态是否符合解析器预期。

用法：
    python scripts/probe_moss.py --base http://127.0.0.1:9000

它会依次检查：
1. 服务健康；
2. 用一段合成的 WAV 走 submit → poll → fetch；
3. 打印 fetch 原始返回的前 500 字；
4. 用本项目的解析器解析，打印句子数与首句；
5. 如果解析失败，直接告诉你原样输出是什么，方便补 parser。

不做任何写库操作 —— 这是纯探测脚本。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import httpx  # noqa: E402

from transcriptum.adapters.moss_parser import MossParseError, parse_moss_transcript  # noqa: E402
from transcriptum.audio import normalize_to_wav  # noqa: E402


def make_wav(seconds: float = 3.0, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * seconds))
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:9000")
    ap.add_argument("--model", default="MOSS-Transcribe-Diarize")
    ap.add_argument("--submit-path", default="/v1/jobs")
    ap.add_argument("--poll-path", default="/v1/jobs/{job_id}")
    ap.add_argument("--fetch-path", default="/v1/jobs/{job_id}/result")
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()

    base = args.base.rstrip("/")
    print(f"[1] 健康检查 {base}")
    with httpx.Client(base_url=base, timeout=10) as c:
        try:
            r = c.get("/health")
            print(f"    {r.status_code} {r.text[:200]}")
        except Exception as exc:
            print(f"    [WARN] /health 不可用：{exc}（有的服务没有该端点，继续）")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "probe.wav"
        raw.write_bytes(make_wav())
        wav = Path(td) / "probe_16k.wav"
        normalize_to_wav(raw, wav)
        print(f"[2] 上传音频 {wav.stat().st_size} bytes")

        with httpx.Client(base_url=base, timeout=args.timeout) as c:
            try:
                with open(wav, "rb") as fh:
                    resp = c.post(
                        args.submit_path,
                        files={"file": ("probe.wav", fh, "audio/wav")},
                        data={"model": args.model},
                    )
            except Exception as exc:
                print(f"    [FAIL] submit 失败：{exc}")
                return 1
            print(f"    submit {resp.status_code}: {resp.text[:300]}")
            if resp.status_code >= 400:
                return 1
            try:
                job_id = resp.json().get("job_id") or resp.json().get("id")
            except Exception:
                print("    [FAIL] submit 未返回 JSON job_id")
                return 1
            if not job_id:
                print("    [FAIL] submit 未返回 job_id")
                return 1

            deadline = time.monotonic() + args.timeout
            status = "running"
            while time.monotonic() < deadline:
                p = c.get(args.poll_path.format(job_id=job_id))
                if p.status_code >= 400:
                    print(f"    [FAIL] poll {p.status_code}: {p.text[:200]}")
                    return 1
                status = str(p.json().get("status", "")).lower()
                if status in ("done", "succeeded", "completed", "finished", "failed", "error"):
                    break
                time.sleep(1)
            print(f"    poll 最终状态：{status}")
            if status in ("failed", "error"):
                return 1

            f = c.get(args.fetch_path.format(job_id=job_id))
            print(f"[3] fetch {f.status_code}")
            try:
                payload = f.json()
            except Exception:
                print(f"    [FAIL] fetch 非 JSON：{f.text[:500]}")
                return 1
            print("    原始返回（前 500 字）：")
            print(json.dumps(payload, ensure_ascii=False)[:500])

            text = payload.get("text") if isinstance(payload, dict) else None
            if not text:
                print("    [FAIL] fetch 没有 text 字段 —— 需要给 CustomHttpAsr 加对应解析分支")
                return 1
            print("[4] 用项目解析器解析")
            try:
                parsed = parse_moss_transcript(text)
            except MossParseError as exc:
                print(f"    [FAIL] 解析失败：{exc}")
                print("    ↑ 说明真实输出格式与文档的 [时间][说话人] 有差异，"
                      "请把上面的原始返回贴给我，我来补 moss_parser.py")
                return 1
            print(f"    解析出 {len(parsed)} 句")
            if parsed:
                p0 = parsed[0]
                print(f"    首句：{p0.start_s}s → {p0.end_s}s [{p0.speaker_label}] {p0.text[:40]}")
            print("[OK] MOSS 输出格式与本项目的解析器兼容")
    return 0


if __name__ == "__main__":
    sys.exit(main())
