"""MOSS 服务探测脚本（支持两种接口形态）。

用法：
  # 标准 OpenAI-compatible 同步接口（vLLM/SGLang 的 /v1/audio/transcriptions）
  python scripts/probe_moss.py --base http://<host>:9000 --mode sync

  # 作业式三段式接口（自建包装层 submit/poll/fetch）
  python scripts/probe_moss.py --base http://<host>:9000 --mode job

功能：
1. 探测 /v1/models，打印模型 id；
2. 上传一段 WAV，调用 /v1/audio/transcriptions；
3. 打印原始返回前 500 字；
4. 用 Transcriptum 的解析器解析 text，打印句子数/首句/说话人；
5. 失败时明确告诉你原样输出是什么，便于补解析器。

纯探测，不写库。
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


def make_wav(seconds: float = 3.0, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * seconds))
    return buf.getvalue()


def probe_models(client: httpx.Client) -> str | None:
    try:
        r = client.get("/v1/models", timeout=10)
        if r.status_code == 200:
            data = r.json()
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
            print(f"[models] {ids}")
            return ids[0] if ids else None
        print(f"[models] {r.status_code} {r.text[:200]}")
    except Exception as exc:
        print(f"[models] 不可用：{exc}")
    return None


def parse_and_report(text: str) -> bool:
    print("[4] 用 Transcriptum 解析器解析 text")
    try:
        segs = parse_moss_transcript(text)
    except MossParseError as exc:
        print(f"    [FAIL] 解析失败：{exc}")
        print("    ↑ 真实输出格式与官方样例有差异，请把上面的原始返回贴回来，"
              "我会补 moss_parser.py")
        return False
    print(f"    解析出 {len(segs)} 句")
    if segs:
        p = segs[0]
        print(f"    首句：{p.start_s:.2f}-{p.end_s:.2f} [{p.speaker_label}] {p.text[:40]}")
        spks = sorted({s.speaker_label for s in segs if s.speaker_label})
        print(f"    说话人标签：{spks}")
    print("[OK] MOSS 输出格式与项目解析器兼容")
    return True


def run_sync(client: httpx.Client, args, wav: bytes) -> int:
    print(f"[2] 同步调用 {args.transcriptions_path}")
    payload = {
        "model": args.model,
        # vLLM 的 MOSS 实现不支持 verbose_json / diarized_json 对纯音输入；
        # json 返回原生 "[ts][Sxx]text" 文本，正是项目解析器需要的格式。
        "response_format": "json",
        "max_new_tokens": str(args.max_new_tokens),
    }
    try:
        r = client.post(
            args.transcriptions_path,
            files={"file": ("probe.wav", wav, "audio/wav")},
            data=payload,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"    [FAIL] 请求失败：{exc}")
        return 1
    print(f"    HTTP {r.status_code}")
    if r.status_code >= 400:
        print(f"    {r.text[:600]}")
        return 1
    print("[3] 原始返回（前 500 字）：")
    raw = r.text
    print(json.dumps(r.json(), ensure_ascii=False)[:500] if _is_json(raw) else raw[:500])

    try:
        data = r.json()
    except Exception:
        print("    [FAIL] 返回非 JSON")
        return 1

    # verbose_json: 从 segments 提取带 [Sxx] 的 text，拼成 MOSS 全文交给解析器
    segs = data.get("segments")
    if isinstance(segs, list) and segs:
        text = "".join(str(s.get("text", "")) for s in segs)
        print(f"    从 {len(segs)} 个 segment 拼出全文（含 [Sxx] 前缀）")
    else:
        text = str(data.get("text", ""))
    if not text.strip():
        print("    [FAIL] 返回里既无 segments 也无 text")
        return 1
    return 0 if parse_and_report(text) else 1


def run_job(client: httpx.Client, args, wav: bytes) -> int:
    print(f"[2] 作业式提交 {args.submit_path}")
    try:
        r = client.post(
            args.submit_path,
            files={"file": ("probe.wav", wav, "audio/wav")},
            data={"model": args.model, "max_new_tokens": str(args.max_new_tokens)},
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"    [FAIL] submit 失败：{exc}")
        return 1
    print(f"    submit {r.status_code}: {r.text[:300]}")
    if r.status_code >= 400:
        return 1
    try:
        job_id = r.json().get("job_id") or r.json().get("id")
    except Exception:
        print("    [FAIL] submit 未返回 JSON job_id")
        return 1
    if not job_id:
        print("    [FAIL] submit 未返回 job_id")
        return 1

    deadline = time.monotonic() + args.timeout
    status = "running"
    while time.monotonic() < deadline:
        p = client.get(args.poll_path.format(job_id=job_id), timeout=30)
        if p.status_code >= 400:
            print(f"    [FAIL] poll {p.status_code}: {p.text[:200]}")
            return 1
        status = str(p.json().get("status", "")).lower()
        if status in ("done", "succeeded", "completed", "finished", "failed", "error"):
            break
        time.sleep(2)
    print(f"    poll 最终状态：{status}")
    if status in ("failed", "error"):
        return 1

    f = client.get(args.fetch_path.format(job_id=job_id), timeout=30)
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
        segs = payload.get("segments") if isinstance(payload, dict) else None
        if isinstance(segs, list) and segs:
            text = "".join(str(s.get("text", "")) for s in segs)
    if not text:
        print("    [FAIL] 没有 text 字段")
        return 1
    return 0 if parse_and_report(text) else 1


def _is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="模型服务根地址，如 http://127.0.0.1:9000")
    ap.add_argument("--mode", choices=["sync", "job"], default="sync")
    ap.add_argument("--model", default=None, help="不填则用 /v1/models 返回的第一个 id")
    ap.add_argument("--max-new-tokens", type=int, default=65536)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--transcriptions-path", default="/v1/audio/transcriptions")
    ap.add_argument("--submit-path", default="/v1/jobs")
    ap.add_argument("--poll-path", default="/v1/jobs/{job_id}")
    ap.add_argument("--fetch-path", default="/v1/jobs/{job_id}/result")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    print(f"[1] 目标：{base}  模式：{args.mode}")
    with httpx.Client(base_url=base) as client:
        detected = probe_models(client)
        if args.model is None:
            args.model = detected or "MOSS-Transcribe-Diarize"
        print(f"    使用 model id：{args.model}")
        wav = make_wav()
        print(f"[音频] 合成 3s 16k 单声道 WAV（{len(wav)} bytes）")
        return run_sync(client, args, wav) if args.mode == "sync" else run_job(client, args, wav)


if __name__ == "__main__":
    sys.exit(main())
