"""ASR 适配器：自定义 HTTP 三段式 + OpenAI-compatible audio/transcriptions（§7）。

对外统一为 submit / poll / fetch，上层不感知差异。
失败抛 AdapterError(retryable=...) —— 由队列决定重试，绝不自动降级换模型。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from .base import AdapterError, Capabilities, JobState, RawSegment
from .moss_parser import MossParseError, parse_moss_transcript, to_raw_segments


class CustomHttpAsr:
    """submit → poll → fetch 三段式，路径可配（适配各厂商长音频接口）。"""

    name = "custom_http"
    capabilities = Capabilities(
        has_word_ts=False, has_confidence=False, has_speaker=True, json_mode=False
    )

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str = "",
        timeout_s: float = 60.0,
        custom_http: dict | None = None,
        adapter_name: str = "custom_http",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.name = adapter_name
        cfg = custom_http or {}
        self.submit_path = cfg.get("submit_path", "/v1/jobs")
        self.poll_path = cfg.get("poll_path", "/v1/jobs/{job_id}")
        self.fetch_path = cfg.get("fetch_path", "/v1/jobs/{job_id}/result")
        self.poll_interval_s = float(cfg.get("poll_interval_s", 2))
        self.max_poll_s = float(cfg.get("max_poll_s", 3600))

        # 能力可由配置覆盖：某些部署的 ASR 跑在带字级时间戳的服务上
        caps = cfg.get("capabilities") or {}
        self.capabilities = Capabilities(
            has_word_ts=bool(caps.get("has_word_ts", False)),
            has_confidence=bool(caps.get("has_confidence", False)),
            has_speaker=bool(caps.get("has_speaker", True)),
            json_mode=False,
        )

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def submit(self, audio_path: Path, hotwords: list[str]) -> str:
        url = f"{self.base_url}{self.submit_path}"
        try:
            with open(audio_path, "rb") as fh:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    files={"file": (Path(audio_path).name, fh, "audio/wav")},
                    data={
                        "model": self.model,
                        "hotwords": json.dumps(hotwords, ensure_ascii=False),
                    },
                    timeout=self.timeout_s,
                )
        except requests.RequestException as exc:
            raise AdapterError(f"ASR submit 网络错误：{exc}") from exc
        if resp.status_code >= 500:
            raise AdapterError(f"ASR submit 服务端错误 {resp.status_code}")
        if resp.status_code >= 400:
            raise AdapterError(
                f"ASR submit 请求错误 {resp.status_code}: {resp.text[:300]}", retryable=False
            )
        try:
            job_id = resp.json().get("job_id") or resp.json().get("id")
        except ValueError as exc:
            raise AdapterError("ASR submit 返回非 JSON") from exc
        if not job_id:
            raise AdapterError("ASR submit 未返回 job_id")
        return str(job_id)

    def poll(self, job_id: str) -> JobState:
        url = f"{self.base_url}{self.poll_path.format(job_id=job_id)}"
        deadline = time.monotonic() + self.max_poll_s
        while True:
            try:
                resp = requests.get(url, headers=self._headers(), timeout=self.timeout_s)
            except requests.RequestException as exc:
                raise AdapterError(f"ASR poll 网络错误：{exc}") from exc
            if resp.status_code >= 400:
                raise AdapterError(f"ASR poll 失败 {resp.status_code}")
            data = resp.json()
            status = str(data.get("status", "")).lower()
            if status in ("done", "succeeded", "completed", "finished"):
                return JobState(job_id=job_id, status="done", progress=1.0)
            if status in ("failed", "error", "cancelled"):
                return JobState(
                    job_id=job_id, status="failed", error=str(data.get("error", "unknown"))
                )
            if time.monotonic() > deadline:
                raise AdapterError(f"ASR poll 超时（>{self.max_poll_s}s）")
            time.sleep(self.poll_interval_s)

    def fetch(self, job_id: str) -> list[RawSegment]:
        url = f"{self.base_url}{self.fetch_path.format(job_id=job_id)}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise AdapterError(f"ASR fetch 网络错误：{exc}") from exc
        if resp.status_code >= 400:
            raise AdapterError(f"ASR fetch 失败 {resp.status_code}: {resp.text[:300]}")
        data = resp.json()

        # 两种常见返回：结构化 segments，或 MOSS 风格全文
        if isinstance(data, dict) and isinstance(data.get("segments"), list):
            return _from_structured(data["segments"], self.capabilities)
        text = data.get("text") if isinstance(data, dict) else None
        if not text:
            raise AdapterError("ASR fetch 返回既无 segments 也无 text", retryable=False)
        try:
            parsed = parse_moss_transcript(text)
        except MossParseError as exc:
            raise AdapterError(f"MOSS 输出解析失败：{exc}", retryable=False) from exc
        return to_raw_segments(parsed, has_speaker=self.capabilities.has_speaker)


class OpenAiAudioAsr:
    """标准 OpenAI /v1/audio/transcriptions（同步接口，内部包成 job）。"""

    name = "openai_audio"
    capabilities = Capabilities(
        has_word_ts=False, has_confidence=False, has_speaker=False, json_mode=False
    )

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str = "whisper-1",
        timeout_s: float = 300.0,
        custom_http: dict | None = None,
        adapter_name: str = "openai_audio",
        max_new_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model or "whisper-1"
        self.timeout_s = timeout_s
        # MOSS 默认 max_new_tokens=5120，长音频会被截断；由配置显式调高。
        self.max_new_tokens = max_new_tokens
        self.name = adapter_name
        caps = (custom_http or {}).get("capabilities") or {}
        self.capabilities = Capabilities(
            has_word_ts=bool(caps.get("has_word_ts", False)),
            has_confidence=bool(caps.get("has_confidence", False)),
            has_speaker=bool(caps.get("has_speaker", False)),
            json_mode=False,
        )
        self._cache: dict[str, list[RawSegment]] = {}

    def submit(self, audio_path: Path, hotwords: list[str]) -> str:
        url = f"{self.base_url}/audio/transcriptions"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        prompt = "、".join(hotwords[:200]) if hotwords else ""
        data = {
            "model": self.model,
            # vLLM 的 MOSS 实现不支持 verbose_json（400 BadRequest），
            # 但 json 格式返回原生 "[ts][Sxx]text" 文本，含时间戳与说话人标签，
            # 由下方 moss_parser 解析，信息更完整。
            "response_format": "json",
        }
        if self.max_new_tokens is not None:
            data["max_new_tokens"] = str(self.max_new_tokens)
        if prompt:
            data["prompt"] = prompt
        try:
            with open(audio_path, "rb") as fh:
                resp = requests.post(
                    url,
                    headers=headers,
                    files={"file": (Path(audio_path).name, fh, "audio/wav")},
                    data=data,
                    timeout=self.timeout_s,
                )
        except requests.RequestException as exc:
            raise AdapterError(f"OpenAI ASR 网络错误：{exc}") from exc
        if resp.status_code >= 500:
            raise AdapterError(f"OpenAI ASR 服务端错误 {resp.status_code}")
        if resp.status_code >= 400:
            raise AdapterError(
                f"OpenAI ASR 请求错误 {resp.status_code}: {resp.text[:300]}", retryable=False
            )
        payload = resp.json()
        segs = []
        for item in payload.get("segments") or []:
            # 官方 verbose_json 的 segment.text 形如 "[S01]实际内容"：
            # 必须把说话人前缀拆出来，否则会混进正文、且对齐层拿不到 speaker。
            text = str(item.get("text", "")).strip()
            speaker = None
            sm = re.match(r"^\[(S\d{2,3}|[A-Z]{1,3})\]\s*", text)
            if sm:
                speaker = sm.group(1)
                text = text[sm.end():].strip()
            segs.append(
                RawSegment(
                    start_s=float(item.get("start", 0.0)),
                    end_s=float(item.get("end", 0.0)),
                    text=text,
                    conf=item.get("avg_logprob"),
                    speaker_label=speaker,
                )
            )
        if not segs and payload.get("text"):
            # json/text 格式只有全文：交给 MOSS 解析器（含显式 end 时间戳）
            from .moss_parser import MossParseError, parse_moss_transcript, to_raw_segments
            try:
                segs = to_raw_segments(
                    parse_moss_transcript(str(payload["text"])),
                    has_speaker=True,
                )
            except MossParseError:
                segs = [RawSegment(start_s=0.0, end_s=0.0,
                                   text=str(payload["text"]).strip())]
        job_id = f"oai-{abs(hash((str(audio_path), self.model))) % 10**12}"
        self._cache[job_id] = segs
        return job_id

    def poll(self, job_id: str) -> JobState:
        if job_id in self._cache:
            return JobState(job_id=job_id, status="done", progress=1.0)
        return JobState(job_id=job_id, status="failed", error="unknown job")

    def fetch(self, job_id: str) -> list[RawSegment]:
        if job_id not in self._cache:
            raise AdapterError(f"unknown job {job_id}", retryable=False)
        return list(self._cache[job_id])


def _from_structured(items: list, caps: Capabilities) -> list[RawSegment]:
    out: list[RawSegment] = []
    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = float(item.get("start", item.get("start_s", 0.0)))
        end = float(item.get("end", item.get("end_s", start)))
        words = item.get("words") if caps.has_word_ts else None
        conf = item.get("confidence", item.get("conf")) if caps.has_confidence else None
        speaker = None
        if caps.has_speaker:
            speaker = item.get("speaker") or item.get("speaker_label")
        out.append(
            RawSegment(
                start_s=start,
                end_s=end,
                text=text,
                words=words,
                conf=conf,
                speaker_label=speaker,
            )
        )
    return out
