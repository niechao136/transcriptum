"""LLM 适配器：OpenAI-compatible /v1/chat/completions（§7）。

JSON mode 不可用时降级为「提示词要求 JSON + 严格 schema 校验 + 失败重试」，
仍不可靠则拒绝提交 —— 不做模糊解析（§7.1 能力降级表）。
"""

from __future__ import annotations

import json

import requests

from .base import AdapterError, Capabilities


class OpenAICompatibleLlm:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str = "",
        timeout_s: float = 120.0,
        json_mode: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.capabilities = Capabilities(json_mode=json_mode)

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "stream": False,
        }
        # 只有适配器声明支持 json_mode 时才带 response_format
        if json_mode and self.capabilities.json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise AdapterError(f"LLM 网络错误：{exc}") from exc

        if resp.status_code >= 500:
            raise AdapterError(f"LLM 服务端错误 {resp.status_code}")
        if resp.status_code >= 400:
            # json_mode 不被支持时，退回不带 response_format 再试一次
            if "response_format" in payload and resp.status_code in (400, 422):
                payload.pop("response_format", None)
                try:
                    resp = requests.post(
                        url, headers=headers, json=payload, timeout=self.timeout_s
                    )
                except requests.RequestException as exc:
                    raise AdapterError(f"LLM 网络错误：{exc}") from exc
            if resp.status_code >= 400:
                raise AdapterError(
                    f"LLM 请求错误 {resp.status_code}: {resp.text[:300]}", retryable=False
                )

        try:
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AdapterError("LLM 返回结构异常") from exc


def strict_json(raw: str, required_keys: list[str]) -> dict:
    """严格 schema 校验：缺字段直接失败，不透传残缺结果。"""
    text = raw.strip()
    # 容忍 ```json 包裹
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"LLM 输出不是合法 JSON：{exc}", retryable=True) from exc
    if not isinstance(obj, dict):
        raise AdapterError("LLM 输出不是 JSON 对象")
    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise AdapterError(f"LLM 输出缺少必需字段：{missing}")
    return obj
