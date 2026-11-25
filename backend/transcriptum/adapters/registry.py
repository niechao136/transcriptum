"""适配器注册表 —— 编译期固定，不做动态发现（§7）。

新增模型 = 加一个 adapter 类 + 注册一行。
不提供上传 Python 代码、不动态 import、不写 entry_points。
"""

from __future__ import annotations

from .base import AsrAdapter, DiarizerAdapter, LlmAdapter

from .http_asr import CustomHttpAsr, OpenAiAudioAsr
from .http_llm import OpenAICompatibleLlm
from .mock import MockLlm, MockMossAsr, MockOpenAiAsr, NoopDiarizer

# ASR：moss 走自定义 HTTP（vLLM/SGLang 暴露的自定义作业接口），
# openai_audio 走标准 /v1/audio/transcriptions，custom_http 走可配置三段式。
ASR_ADAPTERS: dict[str, type] = {
    "mock": MockMossAsr,
    "mock_openai": MockOpenAiAsr,
    "moss": CustomHttpAsr,
    "openai_audio": OpenAiAudioAsr,
    "custom_http": CustomHttpAsr,
}

DIARIZERS: dict[str, type] = {
    "none": NoopDiarizer,
    "moss_inline": NoopDiarizer,   # MOSS 在 ASR 阶段已内联给出说话人
}

LLM_ADAPTERS: dict[str, type] = {
    "mock": MockLlm,
    "openai_compatible": OpenAICompatibleLlm,
}


def build_asr(cfg) -> AsrAdapter:
    """按 config 构造 ASR 适配器。"""
    cls = ASR_ADAPTERS[cfg.asr.adapter]
    if cfg.asr.adapter == "mock":
        return cls()
    if cfg.asr.adapter == "mock_openai":
        return cls()
    return cls(
        base_url=cfg.asr.base_url,
        api_key=cfg.asr.api_key,
        model=cfg.asr.model,
        timeout_s=cfg.asr.timeout_s,
        custom_http=cfg.asr.custom_http,
        adapter_name=cfg.asr.adapter,
        max_new_tokens=cfg.asr.max_new_tokens,
    )


def build_diarizer(cfg) -> DiarizerAdapter:
    cls = DIARIZERS[cfg.diarizer.adapter]
    return cls()


def build_llm(cfg) -> LlmAdapter:
    cls = LLM_ADAPTERS[cfg.llm.adapter]
    if cfg.llm.adapter == "mock":
        return cls(json_mode=cfg.llm.json_mode)
    return cls(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
        timeout_s=cfg.llm.timeout_s,
        json_mode=cfg.llm.json_mode,
    )
