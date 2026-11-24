"""配置加载与安全边界校验（设计文档 §7.1 安全边界、§7.5 阈值可配）。"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


class ConfigError(ValueError):
    """配置非法 —— 启动期失败，不静默回退。

    继承 ValueError：配置错误本质是取值错误，方便调用方按值错误统一捕获。
    """


DEFAULT_EGRESS = ["localhost", "127.0.0.1", "::1", "*.internal"]


def _is_private_host(host: str) -> bool:
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback


def _domain_allowed(host: str, allowed: list[str]) -> bool:
    """精确匹配、通配后缀匹配，或本身就是内网/回环地址。"""
    host = host.lower().strip()
    for pattern in allowed:
        p = pattern.lower().strip()
        if not p:
            continue
        if p.startswith("*."):
            if host.endswith(p[1:]):  # "*.internal" -> ".internal"
                return True
        elif host == p:
            return True
    return _is_private_host(host)


def validate_egress(url: str, allowed: list[str], *, what: str) -> None:
    """所有 base_url 必须落在 allowed_egress_domains 内，否则拒绝启动。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ConfigError(f"{what}: base_url 必须是 http/https，得到 {url!r}")
    host = parsed.hostname
    if not host:
        raise ConfigError(f"{what}: base_url 缺少主机名：{url!r}")
    if not _domain_allowed(host, allowed):
        raise ConfigError(
            f"{what}: base_url 主机 {host!r} 不在 allowed_egress_domains {allowed} 内。"
            "外接端点误配成公网 URL 属安全事故，已拒绝启动。"
        )


@dataclass
class AsrConfig:
    adapter: str = "mock"
    base_url: str = "http://127.0.0.1:9000"
    api_key_env: str = "ASR_API_KEY"
    hotwords_file: str | None = None
    model: str = "moss-transcribe-diarize"
    custom_http: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 60.0

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass
class DiarizerConfig:
    adapter: str = "moss_inline"


@dataclass
class LlmConfig:
    adapter: str = "openai_compatible"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "LLM_API_KEY"
    model: str = "Qwen2.5-7B-Instruct"
    json_mode: bool = True
    prompt_version: str = "proofread-v1"
    timeout_s: float = 120.0

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass
class DiarizationConfig:
    """§7.5 第三步：阈值全部可配。"""

    assign_mode: str = "overlap_max"       # overlap_max | midpoint_split
    ambiguous_margin: float = 0.15         # 前两名重叠差 < 15% → 标记待确认
    min_overlap_ratio: float = 0.3         # 低于此比例视为无重叠，走最近块兜底
    on_ambiguous: str = "flag_review"      # flag_review | auto_split
    switch_overlap_threshold: float = 0.25  # 句内切换：两人重叠都超过该比例才切分

    def validate(self) -> None:
        if self.assign_mode not in ("overlap_max", "midpoint_split"):
            raise ConfigError(f"diarization.assign_mode 非法：{self.assign_mode}")
        if self.on_ambiguous not in ("flag_review", "auto_split"):
            raise ConfigError(f"diarization.on_ambiguous 非法：{self.on_ambiguous}")
        for name in ("ambiguous_margin", "min_overlap_ratio", "switch_overlap_threshold"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ConfigError(f"diarization.{name} 必须在 [0,1]，得到 {v}")


@dataclass
class ProofreadConfig:
    low_confidence_threshold: float = 0.6   # ASR 置信度低于此→标红（§4.3）
    batch_accept_min_confidence: float = 0.85  # 批量采纳的 AI 置信度门槛
    window_segments: int = 3               # 上下文窗口
    prompt_version: str = "proofread-v1"


@dataclass
class SecurityConfig:
    allowed_egress_domains: list[str] = field(default_factory=lambda: list(DEFAULT_EGRESS))
    audio_retention_days: int = 30
    encrypt_at_rest: bool = False
    export_redact: bool = True


@dataclass
class QueueConfig:
    max_workers: int = 1
    asr_concurrency: int = 1
    poll_interval_s: float = 1.0
    max_attempts: int = 3


@dataclass
class Config:
    db_path: str = "./data/transcriptum.db"
    data_dir: str = "./data"
    asr: AsrConfig = field(default_factory=AsrConfig)
    diarizer: DiarizerConfig = field(default_factory=DiarizerConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    proofread: ProofreadConfig = field(default_factory=ProofreadConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)

    # ---- 构造与校验 ----
    @staticmethod
    def load(path: str | Path | None) -> "Config":
        raw: dict[str, Any] = {}
        if path and Path(path).exists():
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = _from_dict(raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.diarization.validate()
        allowed = self.security.allowed_egress_domains
        if not allowed:
            raise ConfigError("security.allowed_egress_domains 不得为空")

        from .adapters.registry import ASR_ADAPTERS, DIARIZERS, LLM_ADAPTERS

        if self.asr.adapter not in ASR_ADAPTERS:
            raise ConfigError(
                f"asr.adapter={self.asr.adapter!r} 未注册，可选：{sorted(ASR_ADAPTERS)}"
            )
        if self.diarizer.adapter not in DIARIZERS:
            raise ConfigError(
                f"diarizer.adapter={self.diarizer.adapter!r} 未注册，可选：{sorted(DIARIZERS)}"
            )
        if self.llm.adapter not in LLM_ADAPTERS:
            raise ConfigError(
                f"llm.adapter={self.llm.adapter!r} 未注册，可选：{sorted(LLM_ADAPTERS)}"
            )

        # 外接端点必须过数据边界
        if self.asr.adapter != "mock":
            validate_egress(self.asr.base_url, allowed, what="asr")
        if self.llm.adapter not in ("mock",):
            validate_egress(self.llm.base_url, allowed, what="llm")

        for key in ("db_path", "data_dir"):
            p = Path(getattr(self, key))
            p.parent.mkdir(parents=True, exist_ok=True) if p.suffix else p.mkdir(
                parents=True, exist_ok=True
            )


def _kwargs(dc, raw: dict[str, Any], section: str):
    if not isinstance(raw, dict):
        raise ConfigError(f"{section} 必须是对象，得到 {type(raw).__name__}")
    known = {f for f in dc.__dataclass_fields__}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{section} 存在未知配置项：{sorted(unknown)}")
    return {k: v for k, v in raw.items() if k in known}


def _from_dict(raw: dict[str, Any]) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("配置根必须是对象")
    top = {"db_path", "data_dir", "asr", "diarizer", "llm", "diarization",
           "proofread", "security", "queue"}
    unknown = set(raw) - top
    if unknown:
        raise ConfigError(f"配置根存在未知配置项：{sorted(unknown)}")

    cfg = Config()
    for key in ("db_path", "data_dir"):
        if key in raw:
            setattr(cfg, key, raw[key])
    if "asr" in raw:
        cfg.asr = AsrConfig(**_kwargs(AsrConfig, raw["asr"], "asr"))
    if "diarizer" in raw:
        cfg.diarizer = DiarizerConfig(**_kwargs(DiarizerConfig, raw["diarizer"], "diarizer"))
    if "llm" in raw:
        cfg.llm = LlmConfig(**_kwargs(LlmConfig, raw["llm"], "llm"))
    if "diarization" in raw:
        cfg.diarization = DiarizationConfig(
            **_kwargs(DiarizationConfig, raw["diarization"], "diarization")
        )
    if "proofread" in raw:
        cfg.proofread = ProofreadConfig(
            **_kwargs(ProofreadConfig, raw["proofread"], "proofread")
        )
    if "security" in raw:
        cfg.security = SecurityConfig(
            **_kwargs(SecurityConfig, raw["security"], "security")
        )
    if "queue" in raw:
        cfg.queue = QueueConfig(**_kwargs(QueueConfig, raw["queue"], "queue"))
    return cfg
