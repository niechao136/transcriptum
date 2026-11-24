"""容器化配置与单镜像部署的回归测试（Dockerfile 之外的可测部分）。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from transcriptum.api import create_app
from transcriptum.config import Config, ConfigError


def test_env_overrides_paths(tmp_path):
    c = Config()
    c.apply_env({
        "TRANSCRIPTUM_DB_PATH": str(tmp_path / "sub" / "x.db"),
        "TRANSCRIPTUM_DATA_DIR": str(tmp_path / "data"),
    })
    assert c.db_path == str(tmp_path / "sub" / "x.db")
    assert c.data_dir == str(tmp_path / "data")


def test_env_overrides_adapters(cfg):
    cfg.apply_env({
        "TRANSCRIPTUM_ASR_ADAPTER": "openai_audio",
        "TRANSCRIPTUM_LLM_ADAPTER": "openai_compatible",
        "TRANSCRIPTUM_LLM_JSON_MODE": "false",
        "TRANSCRIPTUM_WORKERS": "3",
        "TRANSCRIPTUM_EXPORT_REDACT": "yes",
    })
    assert cfg.asr.adapter == "openai_audio"
    assert cfg.llm.adapter == "openai_compatible"
    assert cfg.llm.json_mode is False
    assert cfg.queue.max_workers == 3
    assert cfg.security.export_redact is True


def test_env_bad_bool_fails(tmp_path):
    c = Config()
    with pytest.raises(ConfigError):
        c.apply_env({"TRANSCRIPTUM_LLM_JSON_MODE": "perhaps"})


def test_env_egress_appended(cfg):
    before = len(cfg.security.allowed_egress_domains)
    cfg.apply_env({"TRANSCRIPTUM_ALLOWED_EGRESS": "gpu.internal, host.docker.internal"})
    assert cfg.security.allowed_egress_domains[-2:] == ["gpu.internal", "host.docker.internal"]
    assert len(cfg.security.allowed_egress_domains) == before + 2


def test_remote_endpoint_requires_egress_whitelist(cfg):
    # 默认白名单含 *.internal；这里用一个明确不在白名单里的公网主机
    cfg.apply_env({
        "TRANSCRIPTUM_ASR_ADAPTER": "moss",
        "TRANSCRIPTUM_ASR_BASE_URL": "http://model.example.com:9000",
    })
    with pytest.raises(ConfigError):
        cfg.validate()


def test_static_dir_serves_index_when_built(tmp_path):
    """镜像里 /app/static 存在时，API 应在 / 直接返回构建好的前端。"""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Transcriptum</h1>", encoding="utf-8")
    (static / "asset.js").write_text("// js", encoding="utf-8")

    old = os.environ.get("TRANSCRIPTUM_STATIC_DIR")
    os.environ["TRANSCRIPTUM_STATIC_DIR"] = str(static)
    try:
        from fastapi.testclient import TestClient

        cfg = Config()
        cfg.db_path = str(tmp_path / "app.db")
        cfg.data_dir = str(tmp_path / "data")
        app = create_app(cfg)
        c = TestClient(app)
        r = c.get("/")
        assert r.status_code == 200
        assert "Transcriptum" in r.text
        assert c.get("/asset.js").status_code == 200
    finally:
        if old is None:
            os.environ.pop("TRANSCRIPTUM_STATIC_DIR", None)
        else:
            os.environ["TRANSCRIPTUM_STATIC_DIR"] = old


def test_no_static_dir_still_serves_api(cfg):
    """没有前端构建产物时，API 必须仍能独立工作（开发/纯后端部署）。"""
    old = os.environ.get("TRANSCRIPTUM_STATIC_DIR")
    os.environ["TRANSCRIPTUM_STATIC_DIR"] = str(Path(cfg.data_dir) / "nonexistent-static")
    try:
        from fastapi.testclient import TestClient

        app = create_app(cfg)
        c = TestClient(app)
        assert c.get("/api/health").json()["ok"] is True
    finally:
        if old is None:
            os.environ.pop("TRANSCRIPTUM_STATIC_DIR", None)
        else:
            os.environ["TRANSCRIPTUM_STATIC_DIR"] = old
