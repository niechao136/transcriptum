"""共享测试夹具。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcriptum.config import Config
from transcriptum.db import Db, connect, init_db
from transcriptum.service import seed_frameworks
from transcriptum.worker import WorkerDeps


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    c = Config()
    c.db_path = str(tmp_path / "test.db")
    c.data_dir = str(tmp_path / "data")
    c.asr.adapter = "mock"
    c.llm.adapter = "mock"
    c.diarizer.adapter = "moss_inline"
    Path(c.data_dir).mkdir(parents=True, exist_ok=True)
    c.validate()
    return c


@pytest.fixture()
def db(cfg: Config) -> Db:
    conn = connect(cfg.db_path)
    init_db(conn)
    d = Db(conn)
    with d.tx():
        seed_frameworks(d)
    return d


@pytest.fixture()
def deps(cfg: Config, db: Db) -> WorkerDeps:
    return WorkerDeps.build(cfg, db)


@pytest.fixture()
def wav(tmp_path: Path) -> Path:
    """一个最小的合法 WAV（1 秒静音），让归一化/探针有东西可读。"""
    import wave

    p = tmp_path / "sample.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return p


def make_recording(cfg: Config, db: Db, deps: WorkerDeps, wav: Path, **kwargs):
    from transcriptum.service import RecordingService

    svc = RecordingService(cfg, db, deps)
    return svc.create(title=kwargs.pop("title", "测试会议"), kind="meeting",
                      source="upload", original_path=wav, **kwargs)
