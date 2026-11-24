"""SQLite 访问层：连接、schema 初始化、事务、审计。"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


class Db:
    """薄封装：统一 row→dict、审计写入。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---------- 查询helper ----------
    def one(self, sql: str, params: tuple = ()) -> dict | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    # ---------- 审计（§10） ----------
    def audit(
        self,
        actor: str,
        action: str,
        entity: str,
        entity_id: str | int | None = None,
        *,
        recording_id: str | None = None,
        before: Any = None,
        after: Any = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO audit_log (actor, action, entity, entity_id, recording_id,"
            " before, after, at) VALUES (?,?,?,?,?,?,?,?)",
            (
                actor,
                action,
                entity,
                None if entity_id is None else str(entity_id),
                recording_id,
                json.dumps(before, ensure_ascii=False) if before is not None else None,
                json.dumps(after, ensure_ascii=False) if after is not None else None,
                now_iso(),
            ),
        )


def jloads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
