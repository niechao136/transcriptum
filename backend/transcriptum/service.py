"""业务服务层：录音/句子/校对/评估/导出。API 层只做协议转换。"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .assessment import (
    BUILTIN_FRAMEWORKS,
    FrameworkDef,
    can_submit,
    parse_assessment,
)
from .config import Config
from .db import Db, jloads, new_id, now_iso
from .proofread import is_low_risk, proofread_segment
from .worker import WorkerDeps, enqueue


class NotFound(RuntimeError):
    pass


class Conflict(RuntimeError):
    pass


class ValidationFailed(RuntimeError):
    def __init__(self, message: str, reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = reasons or []


def seed_frameworks(db: Db) -> None:
    """内置框架写入（幂等）。"""
    for fw in BUILTIN_FRAMEWORKS:
        existing = db.one(
            "SELECT id FROM framework WHERE code=? AND version=?", (fw.code, fw.version)
        )
        if existing:
            continue
        db.exec(
            "INSERT INTO framework (code, name, builtin, fields, version, created_by,"
            " created_at, updated_at) VALUES (?,?,1,?,?,'system',?,?)",
            (fw.code, fw.name, json.dumps(fw.as_dict()["fields"], ensure_ascii=False),
             fw.version, now_iso(), now_iso()),
        )


class RecordingService:
    def __init__(self, cfg: Config, db: Db, deps: WorkerDeps) -> None:
        self.cfg = cfg
        self.db = db
        self.deps = deps

    # ---------- 创建 ----------
    def create(
        self,
        *,
        title: str,
        kind: str,
        source: str,
        original_path: Path,
        actor: str = "local",
        consent: bool = True,
    ) -> dict:
        if kind not in ("meeting", "assessment"):
            raise ValidationFailed(f"kind 非法：{kind}")
        if source not in ("upload", "live"):
            raise ValidationFailed(f"source 非法：{source}")
        if not Path(original_path).exists():
            raise ValidationFailed("上传文件不存在")

        rec_id = new_id()
        data_dir = Path(self.cfg.data_dir)
        audio_dir = data_dir / "audio"
        original_dir = data_dir / "original"
        audio_dir.mkdir(parents=True, exist_ok=True)
        original_dir.mkdir(parents=True, exist_ok=True)
        # 原始文件与归一化产物必须分目录：同路径会让归一化覆盖原始文件，
        # 丢掉「模型换版本后可重跑」的保底能力（§10）。
        stored = original_dir / f"{rec_id}{Path(original_path).suffix}"
        shutil.copyfile(original_path, stored)

        with self.db.tx():
            self.db.exec(
                "INSERT INTO recording (id, title, kind, source, audio_path, original_path,"
                " status, created_by, created_at, consent_at) VALUES (?,?,?,?,NULL,?,'uploaded',?,?,?)",
                (rec_id, title, kind, source, str(stored), actor, now_iso(),
                 now_iso() if consent else None),
            )
            self.db.audit(actor, "recording_created", "recording", rec_id, recording_id=rec_id,
                          after={"title": title, "kind": kind, "source": source})
        return self.get(rec_id)

    def get(self, rec_id: str) -> dict:
        rec = self.db.one("SELECT * FROM recording WHERE id=?", (rec_id,))
        if not rec:
            raise NotFound(f"录音不存在：{rec_id}")
        rec["asr_capabilities"] = jloads(rec.get("asr_capabilities"), {})
        rec["warnings"] = jloads(rec.get("warnings"), [])
        # 详情一次返回句子与进度：校对台需要一次性拿全，避免 N+1 往返
        rec["segments"] = self.segments(rec_id)
        rec["review_progress"] = self.review_progress(rec_id)
        rec["has_audio"] = bool(rec.get("audio_path")) and Path(rec["audio_path"]).exists()
        return rec

    def review_progress(self, rec_id: str) -> dict:
        row = self.db.one(
            "SELECT COUNT(*) AS total,"
            " SUM(CASE WHEN review_status!='pending' THEN 1 ELSE 0 END) AS reviewed,"
            " SUM(needs_review_speaker) AS needs_speaker"
            " FROM segment WHERE recording_id=?", (rec_id,))
        total = int(row["total"] or 0) if row else 0
        reviewed = int(row["reviewed"] or 0) if row else 0
        return {
            "total": total,
            "reviewed": reviewed,
            "pending": total - reviewed,
            "needs_speaker": int(row["needs_speaker"] or 0) if row else 0,
        }

    def list(self, *, status: str | None = None, limit: int = 50, offset: int = 0) -> dict:
        sql = "SELECT id,title,kind,source,status,duration_ms,created_at,error_msg FROM recording"
        params: list = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        total = self.db.one(f"SELECT COUNT(*) AS n FROM ({sql})", tuple(params))["n"]
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        return {"items": self.db.all(sql, tuple(params)), "total": total}

    # ---------- 流水线 ----------
    def start_transcribe(self, rec_id: str) -> str:
        rec = self.get(rec_id)
        if rec["status"] in ("transcribing",):
            raise Conflict("该录音正在转写中")
        return enqueue(self.db, rec_id, "transcribe",
                       model=self.cfg.asr.model, provider=self.cfg.asr.adapter)

    def start_proofread(self, rec_id: str) -> str:
        rec = self.get(rec_id)
        if rec["status"] not in ("transcribed", "proofread", "done"):
            raise Conflict(f"当前状态 {rec['status']} 不能进入校对")
        return enqueue(self.db, rec_id, "proofread",
                       model=self.cfg.llm.model, provider=self.cfg.llm.adapter,
                       prompt_version=self.cfg.proofread.prompt_version)

    # ---------- 句子 ----------
    def segments(self, rec_id: str) -> list[dict]:
        rows = self.db.all(
            "SELECT s.*, c.label AS speaker_label FROM segment s"
            " LEFT JOIN speaker_cluster c ON c.id=s.speaker_id"
            " WHERE s.recording_id=? ORDER BY s.seq", (rec_id,))
        for r in rows:
            r["ai_edits"] = jloads(r.get("ai_edits"), [])
            r["words"] = jloads(r.get("words"), None)
        return rows

    def update_segment(
        self,
        rec_id: str,
        seq: int,
        *,
        final_text: str | None = None,
        review_status: str | None = None,
        speaker_label: str | None = None,
        confirm_speaker: bool = False,
        actor: str = "local",
    ) -> dict:
        seg = self.db.one("SELECT * FROM segment WHERE recording_id=? AND seq=?",
                          (rec_id, seq))
        if not seg:
            raise NotFound(f"句子不存在：seq={seq}")

        if review_status and review_status not in ("pending", "accepted", "rejected", "edited"):
            raise ValidationFailed(f"review_status 非法：{review_status}")

        before = {"final_text": seg["final_text"], "review_status": seg["review_status"],
                  "speaker_id": seg["speaker_id"]}

        with self.db.tx():
            if final_text is not None:
                self.db.exec(
                    "UPDATE segment SET final_text=?, review_status=?, reviewed_by=?,"
                    " reviewed_at=? WHERE id=?",
                    (final_text, review_status or "edited", actor, now_iso(), seg["id"]))
            elif review_status is not None:
                # 采纳 = 把 ai_text 落到 final_text
                target = seg["ai_text"] if review_status == "accepted" else None
                self.db.exec(
                    "UPDATE segment SET review_status=?, final_text=COALESCE(?, final_text),"
                    " reviewed_by=?, reviewed_at=? WHERE id=?",
                    (review_status, target, actor, now_iso(), seg["id"]))
            if speaker_label is not None:
                cluster = self.db.one(
                    "SELECT id FROM speaker_cluster WHERE recording_id=? AND label=?",
                    (rec_id, speaker_label))
                if not cluster:
                    cur = self.db.exec(
                        "INSERT INTO speaker_cluster (recording_id, label) VALUES (?,?)",
                        (rec_id, speaker_label))
                    cluster_id = int(cur.lastrowid)
                else:
                    cluster_id = cluster["id"]
                self.db.exec("UPDATE segment SET speaker_id=? WHERE id=?",
                             (cluster_id, seg["id"]))
            if confirm_speaker:
                self.db.exec("UPDATE segment SET needs_review_speaker=0 WHERE id=?",
                             (seg["id"],))

            self.db.audit(actor, "segment_updated", "segment", seg["id"], recording_id=rec_id,
                          before=before,
                          after={"final_text": final_text, "review_status": review_status,
                                 "speaker_label": speaker_label})

        return self.db.one(
            "SELECT s.*, c.label AS speaker_label FROM segment s"
            " LEFT JOIN speaker_cluster c ON c.id=s.speaker_id"
            " WHERE s.id=?", (seg["id"],))

    def batch_accept(self, rec_id: str, *, actor: str = "local",
                     min_confidence: float | None = None) -> dict:
        """批量采纳：只覆盖「低风险」句 —— 模型改动小且置信度高。"""
        threshold = (min_confidence if min_confidence is not None
                     else self.cfg.proofread.batch_accept_min_confidence)
        rows = self.db.all(
            "SELECT s.* FROM segment s WHERE s.recording_id=? AND s.review_status='pending'",
            (rec_id,))
        accepted, skipped = [], []
        with self.db.tx():
            for r in rows:
                edits = jloads(r["ai_edits"], [])
                from .proofread import Edit, ProofResult
                try:
                    parsed = [Edit(type=e["type"], frm=e["from"], to=e["to"],
                                   reason=e.get("reason", ""),
                                   confidence=float(e.get("confidence", 0)))
                              for e in edits]
                except (KeyError, TypeError, ValueError):
                    parsed = []
                res = ProofResult(corrected_text=r["ai_text"] or r["asr_text"],
                                  edits=parsed, accepted=bool(parsed),
                                  requires_review=bool(r["low_confidence"]),
                                  warning=None)
                if is_low_risk(res, min_confidence=threshold) and not r["low_confidence"]:
                    self.db.exec(
                        "UPDATE segment SET final_text=?, review_status='accepted',"
                        " reviewed_by=?, reviewed_at=? WHERE id=?",
                        (r["ai_text"], actor, now_iso(), r["id"]))
                    accepted.append(r["seq"])
                else:
                    skipped.append(r["seq"])
            self.db.audit(actor, "batch_accept", "recording", rec_id, recording_id=rec_id,
                          after={"accepted": len(accepted), "skipped": len(skipped)})
        return {"accepted": accepted, "skipped": skipped}

    # ---------- 评估 ----------
    def frameworks(self) -> list[dict]:
        rows = self.db.all(
            "SELECT * FROM framework ORDER BY code, version DESC")
        for r in rows:
            r["fields"] = jloads(r["fields"], [])
        return rows

    def create_framework(self, body: dict, *, actor: str = "local") -> dict:
        code = str(body.get("code", "")).strip()
        name = str(body.get("name", "")).strip()
        fields = body.get("fields")
        if not code or not name:
            raise ValidationFailed("code 与 name 必填")
        if not isinstance(fields, list) or not fields:
            raise ValidationFailed("fields 必须是非空数组")
        keys = [f.get("key") for f in fields]
        if any(not k for k in keys):
            raise ValidationFailed("每个字段必须有 key")
        if len(set(keys)) != len(keys):
            raise ValidationFailed("字段 key 不能重复")
        for f in fields:
            if f.get("type") not in ("text", "number", "enum", "list"):
                raise ValidationFailed(f"字段 {f.get('key')} 的 type 非法")
            if f.get("type") == "enum" and not f.get("options"):
                raise ValidationFailed(f"enum 字段 {f.get('key')} 必须给 options")

        latest = self.db.one(
            "SELECT MAX(version) AS v FROM framework WHERE code=?", (code,))
        version = int(latest["v"] or 0) + 1
        with self.db.tx():
            cur = self.db.exec(
                "INSERT INTO framework (code, name, builtin, fields, version, created_by,"
                " created_at, updated_at) VALUES (?,?,0,?,?,?,?,?)",
                (code, name, json.dumps(fields, ensure_ascii=False), version, actor,
                 now_iso(), now_iso()))
            fid = int(cur.lastrowid)
            self.db.audit(actor, "framework_created", "framework", fid,
                          after={"code": code, "version": version})
        return self.db.one("SELECT * FROM framework WHERE id=?", (fid,))

    def start_assessment(self, rec_id: str, framework_id: int,
                         *, actor: str = "local") -> str:
        fw = self.db.one("SELECT * FROM framework WHERE id=?", (framework_id,))
        if not fw:
            raise NotFound("框架不存在")
        return enqueue(self.db, rec_id, "assessment",
                       model=self.cfg.llm.model, provider=self.cfg.llm.adapter,
                       prompt_version="assess-v1")

    def assessments(self, rec_id: str | None = None) -> list[dict]:
        if rec_id:
            rows = self.db.all(
                "SELECT * FROM assessment WHERE recording_id=? ORDER BY id DESC", (rec_id,))
        else:
            rows = self.db.all("SELECT * FROM assessment ORDER BY id DESC")
        for r in rows:
            r["values"] = self._values(r["id"])
            r["framework"] = jloads(r.get("framework_snapshot"), {})
            from .assessment import FieldValue

            fvs = [
                FieldValue(
                    key=v["field_key"], content=v["content"] or "",
                    citations=v["citations"] or [], status=v["status"],
                )
                for v in r["values"]
            ]
            ok, reasons = can_submit(fvs, FrameworkDef.from_dict(r["framework"]))
            r["submittable"] = ok
            r["blockers"] = reasons
        return rows

    def _values(self, aid: int) -> list[dict]:
        rows = self.db.all(
            "SELECT * FROM assessment_field_value WHERE assessment_id=? ORDER BY id", (aid,))
        for r in rows:
            r["citations"] = jloads(r.get("citations"), [])
        return rows

    def update_field(
        self, assessment_id: int, field_key: str, *, content: str,
        citations: list[int], actor: str = "local",
    ) -> dict:
        a = self.db.one("SELECT * FROM assessment WHERE id=?", (assessment_id,))
        if not a:
            raise NotFound("评估不存在")
        if a["status"] == "submitted":
            raise Conflict("已提交的评估不可直接修改，请新建修订")
        row = self.db.one(
            "SELECT * FROM assessment_field_value WHERE assessment_id=? AND field_key=?",
            (assessment_id, field_key))
        if not row:
            raise NotFound(f"字段不存在：{field_key}")

        # 人工修改必须重新确认引用有效性
        valid = {s["seq"] for s in self.db.all(
            "SELECT seq FROM segment WHERE recording_id=?", (a["recording_id"],))}
        bad = [c for c in citations if c not in valid]
        if bad:
            raise ValidationFailed(f"引用了不存在的句子 seq：{bad}")

        status = "ok"
        if not content.strip():
            status = "not_extracted"
        elif not citations:
            status = "missing_citation"

        with self.db.tx():
            self.db.exec(
                "UPDATE assessment_field_value SET content=?, citations=?, source='human_edited',"
                " status=? WHERE id=?",
                (content, json.dumps(citations), status, row["id"]))
            self.db.exec("UPDATE assessment SET updated_at=? WHERE id=?",
                         (now_iso(), assessment_id))
            self.db.audit(actor, "assessment_field_updated", "assessment", assessment_id,
                          recording_id=a["recording_id"],
                          before={"content": row["content"], "citations": row["citations"]},
                          after={"content": content, "citations": citations})
        # 返回完整评估（含 submittable/blockers），前端改完字段即可看到提交状态变化
        full = self.assessments(a["recording_id"])
        match = next((x for x in full if x["id"] == assessment_id), None)
        if match is None:
            raise NotFound("评估不存在")
        return match

    def submit_assessment(self, assessment_id: int, *, actor: str = "local") -> dict:
        a = self.db.one("SELECT * FROM assessment WHERE id=?", (assessment_id,))
        if not a:
            raise NotFound("评估不存在")
        fw = FrameworkDef.from_dict(jloads(a["framework_snapshot"], {}))
        values = self._values(assessment_id)
        from .assessment import FieldValue
        fvs = [FieldValue(key=v["field_key"], content=v["content"] or "",
                          citations=v["citations"], status=v["status"]) for v in values]
        ok, reasons = can_submit(fvs, fw)
        if not ok:
            raise ValidationFailed("提交校验失败", reasons)
        with self.db.tx():
            self.db.exec("UPDATE assessment SET status='submitted', updated_at=? WHERE id=?",
                         (now_iso(), assessment_id))
            self.db.audit(actor, "assessment_submitted", "assessment", assessment_id,
                          recording_id=a["recording_id"])
        return {"status": "submitted"}

    # ---------- 保留期（§10） ----------
    def purge_expired(self, *, actor: str = "system", days: int | None = None) -> dict:
        if days is not None and days < 0:
            raise ValueError("保留天数不能为负")
        days = self.cfg.security.audio_retention_days if days is None else days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
            timespec="seconds")
        rows = self.db.all(
            "SELECT id, audio_path, original_path FROM recording WHERE created_at < ?"
            " AND audio_path IS NOT NULL", (cutoff,))
        removed = 0
        for r in rows:
            for p in (r["audio_path"], r["original_path"]):
                if p:
                    f = Path(p)
                    if f.exists():
                        f.unlink()
                        removed += 1
            with self.db.tx():
                self.db.exec("UPDATE recording SET audio_path=NULL, original_path=NULL"
                             " WHERE id=?", (r["id"],))
                self.db.audit(actor, "audio_purged", "recording", r["id"],
                              recording_id=r["id"], after={"retention_days": days})
        return {"purged_recordings": len(rows), "removed_files": removed}
