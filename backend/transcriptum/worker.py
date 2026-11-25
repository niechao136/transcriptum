"""流水线 Worker（§4.3 队列与阶段）。

- SQLite job 表 + 单进程轮询，够用；
- 一次只跑一个 ASR（asr_concurrency 可配，默认 1）—— 本地模型显存限制；
- 失败重试 N 次后标 failed，绝静默降级换模型（§7.1）；
- 换模型只影响新任务，不重跑历史（§7.4 可复现）。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import align as align_mod
from .adapters.base import AdapterError, AsrAdapter, LlmAdapter, RawSegment
from .adapters.registry import build_asr, build_diarizer, build_llm
from .assessment import FrameworkDef, can_submit, parse_assessment
from .audio import normalize_to_wav, probe_duration_ms
from .config import Config
from .db import Db, jloads, new_id, now_iso
from .proofread import is_low_risk, proofread_segment

log = logging.getLogger("transcriptum.worker")


@dataclass
class WorkerDeps:
    cfg: Config
    db: Db
    asr: AsrAdapter
    llm: LlmAdapter
    diarizer: object

    @staticmethod
    def build(cfg: Config, db: Db) -> "WorkerDeps":
        return WorkerDeps(
            cfg=cfg, db=db, asr=build_asr(cfg), llm=build_llm(cfg),
            diarizer=build_diarizer(cfg),
        )


def enqueue(db: Db, recording_id: str, kind: str, *, model: str = "",
            provider: str = "", prompt_version: str = "") -> str:
    job_id = new_id()
    with db.tx():
        db.exec(
            "INSERT INTO job (id, recording_id, kind, status, next_run_at,"
            " model_name, provider, prompt_version, trace_id, created_at)"
            " VALUES (?,?,?,'queued',?,?,?,?,?,?)",
            (job_id, recording_id, kind, now_iso(), model, provider,
             prompt_version, new_id(), now_iso()),
        )
    return job_id


def claim_next(db: Db) -> dict | None:
    """原子领取一个待跑任务。"""
    with db.tx():
        row = db.one(
            "SELECT * FROM job WHERE status IN ('queued','retry')"
            " AND (next_run_at IS NULL OR next_run_at <= ?)"
            " ORDER BY created_at LIMIT 1",
            (now_iso(),),
        )
        if not row:
            return None
        db.exec(
            "UPDATE job SET status='running', started_at=?, attempts=attempts+1 WHERE id=?",
            (now_iso(), row["id"]),
        )
    row["attempts"] = int(row["attempts"]) + 1
    return row


def run_job(deps: WorkerDeps, job: dict) -> None:
    """执行单个任务；成功/失败都落库。"""
    db, cfg = deps.db, deps.cfg
    kind = job["kind"]
    rec_id = job["recording_id"]
    try:
        if kind == "transcribe":
            _run_transcribe(deps, job)
        elif kind == "proofread":
            _run_proofread(deps, job)
        elif kind == "assessment":
            _run_assessment(deps, job)
        else:
            raise ValueError(f"未知任务类型 {kind}")
        with db.tx():
            db.exec("UPDATE job SET status='done', finished_at=? WHERE id=?",
                    (now_iso(), job["id"]))
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        log.warning("job %s 失败：%s", job["id"], msg)
        attempts = int(job["attempts"])
        max_attempts = int(job.get("max_attempts") or cfg.queue.max_attempts)
        retryable = getattr(exc, "retryable", True)
        status = "retry" if (retryable and attempts < max_attempts) else "failed"
        delay = min(60, 2 ** attempts)
        with db.tx():
            db.exec(
                "UPDATE job SET status=?, error_msg=?, finished_at=?,"
                " next_run_at=datetime('now', ?) WHERE id=?",
                (status, msg, now_iso(),
                 f"+{delay} seconds" if status == "retry" else None, job["id"]),
            )
            if status == "failed":
                db.exec("UPDATE recording SET status='failed', error_msg=? WHERE id=?",
                        (msg, rec_id))
                db.audit("system", "job_failed", "job", job["id"], recording_id=rec_id,
                         after={"error": msg})


# ---------------- transcribe ----------------
def _run_transcribe(deps: WorkerDeps, job: dict) -> None:
    db, cfg = deps.db, deps.cfg
    rec_id = job["recording_id"]
    rec = db.one("SELECT * FROM recording WHERE id=?", (rec_id,))
    if not rec:
        raise AdapterError(f"录音不存在：{rec_id}", retryable=False)

    with db.tx():
        db.exec("UPDATE recording SET status='transcribing' WHERE id=?", (rec_id,))

    original = Path(rec["original_path"])
    wav = Path(rec["audio_path"]) if rec.get("audio_path") else Path(
        cfg.data_dir) / "audio" / f"{rec_id}.wav"
    if not wav.exists():
        normalize_to_wav(original, wav)
        # 归一化后才回填 audio_path：创建时不预写路径，避免把「未归一化」误当「已就绪」
        with db.tx():
            db.exec("UPDATE recording SET audio_path=? WHERE id=?", (str(wav), rec_id))
    duration_ms = probe_duration_ms(wav)
    if duration_ms:
        with db.tx():
            db.exec("UPDATE recording SET duration_ms=? WHERE id=?", (duration_ms, rec_id))
    # ffprobe 未安装时 duration_ms=0，拿到 ASR 句子后按最后一句 end 回退（见下方）

    hotwords = _glossary_terms(db, rec_id)
    job_id = deps.asr.submit(wav, hotwords)
    state = _wait_asr(deps.asr, job_id, timeout_s=cfg.asr.timeout_s * 60)
    if state.status != "done":
        raise AdapterError(f"ASR 任务未成功：{state.error or state.status}")
    raw = deps.asr.fetch(job_id)
    if not raw:
        raise AdapterError("ASR 返回空转写", retryable=False)

    caps = deps.asr.capabilities
    # 说话人对齐（§7.5）
    blocks = align_mod.blocks_from_labels(raw) if caps.has_speaker else []
    aligned = align_mod.assign_speakers(raw, blocks, cfg.diarization)

    with db.tx():
        db.exec("DELETE FROM segment WHERE recording_id=?", (rec_id,))
        db.exec("DELETE FROM speaker_cluster WHERE recording_id=?", (rec_id,))

        label_to_cluster: dict[str, int] = {}
        for a in aligned:
            label = a.speaker_label or "UNKNOWN"
            if label not in label_to_cluster:
                cur = db.exec(
                    "INSERT INTO speaker_cluster (recording_id, label, needs_review)"
                    " VALUES (?,?,?)", (rec_id, label, 1 if a.needs_review else 0))
                label_to_cluster[label] = int(cur.lastrowid)
            elif a.needs_review:
                db.exec("UPDATE speaker_cluster SET needs_review=1 WHERE id=?",
                        (label_to_cluster[label],))

        warnings: list[str] = []
        for seq, a in enumerate(aligned, start=1):
            low_conf = (
                a.conf is not None
                and a.conf < cfg.proofread.low_confidence_threshold
            )
            db.exec(
                "INSERT INTO segment (recording_id, seq, start_ms, end_ms, speaker_id,"
                " asr_text, asr_confidence, needs_review_speaker, low_confidence)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (rec_id, seq, int(a.start_s * 1000), int(a.end_s * 1000),
                 label_to_cluster.get(a.speaker_label or "UNKNOWN"),
                 a.text, a.conf, 1 if a.needs_review else 0, 1 if low_conf else 0),
            )
            if a.review_reason and a.needs_review:
                warnings.append(f"seq{seq}: {a.review_reason}")

        # 只有 WAV 头也读不出时长时才用最后一句 end 兜底：
        # ASR 在短音频/纯音上会幻觉出远超文件长度的时间戳，优先级最低。
        if not duration_ms:
            fallback = max((int(a.end_s * 1000) for a in aligned if a.end_s > 0), default=0)
            if fallback > 0:
                duration_ms = fallback
                db.exec("UPDATE recording SET duration_ms=? WHERE id=?", (duration_ms, rec_id))

        db.exec(
            "UPDATE recording SET status='transcribed', duration_ms=?, asr_adapter=?,"
            " asr_capabilities=?, diarizer_adapter=?, warnings=?, error_msg=NULL"
            " WHERE id=?",
            (duration_ms, deps.asr.name,
             json.dumps(caps.as_dict(), ensure_ascii=False),
             deps.diarizer.name,
             json.dumps(warnings, ensure_ascii=False) if warnings else None,
             rec_id),
        )
        db.audit("system", "transcribed", "recording", rec_id, recording_id=rec_id,
                 after={"segments": len(aligned), "adapter": deps.asr.name})


def _wait_asr(asr, job_id: str, *, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while True:
        state = asr.poll(job_id)
        if state.status in ("done", "failed"):
            return state
        if time.monotonic() > deadline:
            raise AdapterError(f"ASR 轮询超时（>{timeout_s}s）")
        time.sleep(1.0)


# ---------------- proofread ----------------
def _run_proofread(deps: WorkerDeps, job: dict) -> None:
    db, cfg = deps.db, deps.cfg
    rec_id = job["recording_id"]
    with db.tx():
        db.exec("UPDATE recording SET status='proofreading' WHERE id=?", (rec_id,))

    segs = db.all(
        "SELECT s.*, c.label AS speaker_label FROM segment s"
        " LEFT JOIN speaker_cluster c ON c.id=s.speaker_id"
        " WHERE s.recording_id=? ORDER BY s.seq", (rec_id,))
    if not segs:
        raise AdapterError("没有可校对的句子", retryable=False)

    glossary = _glossary_pairs(db, rec_id)
    window = cfg.proofread.window_segments
    results = []
    for i, seg in enumerate(segs):
        ctx = [s["asr_text"] for s in segs[max(0, i - window):i]]
        r = proofread_segment(
            seg["asr_text"], llm=deps.llm, context=ctx, glossary=glossary,
            asr_confidence=seg["asr_confidence"],
            low_confidence_threshold=cfg.proofread.low_confidence_threshold,
            json_mode=cfg.llm.json_mode,
        )
        results.append((seg, r))

    with db.tx():
        for seg, r in results:
            db.exec(
                "UPDATE segment SET ai_text=?, ai_edits=? WHERE id=?",
                (r.corrected_text, r.edits_json, seg["id"]),
            )
        db.exec("UPDATE recording SET status='proofread' WHERE id=?", (rec_id,))
        db.audit("system", "proofread", "recording", rec_id, recording_id=rec_id,
                 after={"segments": len(results)})


# ---------------- assessment ----------------
def _run_assessment(deps: WorkerDeps, job: dict) -> None:
    db, cfg = deps.db, deps.cfg
    rec_id = job["recording_id"]
    rec = db.one("SELECT * FROM recording WHERE id=?", (rec_id,))
    if not rec:
        raise AdapterError("录音不存在", retryable=False)
    with db.tx():
        db.exec("UPDATE recording SET status='parsing' WHERE id=?", (rec_id,))

    fw_row = db.one(
        "SELECT * FROM framework WHERE code=? ORDER BY version DESC LIMIT 1",
        ("shift_handover_assessment",))
    if not fw_row:
        raise AdapterError("未找到评估框架", retryable=False)
    fw = FrameworkDef.from_dict(
        {
            "code": fw_row["code"],
            "name": fw_row["name"],
            "version": fw_row["version"],
            "fields": jloads(fw_row["fields"], []),
        }
    )

    segs = db.all(
        "SELECT s.seq, COALESCE(s.final_text, s.ai_text, s.asr_text) AS text,"
        " c.label AS speaker_label FROM segment s"
        " LEFT JOIN speaker_cluster c ON c.id=s.speaker_id"
        " WHERE s.recording_id=? ORDER BY s.seq", (rec_id,))
    values = parse_assessment(fw, segs, llm=deps.llm, json_mode=cfg.llm.json_mode)
    ok, reasons = can_submit(values, fw)

    with db.tx():
        cur = db.exec(
            "INSERT INTO assessment (recording_id, framework_id, framework_version,"
            " framework_snapshot, status, created_at, updated_at)"
            " VALUES (?,?,?,?,'draft',?,?)",
            (rec_id, fw_row["id"], fw_row["version"],
             json.dumps(fw.as_dict(), ensure_ascii=False), now_iso(), now_iso()))
        aid = int(cur.lastrowid)
        for v in values:
            db.exec(
                "INSERT INTO assessment_field_value (assessment_id, field_key, content,"
                " citations, confidence, source, status) VALUES (?,?,?,?,?,?,?)",
                (aid, v.key, v.content, json.dumps(v.citations), v.confidence,
                 v.source, v.status))
        db.exec("UPDATE recording SET status='done' WHERE id=?", (rec_id,))
        db.audit("system", "assessment_parsed", "assessment", aid, recording_id=rec_id,
                 after={"submittable": ok, "reasons": reasons})


# ---------------- 辅助 ----------------
def _glossary_terms(db: Db, rec_id: str) -> list[str]:
    return [r["term"] for r in db.all(
        "SELECT term FROM glossary WHERE enabled=1 ORDER BY id LIMIT 500")]


def _glossary_pairs(db: Db, rec_id: str) -> list[tuple[str, list[str]]]:
    out = []
    for r in db.all("SELECT term, variants FROM glossary WHERE enabled=1"):
        out.append((r["term"], jloads(r["variants"], []) or []))
    return out


class Worker:
    """单线程轮询循环，可 in-process 启动或独立进程跑。"""

    def __init__(self, deps: WorkerDeps) -> None:
        self.deps = deps
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> bool:
        job = claim_next(self.deps.db)
        if not job:
            return False
        run_job(self.deps, job)
        return True

    def run_forever(self) -> None:
        log.info("worker 启动，轮询间隔 %.1fs", self.deps.cfg.queue.poll_interval_s)
        while not self._stop.is_set():
            try:
                if not self.run_once():
                    time.sleep(self.deps.cfg.queue.poll_interval_s)
            except Exception:  # noqa: BLE001
                log.exception("worker 主循环异常")
                time.sleep(self.deps.cfg.queue.poll_interval_s)
