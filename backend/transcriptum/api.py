"""FastAPI 接口层（§9 API 设计）。

原始音频播放必须支持 Range（浏览器要 seek）—— 不支持的话校对台无法逐句跳播。
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from .adapters.registry import build_asr, build_diarizer, build_llm
from .config import Config
from .db import Db, connect, init_db
from .exporters import ExportOptions, export
from .service import (
    Conflict,
    NotFound,
    RecordingService,
    ValidationFailed,
    seed_frameworks,
)
from .worker import WorkerDeps, enqueue


def create_app(cfg: Config | None = None, *, start_worker: bool = False) -> FastAPI:
    cfg = cfg or Config.load(None)
    conn = connect(cfg.db_path)
    init_db(conn)
    db = Db(conn)
    with db.tx():
        seed_frameworks(db)

    deps = WorkerDeps.build(cfg, db)
    svc = RecordingService(cfg, db, deps)

    app = FastAPI(title="Transcriptum", version="0.1.0")
    app.state.cfg = cfg
    app.state.db = db
    app.state.svc = svc
    # 暴露给测试与运维脚本：复用同进程适配器，避免重复建连
    app.state.worker_deps = deps
    app.state.cfg = cfg
    app.state.db = db

    # 本地前端开发用；生产走同源
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def handle(exc: Exception):
        if isinstance(exc, NotFound):
            raise HTTPException(404, str(exc))
        if isinstance(exc, Conflict):
            raise HTTPException(409, str(exc))
        if isinstance(exc, ValidationFailed):
            raise HTTPException(422, {"message": str(exc), "reasons": exc.reasons})
        raise exc

    # ---------------- 录音 ----------------
    @app.post("/api/recordings")
    async def create_recording(
        title: str = Form(...),
        kind: str = Form("meeting"),
        source: str = Form("upload"),
        consent: bool = Form(True),
        file: UploadFile | None = File(None),
    ):
        try:
            if source == "upload":
                if file is None:
                    raise ValidationFailed("upload 类型必须带文件")
                tmpdir = Path(tempfile.mkdtemp(prefix="tsu-"))
                tmp = tmpdir / (Path(file.filename or "audio").name)
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(file.file, fh)
                rec = svc.create(title=title, kind=kind, source=source,
                                 original_path=tmp, consent=consent)
                shutil.rmtree(tmpdir, ignore_errors=True)
            else:
                # 实时录音：前端上传同一路音频，这里同样按 upload 处理落盘
                if file is None:
                    raise ValidationFailed("live 类型需先上传录音文件")
                tmpdir = Path(tempfile.mkdtemp(prefix="tsu-"))
                tmp = tmpdir / (Path(file.filename or "audio.wav").name)
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(file.file, fh)
                rec = svc.create(title=title, kind=kind, source="live",
                                 original_path=tmp, consent=consent)
                shutil.rmtree(tmpdir, ignore_errors=True)
            return rec
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    @app.get("/api/recordings")
    def list_recordings(status: str | None = None, limit: int = Query(50, le=200),
                        offset: int = 0):
        return svc.list(status=status, limit=limit, offset=offset)

    @app.get("/api/recordings/{rec_id}")
    def get_recording(rec_id: str):
        try:
            return svc.get(rec_id)
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    @app.get("/api/recordings/{rec_id}/segments")
    def get_segments(rec_id: str):
        try:
            svc.get(rec_id)
            return {"items": svc.segments(rec_id)}
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    @app.patch("/api/recordings/{rec_id}/segments/{seq}")
    def patch_segment(rec_id: str, seq: int, body: dict = Body(...)):
        # asr_text 是不可变证据链：明确拒绝，不要静默忽略（否则调用方误以为改成功了）
        if "asr_text" in body:
            raise HTTPException(422, "asr_text 不可修改（证据链要求）")
        try:
            return svc.update_segment(
                rec_id, seq,
                final_text=body.get("final_text"),
                review_status=body.get("review_status"),
                speaker_label=body.get("speaker_label"),
                confirm_speaker=bool(body.get("confirm_speaker", False)),
                actor=body.get("actor", "local"),
            )
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    @app.post("/api/recordings/{rec_id}/batch-accept")
    def batch_accept(rec_id: str, body: dict = Body(default={})):
        try:
            return svc.batch_accept(rec_id, min_confidence=body.get("min_confidence"))
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    # ---------------- 流水线 ----------------
    @app.post("/api/recordings/{rec_id}/transcribe")
    def transcribe(rec_id: str):
        try:
            job_id = svc.start_transcribe(rec_id)
            return {"job_id": job_id}
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    @app.post("/api/recordings/{rec_id}/proofread")
    def proofread(rec_id: str):
        try:
            job_id = svc.start_proofread(rec_id)
            return {"job_id": job_id}
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        row = db.one("SELECT * FROM job WHERE id=?", (job_id,))
        if not row:
            raise HTTPException(404, "任务不存在")
        return row

    # ---------------- 音频（Range 播放） ----------------
    @app.get("/api/recordings/{rec_id}/audio")
    def get_audio(rec_id: str, request: Request):
        try:
            rec = svc.get(rec_id)
        except Exception as exc:  # noqa: BLE001
            handle(exc)
        audio = rec.get("audio_path")
        if not audio or not Path(audio).exists():
            raise HTTPException(404, "音频文件不存在（可能已过保留期）")
        return _range_response(Path(audio), request)

    # ---------------- 评估 ----------------
    @app.get("/api/frameworks")
    def frameworks():
        return {"items": svc.frameworks()}

    @app.post("/api/frameworks")
    def create_framework(body: dict = Body(...)):
        try:
            return svc.create_framework(body)
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    @app.post("/api/recordings/{rec_id}/assessments")
    def start_assessment(rec_id: str, body: dict = Body(...)):
        fid = body.get("framework_id")
        if not fid:
            raise HTTPException(422, "framework_id 必填")
        try:
            job_id = svc.start_assessment(rec_id, int(fid))
            return {"job_id": job_id}
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    @app.get("/api/recordings/{rec_id}/assessments")
    def list_assessments(rec_id: str):
        return {"items": svc.assessments(rec_id)}

    @app.patch("/api/assessments/{aid}/fields/{field_key}")
    def patch_field(aid: int, field_key: str, body: dict = Body(...)):
        try:
            return svc.update_field(
                aid, field_key,
                content=str(body.get("content", "")),
                citations=[int(c) for c in body.get("citations", [])],
            )
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    @app.post("/api/assessments/{aid}/submit")
    def submit_assessment(aid: int):
        try:
            return svc.submit_assessment(aid)
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    # ---------------- 导出 ----------------
    @app.get("/api/recordings/{rec_id}/export")
    def do_export(
        rec_id: str,
        fmt: str = Query("md", pattern="^(md|docx|xlsx|assessment_xlsx)$"),
        redact_sensitive: bool = True,
        include_ai_suggestion: bool = True,
    ):
        try:
            rec = svc.get(rec_id)
            opts = ExportOptions(
                include_ai_suggestion=include_ai_suggestion,
                redact_sensitive=redact_sensitive,
                title=rec["title"],
            )
            body, media, filename = export(db, rec_id, rec, fmt, opts)
            db.audit("local", "export", "recording", rec_id, recording_id=rec_id,
                     after={"format": fmt, "redact": redact_sensitive})
            return Response(
                content=body,
                media_type=media,
                headers={"Content-Disposition": _content_disposition(filename)},
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            handle(exc)

    # ---------------- 运维 ----------------
    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "asr": {"adapter": cfg.asr.adapter, "capabilities": deps.asr.capabilities.as_dict()},
            "llm": {"adapter": cfg.llm.adapter, "model": cfg.llm.model},
            "diarizer": cfg.diarizer.adapter,
            "proofread": {
                "low_confidence_threshold": cfg.proofread.low_confidence_threshold,
                "prompt_version": cfg.proofread.prompt_version,
            },
        }

    @app.get("/api/recordings/{rec_id}/audit")
    def audit_trail(rec_id: str, limit: int = Query(200, ge=1, le=1000)):
        """审计轨迹（§10）：谁在何时看了/改了哪句。"""
        items = db.all(
            "SELECT id, actor, action, entity, entity_id, before, after, at"
            " FROM audit_log WHERE recording_id=? ORDER BY id DESC LIMIT ?",
            (rec_id, limit))
        return {"items": items, "total": len(items)}

    @app.post("/api/admin/purge-expired")
    def purge():
        return svc.purge_expired()

    if start_worker:
        import threading

        from .worker import Worker

        w = Worker(deps)
        threading.Thread(target=w.run_forever, daemon=True).start()
        app.state.worker = w

    return app


_MEDIA_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
                ".flac": "audio/flac", ".ogg": "audio/ogg"}


def _range_response(path: Path, request: Request) -> Response:
    """带 Range 支持的文件响应，浏览器才能 seek。"""
    file_size = path.stat().st_size
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    range_header = request.headers.get("range") or request.headers.get("Range")

    if not range_header:
        return FileResponse(path, media_type=media_type,
                            headers={"Accept-Ranges": "bytes"})

    m = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not m:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
    start_s, end_s = m.group(1), m.group(2)
    if start_s == "" and end_s == "":
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
    if start_s == "":
        length = int(end_s)
        start = max(0, file_size - length)
        end = file_size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1
    end = min(end, file_size - 1)
    if start > end or start >= file_size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    def iter_file():
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = fh.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        iter_file(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        },
    )


def _content_disposition(filename: str) -> str:
    """RFC 5987 编码文件名。

    HTTP 头只能是 latin-1，中文标题直接放进去会抛 UnicodeEncodeError，
    导致「中文标题的录音无法导出」。这里给出 ASCII 回退名 + filename*。
    """
    ascii_fallback = filename.encode("ascii", "ignore").decode("ascii") or "export"
    ascii_fallback = re.sub(r'[\\/:*?"<>|\r\n]', "_", ascii_fallback)
    quoted = quote(filename, safe="")
    return f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{quoted}"
