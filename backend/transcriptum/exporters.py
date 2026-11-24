"""导出：docx / xlsx / md（§9）。

合规要求（§10）：
- 导出时必须能标记「已人工确认」与「AI 建议未确认」，不能把 AI 结果伪装成定稿；
- 默认对敏感内容脱敏（数字/编号可置换）；
- docx 用 python-docx，xlsx 用 openpyxl，缺失时降级为 markdown 并显式告知。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

REDACT_RE = re.compile(
    r"(?:1[3-9]\d{9})"            # 手机号
    r"|(?:\d{17}[\dXx])"          # 身份证
    r"|(?:[\w.+-]+@[\w-]+\.[\w.]+)"  # 邮箱
)


def redact(text: str) -> str:
    return REDACT_RE.sub("〔已脱敏〕", text)


def _need(module: str):
    try:
        return __import__(module)
    except ImportError:  # pragma: no cover - 依赖缺失时的明确降级
        return None


@dataclass
class ExportOptions:
    include_ai_suggestion: bool = True
    redact_sensitive: bool = True
    title: str = ""


def _segments_rows(db, rec_id: str) -> list[dict]:
    rows = db.all(
        "SELECT s.seq, s.start_ms, s.end_ms, c.label AS speaker_label,"
        " s.asr_text, s.ai_text, s.final_text, s.review_status,"
        " s.needs_review_speaker, s.low_confidence"
        " FROM segment s LEFT JOIN speaker_cluster c ON c.id=s.speaker_id"
        " WHERE s.recording_id=? ORDER BY s.seq", (rec_id,))
    return rows


def _effective_text(row: dict) -> str:
    return row["final_text"] or row["ai_text"] or row["asr_text"]


def _status_label(row: dict) -> str:
    if row["final_text"]:
        return "已确认"
    if row["review_status"] == "accepted":
        return "已确认"
    if row["review_status"] == "rejected":
        return "已驳回"
    return "AI 建议未确认"


def _ts(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def to_markdown(db, rec_id: str, rec: dict, opts: ExportOptions) -> str:
    rows = _segments_rows(db, rec_id)
    title = opts.title or rec["title"]
    lines = [f"# {title}", ""]
    lines.append(f"- 类型：{'会议' if rec['kind'] == 'meeting' else '考核评估'}")
    lines.append(f"- 创建时间：{rec['created_at']}")
    lines.append(f"- 时长：{_ts(rec.get('duration_ms') or 0)}")
    lines.append("")
    lines.append("> 说明：标注「AI 建议未确认」的句子未经人工确认，不可作为定稿依据。")
    lines.append("")
    lines.append("| 时间 | 说话人 | 内容 | 状态 |")
    lines.append("| --- | --- | --- | --- |")
    for r in rows:
        text = _effective_text(r)
        if opts.redact_sensitive:
            text = redact(text)
        text = text.replace("|", "\\|")
        spk = r["speaker_label"] or "?"
        flag = ""
        if r["needs_review_speaker"]:
            flag = " ⚠说话人待确认"
        lines.append(
            f"| {_ts(r['start_ms'])} | {spk}{flag} | {text} | {_status_label(r)} |"
        )
    return "\n".join(lines) + "\n"


def to_docx(db, rec_id: str, rec: dict, opts: ExportOptions) -> bytes | None:
    docx = _need("docx")
    if docx is None:
        return None
    from docx import Document
    from docx.shared import Pt

    rows = _segments_rows(db, rec_id)
    doc = Document()
    doc.add_heading(opts.title or rec["title"], level=1)
    doc.add_paragraph(f"类型：{'会议' if rec['kind'] == 'meeting' else '考核评估'}")
    doc.add_paragraph(f"创建时间：{rec['created_at']}　时长：{_ts(rec.get('duration_ms') or 0)}")
    note = doc.add_paragraph("说明：标注「AI 建议未确认」的句子未经人工确认，不可作为定稿依据。")
    note.runs[0].font.size = Pt(9)

    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, name in enumerate(["时间", "说话人", "内容", "状态"]):
        hdr[i].text = name
    for r in rows:
        cells = table.add_row().cells
        cells[0].text = _ts(r["start_ms"])
        spk = r["speaker_label"] or "?"
        if r["needs_review_speaker"]:
            spk += "（待确认）"
        cells[1].text = spk
        text = _effective_text(r)
        cells[2].text = redact(text) if opts.redact_sensitive else text
        cells[3].text = _status_label(r)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def to_xlsx(db, rec_id: str, rec: dict, opts: ExportOptions) -> bytes | None:
    if _need("openpyxl") is None:
        return None
    from openpyxl import Workbook

    rows = _segments_rows(db, rec_id)
    wb = Workbook()
    ws = wb.active
    ws.title = "转写"
    ws.append(["序号", "开始", "结束", "说话人", "内容", "状态", "说话人待确认", "低置信度"])
    for r in rows:
        text = _effective_text(r)
        ws.append([
            r["seq"], _ts(r["start_ms"]), _ts(r["end_ms"]),
            r["speaker_label"] or "?",
            redact(text) if opts.redact_sensitive else text,
            _status_label(r),
            "是" if r["needs_review_speaker"] else "",
            "是" if r["low_confidence"] else "",
        ])
    ws.freeze_panes = "A2"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def to_assessment_xlsx(db, rec_id: str, opts: ExportOptions) -> bytes | None:
    """考核评估导出：字段 + 内容 + 引用句子 + 确认状态。"""
    if _need("openpyxl") is None:
        return None
    from openpyxl import Workbook

    assessments = db.all(
        "SELECT a.*, f.name AS framework_name FROM assessment a"
        " JOIN framework f ON f.id=a.framework_id"
        " WHERE a.recording_id=? ORDER BY a.id", (rec_id,))
    if not assessments:
        return None

    wb = Workbook()
    ws = wb.active
    ws.title = "评估结果"
    ws.append(["评估ID", "框架", "版本", "状态", "字段", "内容", "引用句子", "来源", "校验状态"])
    for a in assessments:
        values = db.all(
            "SELECT * FROM assessment_field_value WHERE assessment_id=? ORDER BY id", (a["id"],))
        for v in values:
            cites = v["citations"] or "[]"
            import json as _json
            try:
                cites = ",".join(str(x) for x in _json.loads(cites))
            except (ValueError, TypeError):
                cites = ""
            content = v["content"] or ""
            ws.append([
                a["id"], a["framework_name"], a["framework_version"], a["status"],
                v["field_key"],
                redact(content) if opts.redact_sensitive else content,
                cites, v["source"], v["status"],
            ])
    ws.freeze_panes = "A2"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def export(db, rec_id: str, rec: dict, fmt: str, opts: ExportOptions):
    """返回 (bytes, media_type, filename)。"""
    fmt = fmt.lower()
    if fmt == "md":
        body = to_markdown(db, rec_id, rec, opts).encode("utf-8")
        return body, "text/markdown; charset=utf-8", f"{_safe(rec['title'])}.md"
    if fmt == "docx":
        body = to_docx(db, rec_id, rec, opts)
        if body is None:
            raise RuntimeError("python-docx 未安装，docx 导出不可用；请改用 md")
        return (body,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                f"{_safe(rec['title'])}.docx")
    if fmt == "xlsx":
        body = to_xlsx(db, rec_id, rec, opts)
        if body is None:
            raise RuntimeError("openpyxl 未安装，xlsx 导出不可用；请改用 md")
        return (body,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                f"{_safe(rec['title'])}.xlsx")
    if fmt == "assessment_xlsx":
        body = to_assessment_xlsx(db, rec_id, opts)
        if body is None:
            raise RuntimeError("openpyxl 未安装或该录音无评估结果")
        return (body,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                f"{_safe(rec['title'])}-评估.xlsx")
    raise ValueError(f"不支持的导出格式：{fmt}")


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)[:80] or "export"
