"""评估（考核）记录解析与校验（§6）。

四条硬约束：
1. 每个字段必须带 citations（segment_id 列表），无引用 → missing_citation，不可提交；
2. 框架字段版本化，历史评估用快照校验，不因字段修改而失效；
3. 字段类型校验（enum/number/text），越界不静默通过；
4. 解析失败/缺失 → not_extracted，重试由队列负责，不填假值。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .adapters.base import LlmAdapter


class AssessmentError(RuntimeError):
    pass


@dataclass
class FieldDef:
    key: str
    label: str
    type: str = "text"        # text | number | enum | list
    required: bool = False
    options: list[str] = field(default_factory=list)
    guidance: str = ""


@dataclass
class FrameworkDef:
    code: str
    name: str
    version: int
    fields: list[FieldDef]

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "version": self.version,
            "fields": [
                {
                    "key": f.key,
                    "label": f.label,
                    "type": f.type,
                    "required": f.required,
                    "options": f.options,
                    "guidance": f.guidance,
                }
                for f in self.fields
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "FrameworkDef":
        return FrameworkDef(
            code=d["code"],
            name=d["name"],
            version=int(d.get("version", 1)),
            fields=[FieldDef(**f) for f in d.get("fields", [])],
        )


@dataclass
class FieldValue:
    key: str
    content: str
    citations: list[int]
    confidence: float | None = None
    source: str = "ai"
    status: str = "ok"

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "content": self.content,
            "citations": self.citations,
            "confidence": self.confidence,
            "source": self.source,
            "status": self.status,
        }


_SYSTEM_ASSESSMENT = """你是一个严格的考核记录提取器，标记为 ASSESSMENT。

只从给定的会议转写内容中提取事实，绝不推测或补充。
每个字段必须给出 citations：即支撑该结论的句子序号（segment seq）列表。
如果某个字段在转写里没有依据，content 返回空字符串、citations 返回空数组。

只输出 JSON：
{"fields": [{"key": "...", "content": "...", "citations": [1,2], "confidence": 0.0-1.0}]}
"""


def build_assessment_prompt(fw: FrameworkDef, segments: list[dict]) -> str:
    lines = ["需要提取的字段："]
    for f in fw.fields:
        opt = f"，可选值：{'/'.join(f.options)}" if f.options else ""
        req = "（必填）" if f.required else ""
        lines.append(f"- {f.key} | {f.label} | 类型 {f.type}{opt}{req} | {f.guidance}")
    body = "\n".join(f"[{s['seq']}] 说话人{s.get('speaker_label') or '?'}: {s['text']}"
                     for s in segments)
    return "\n".join(lines) + "\n\n会议转写（方括号内是 seq）：\n" + body


def parse_assessment(
    fw: FrameworkDef,
    segments: list[dict],
    *,
    llm: LlmAdapter,
    json_mode: bool = True,
) -> list[FieldValue]:
    """调用 LLM 抽取并按框架校验。异常上抛由队列重试。"""
    raw = llm.chat(_SYSTEM_ASSESSMENT, build_assessment_prompt(fw, segments),
                   json_mode=json_mode)
    text = raw.strip()
    if text.startswith("```"):
        seg = text.split("```")
        text = seg[1] if len(seg) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssessmentError(f"评估输出不是合法 JSON：{exc}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("fields"), list):
        raise AssessmentError("评估输出缺少 fields 数组")

    valid_seqs = {int(s["seq"]) for s in segments}
    by_key: dict[str, dict] = {}
    for item in obj["fields"]:
        if isinstance(item, dict) and item.get("key"):
            by_key[str(item["key"])] = item

    out: list[FieldValue] = []
    for f in fw.fields:
        item = by_key.get(f.key)
        if item is None:
            out.append(FieldValue(key=f.key, content="", citations=[],
                                  status="not_extracted"))
            continue

        content = str(item.get("content") or "").strip()
        cites_raw = item.get("citations") or []
        if not isinstance(cites_raw, list):
            cites_raw = []
        cites: list[int] = []
        for c in cites_raw:
            try:
                ci = int(c)
            except (TypeError, ValueError):
                continue
            if ci in valid_seqs and ci not in cites:
                cites.append(ci)

        try:
            conf = float(item.get("confidence")) if item.get("confidence") is not None else None
        except (TypeError, ValueError):
            conf = None

        status = "ok"
        if not content:
            status = "not_extracted"
        elif not cites:
            # 无引用不可提交（§6 关键约束）
            status = "missing_citation"
        elif not validate_value(f, content):
            status = "invalid_value"

        out.append(
            FieldValue(key=f.key, content=content, citations=cites,
                       confidence=conf, source="ai", status=status)
        )
    return out


def validate_value(f: FieldDef, content: str) -> bool:
    if not content:
        return False
    if f.type == "number":
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", content.strip()))
    if f.type == "enum":
        return not f.options or content.strip() in f.options
    return True


def can_submit(values: list[FieldValue], fw: FrameworkDef) -> tuple[bool, list[str]]:
    """提交前校验：必填字段齐全 + 所有有内容的字段都必须有引用。"""
    reasons: list[str] = []
    by_key = {v.key: v for v in values}
    for f in fw.fields:
        v = by_key.get(f.key)
        if v is None or not v.content:
            if f.required:
                reasons.append(f"{f.label}（{f.key}）为必填但未提取到内容")
            continue
        if not v.citations:
            reasons.append(f"{f.label}（{f.key}）缺少引用，不可提交")
        if not validate_value(f, v.content):
            reasons.append(f"{f.label}（{f.key}）取值不符合类型 {f.type}：{v.content!r}")
    return (not reasons, reasons)


BUILTIN_FRAMEWORKS: list[FrameworkDef] = [
    FrameworkDef(
        code="shift_handover_assessment",
        name="交班考核",
        version=1,
        fields=[
            FieldDef("summary", "交班要点", "text", True,
                     guidance="本次交班确认的关键事项，一到两句"),
            FieldDef("risk_points", "风险项", "text", False,
                     guidance="未闭环的风险或异常，没有则留空"),
            FieldDef("todo", "待办事项", "text", True,
                     guidance="明确的后续动作与责任人"),
            FieldDef("score", "评分", "number", False,
                     guidance="如会议中有明确打分则填写，否则留空"),
        ],
    ),
    FrameworkDef(
        code="safety_briefing_assessment",
        name="安全交底考核",
        version=1,
        fields=[
            FieldDef("hazard_identified", "识别的危险源", "text", True,
                     guidance="交底中明确提到的危险源"),
            FieldDef("measures", "控制措施", "text", True,
                     guidance="针对危险源的具体措施"),
            FieldDef("responsible", "责任人", "text", False, guidance="措施责任人"),
            FieldDef(
                "completeness", "交底完整度", "enum", True,
                options=["完整", "基本完整", "不完整"],
                guidance="按是否覆盖危险源/措施/责任人判断",
            ),
        ],
    ),
]
