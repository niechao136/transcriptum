"""逐句保守校对（§4.3）。

核心约束（文档里的几条硬要求，这里逐条落实）：
1. 把 ASR 原文和少量上下文交给**本地小模型**，要求返回结构化修改建议；
2. **只替换，不重写**：不允许模型自由发挥，一次改多处时保留逐处 diff；
3. 医学术语/数值/药名/剂量**不得改动** —— 命中受保护 token 的建议直接丢弃；
4. 修改次数超过长度的 30% 视为「模型开始编」，整句只告警不采纳；
5. 低置信度 + 改动大 → 强制人工（批量采纳只覆盖低风险）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .adapters.base import LlmAdapter

# 受保护 token：数字、剂量、单位、检验值、药名常见后缀、英文缩写
_PROTECTED_RE = re.compile(
    r"""
    \d+(?:\.\d+)?\s*(?:mg|g|ml|mL|μg|ug|kg|cm|mm|mmol|mol|IU|U|次|片|粒|支|滴|小时|天|周|月|%|℃)
  | \d+(?:\.\d+)?
  | [A-Za-z][A-Za-z0-9-]*[A-Z0-9][A-Za-z0-9-]*   # 含大写或数字的缩写：CT / HbA1c / COVID-19
  | [a-zA-Z]+_[a-zA-Z0-9]+
    """,
    re.VERBOSE,
)

_SYSTEM_PROMPT = """你是一个严谨的中文会议转写校对助手。

规则（必须遵守）：
1. 只做最小必要修正：同音字、明显错别字、术语正确写法、标点缺失。
2. 不得改写句式、不得增删信息、不得总结、不得推测说话人意图。
3. 数字、剂量、单位、检验值、药名、英文缩写一律不得改动。
4. 如果一句话没有明显错误，edits 返回空数组。
5. 只输出 JSON，不要任何解释文字。

输出格式：
{"corrected_text": "修正后的完整句子", "edits": [{"type": "replace", "from": "原文片段", "to": "修正片段", "reason": "同音字错写", "confidence": 0.0-1.0}]}
"""

_MAX_EDIT_RATIO = 0.30


@dataclass
class Edit:
    type: str
    frm: str
    to: str
    reason: str
    confidence: float

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "from": self.frm,
            "to": self.to,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class ProofResult:
    corrected_text: str
    edits: list[Edit]
    accepted: bool            # 是否建议采纳
    requires_review: bool     # 是否必须人工逐句看
    warning: str | None = None

    @property
    def edits_json(self) -> str:
        return json.dumps([e.as_dict() for e in self.edits], ensure_ascii=False)


def protected_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _PROTECTED_RE.finditer(text)]


def build_user_prompt(source: str, context: list[str], glossary: list[tuple[str, list[str]]]) -> str:
    parts = []
    if glossary:
        lines = []
        for term, variants in glossary:
            if variants:
                lines.append(f"- {'/'.join(variants)} → {term}")
            else:
                lines.append(f"- 正确写法：{term}")
        parts.append("术语表（出现左侧写法必须改成右侧）：\n" + "\n".join(lines))
    if context:
        parts.append(
            "上下文（仅供理解，不要校对、不要输出）：\n" + "\n".join(f"· {c}" for c in context)
        )
    parts.append(f"需校对的句子：\n<source>{source}</source>")
    return "\n\n".join(parts)


def _parse_llm_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        seg = text.split("```")
        text = seg[1] if len(seg) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("LLM 输出不是 JSON 对象")
    return obj


def proofread_segment(
    source: str,
    *,
    llm: LlmAdapter,
    context: list[str] | None = None,
    glossary: list[tuple[str, list[str]]] | None = None,
    asr_confidence: float | None = None,
    low_confidence_threshold: float = 0.6,
    json_mode: bool = True,
) -> ProofResult:
    """校对单句。任何异常都不向上抛，降级为「保留原文 + 标待复核」。"""
    context = context or []
    glossary = glossary or []
    no_warning_fallback = ProofResult(
        corrected_text=source,
        edits=[],
        accepted=False,
        requires_review=True,
        warning="llm_unavailable:模型调用失败，保留原文",
    )

    try:
        raw = llm.chat(
            _SYSTEM_PROMPT,
            build_user_prompt(source, context, glossary),
            json_mode=json_mode,
        )
        obj = _parse_llm_json(raw)
    except Exception as exc:  # noqa: BLE001 - 降级路径必须覆盖所有异常
        no_warning_fallback.warning = f"llm_unavailable:{type(exc).__name__}"
        return no_warning_fallback

    corrected = str(obj.get("corrected_text") or "").strip()
    raw_edits = obj.get("edits") or []
    if not isinstance(raw_edits, list):
        raw_edits = []

    edits: list[Edit] = []
    dropped: list[str] = []
    protected = set(protected_tokens(source))

    for item in raw_edits:
        if not isinstance(item, dict):
            continue
        frm = str(item.get("from", ""))
        to = str(item.get("to", ""))
        if not frm or frm == to:
            continue
        # 规则 4：受保护 token 被改动 → 丢弃该条建议
        if frm in protected and to != frm:
            dropped.append(f"protected:{frm}")
            continue
        if _changes_protected(frm, to, protected):
            dropped.append(f"protected:{frm}->{to}")
            continue
        try:
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        edits.append(
            Edit(
                type=str(item.get("type", "replace")),
                frm=frm,
                to=to,
                reason=str(item.get("reason", "")),
                confidence=max(0.0, min(1.0, conf)),
            )
        )

    # 只接受能在原文里定位到的替换，保证「只替换不重写」
    applicable = [e for e in edits if e.frm in source]
    if len(applicable) != len(edits):
        dropped.append("unlocatable_edit")

    result_text = source
    for e in applicable:
        result_text = result_text.replace(e.frm, e.to, 1)

    # 规则 4 兜底：模型给的 corrected_text 与「逐处替换」结果不一致 → 以逐处替换为准
    warning = None
    if corrected and corrected != result_text:
        warning = "model_rewrote:模型改写了原文，已回退为逐处替换结果"
        dropped.append("rewrite")

    if dropped:
        prefix = ";".join(dict.fromkeys(dropped))
        warning = f"{prefix}" if warning is None else f"{warning};{prefix}"

    # 规则 5：改动幅度过大 → 视为编造，不采纳
    # 用「最终结果相对原文的净长度变化」而不是逐条替换的长度之和：
    # 同音字替换（如「合砸」→「合闸」）长度不变，不应被误判为大幅改写；
    # 而「好」→「好的没问题」净增 4 字，相对原文翻倍，必须拒绝。
    max_len = max(1, len(source))
    changed = abs(len(result_text) - len(source))
    if changed / max_len > _MAX_EDIT_RATIO:
        return ProofResult(
            corrected_text=source,
            edits=applicable,
            accepted=False,
            requires_review=True,
            warning=(warning + ";" if warning else "") + "edit_ratio_too_high:改动幅度超 30%，疑似改写，不采纳",
        )

    requires_review = bool(applicable)
    if asr_confidence is not None and asr_confidence < low_confidence_threshold:
        requires_review = True

    return ProofResult(
        corrected_text=result_text,
        edits=applicable,
        accepted=bool(applicable),
        requires_review=requires_review,
        warning=warning,
    )


def _changes_protected(frm: str, to: str, protected: set[str]) -> bool:
    """替换片段里含受保护 token 且该 token 在新片段中消失 → 拒绝。"""
    for tok in protected:
        if tok in frm and tok not in to:
            return True
    return False


def is_low_risk(result: ProofResult, *, min_confidence: float = 0.85) -> bool:
    """批量采纳判定：改动小、置信度高、无告警。"""
    if not result.accepted or result.warning or not result.edits:
        return False
    return all(e.confidence >= min_confidence for e in result.edits)
