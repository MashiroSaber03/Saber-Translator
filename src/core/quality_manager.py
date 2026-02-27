"""
翻译质量分析管理器

提供:
- 会话质量分析
- 报告持久化与读取
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.core.glossary_manager import list_entries
from src.core.page_storage import load_session
from src.shared.path_helpers import resource_path
from src.shared.security import validate_relative_path, normalize_rel_path

logger = logging.getLogger("QualityManager")

_REPORT_DIR = resource_path(os.path.join("data", "quality", "reports"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _session_report_file(session_path: str) -> str:
    session_hash = hashlib.sha1(session_path.encode("utf-8")).hexdigest()  # nosec B324
    os.makedirs(_REPORT_DIR, exist_ok=True)
    return os.path.join(_REPORT_DIR, f"{session_hash}.json")


def _safe_len(text: Optional[str]) -> int:
    return len((text or "").strip())


def _extract_book_id(session_path: str) -> Optional[str]:
    normalized = normalize_rel_path(session_path)
    parts = normalized.split("/")
    if len(parts) >= 2 and parts[0] == "bookshelf":
        return parts[1]
    return None


def _extract_bubbles(page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    bubbles: List[Dict[str, Any]] = []

    # 优先使用 bubbleStates（前端标准字段）
    bubble_states = page_data.get("bubbleStates")
    if isinstance(bubble_states, list):
        for index, state in enumerate(bubble_states):
            if not isinstance(state, dict):
                continue
            bubbles.append(
                {
                    "bubble_index": index,
                    "original_text": str(state.get("originalText", "") or ""),
                    "translated_text": str(state.get("translatedText", "") or ""),
                    "ocr_confidence": state.get("ocrConfidence", state.get("confidence", state.get("score"))),
                }
            )
        return bubbles

    # 兼容后端存储中的 snake_case 字段
    bubble_states = page_data.get("bubble_states")
    if isinstance(bubble_states, list):
        for index, state in enumerate(bubble_states):
            if not isinstance(state, dict):
                continue
            bubbles.append(
                {
                    "bubble_index": index,
                    "original_text": str(state.get("original", state.get("original_text", "")) or ""),
                    "translated_text": str(state.get("translated", state.get("translated_text", "")) or ""),
                    "ocr_confidence": state.get("ocr_confidence", state.get("confidence", state.get("score"))),
                }
            )
        return bubbles

    # 兜底：使用并行数组
    originals = page_data.get("original_texts") or []
    translated = page_data.get("bubble_texts") or page_data.get("translated_texts") or []
    if isinstance(originals, list):
        for i, text in enumerate(originals):
            bubbles.append(
                {
                    "bubble_index": i,
                    "original_text": str(text or ""),
                    "translated_text": str(translated[i] if i < len(translated) else ""),
                    "ocr_confidence": None,
                }
            )

    return bubbles


def _coerce_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        conf = float(value)
        if conf < 0:
            return None
        if conf > 1 and conf <= 100:
            conf = conf / 100.0
        return conf
    except (TypeError, ValueError):
        return None


def _append_issue(
    issues: List[Dict[str, Any]],
    issue_type: str,
    severity: str,
    page_index: int,
    bubble_index: Optional[int],
    message: str,
    original_text: str = "",
    translated_text: str = "",
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    issues.append(
        {
            "id": f"qi_{len(issues) + 1:04d}",
            "type": issue_type,
            "severity": severity,
            "page_index": page_index,
            "bubble_index": bubble_index,
            "message": message,
            "original_text": original_text,
            "translated_text": translated_text,
            "metrics": metrics or {},
        }
    )


def _build_summary(issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, int] = {}
    by_severity: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}

    for issue in issues:
        issue_type = str(issue.get("type", "unknown"))
        severity = str(issue.get("severity", "low"))
        by_type[issue_type] = by_type.get(issue_type, 0) + 1
        if severity in by_severity:
            by_severity[severity] += 1
        else:
            by_severity[severity] = 1

    return {
        "total": len(issues),
        "by_type": by_type,
        "by_severity": by_severity,
    }


def _collect_term_rules(session_path: str) -> List[Tuple[str, str]]:
    book_id = _extract_book_id(session_path)
    try:
        entries = list_entries(book_id=book_id, include_global=True, entry_type="term")
    except Exception:
        entries = []

    rules: List[Tuple[str, str]] = []
    for item in entries:
        source_text = str(item.get("source_text", "")).strip()
        target_text = str(item.get("target_text", "")).strip()
        enabled = bool(item.get("enabled", True))
        if not enabled:
            continue
        if source_text and target_text:
            rules.append((source_text, target_text))
    return rules


def analyze_session_quality(session_path: str) -> Dict[str, Any]:
    """
    对会话执行质量分析并返回报告。
    """
    ok, normalized_or_error = validate_relative_path(session_path, allow_unicode=False)
    if not ok:
        raise ValueError(f"无效的 session_path: {normalized_or_error}")

    session_path = normalized_or_error
    session_data = load_session(session_path)
    if not session_data:
        raise FileNotFoundError("会话不存在或尚未保存")

    pages = session_data.get("pages", [])
    if not isinstance(pages, list):
        pages = []

    term_rules = _collect_term_rules(session_path)
    issues: List[Dict[str, Any]] = []

    for page in pages:
        if not isinstance(page, dict):
            continue
        raw_page_index = page.get("index", len(issues))
        try:
            page_index = int(raw_page_index)
        except (TypeError, ValueError):
            page_index = len(issues)
        bubbles = _extract_bubbles(page)
        for bubble in bubbles:
            try:
                bubble_index = int(bubble.get("bubble_index", 0))
            except (TypeError, ValueError):
                bubble_index = 0
            original_text = str(bubble.get("original_text", "") or "")
            translated_text = str(bubble.get("translated_text", "") or "")

            o_len = _safe_len(original_text)
            t_len = _safe_len(translated_text)

            # OCR 置信度
            conf = _coerce_confidence(bubble.get("ocr_confidence"))
            if conf is not None and conf < 0.60:
                severity = "high" if conf < 0.4 else "medium"
                _append_issue(
                    issues,
                    issue_type="ocr_low_confidence",
                    severity=severity,
                    page_index=page_index,
                    bubble_index=bubble_index,
                    message=f"OCR 置信度偏低: {conf:.2f}",
                    original_text=original_text,
                    translated_text=translated_text,
                    metrics={"confidence": round(conf, 3)},
                )

            # 缺失译文/长度异常
            if o_len > 0 and t_len == 0:
                _append_issue(
                    issues,
                    issue_type="missing_translation",
                    severity="high",
                    page_index=page_index,
                    bubble_index=bubble_index,
                    message="检测到原文存在但译文为空",
                    original_text=original_text,
                    translated_text=translated_text,
                )
            elif o_len >= 4 and t_len > 0:
                ratio = t_len / max(o_len, 1)
                if ratio > 1.8:
                    _append_issue(
                        issues,
                        issue_type="length_outlier",
                        severity="medium",
                        page_index=page_index,
                        bubble_index=bubble_index,
                        message="译文长度明显偏长，可能超出气泡空间",
                        original_text=original_text,
                        translated_text=translated_text,
                        metrics={"ratio": round(ratio, 3), "origin_len": o_len, "translated_len": t_len},
                    )
                elif ratio < 0.35:
                    _append_issue(
                        issues,
                        issue_type="length_outlier",
                        severity="low",
                        page_index=page_index,
                        bubble_index=bubble_index,
                        message="译文长度明显偏短，可能漏译",
                        original_text=original_text,
                        translated_text=translated_text,
                        metrics={"ratio": round(ratio, 3), "origin_len": o_len, "translated_len": t_len},
                    )

            # 可疑断句
            if translated_text.count("\n") >= 3:
                _append_issue(
                    issues,
                    issue_type="suspicious_line_break",
                    severity="low",
                    page_index=page_index,
                    bubble_index=bubble_index,
                    message="译文存在较多换行，建议检查断句",
                    original_text=original_text,
                    translated_text=translated_text,
                    metrics={"line_breaks": translated_text.count("\n")},
                )

            # 术语一致性
            if original_text and translated_text and term_rules:
                for source_term, target_term in term_rules:
                    if source_term in original_text and target_term not in translated_text:
                        _append_issue(
                            issues,
                            issue_type="term_inconsistency",
                            severity="medium",
                            page_index=page_index,
                            bubble_index=bubble_index,
                            message=f"术语未命中建议译法: {source_term} -> {target_term}",
                            original_text=original_text,
                            translated_text=translated_text,
                            metrics={"expected_target": target_term, "source_term": source_term},
                        )

    report = {
        "session": session_path,
        "generated_at": _now_iso(),
        "issue_count": len(issues),
        "summary": _build_summary(issues),
        "issues": issues,
    }

    report_file = _session_report_file(session_path)
    tmp_file = report_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, report_file)

    return report


def get_session_report(session_path: str) -> Optional[Dict[str, Any]]:
    """读取会话最新质量报告。"""
    ok, normalized_or_error = validate_relative_path(session_path, allow_unicode=False)
    if not ok:
        raise ValueError(f"无效的 session_path: {normalized_or_error}")
    session_path = normalized_or_error

    report_file = _session_report_file(session_path)
    if not os.path.exists(report_file):
        return None

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception as e:
        logger.error(f"读取质量报告失败: {e}", exc_info=True)
        return None
