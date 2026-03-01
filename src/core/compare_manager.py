"""
译文版本对比管理器

提供:
- 对比运行
- 结果持久化读取
"""

import difflib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.shared.path_helpers import resource_path
from src.shared.security import validate_safe_id

logger = logging.getLogger("CompareManager")

_RUN_DIR = resource_path(os.path.join("data", "compare", "runs"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ensure_dir() -> None:
    os.makedirs(_RUN_DIR, exist_ok=True)


def _build_segments(base_text: str, candidate_text: str) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    matcher = difflib.SequenceMatcher(a=base_text, b=candidate_text)
    segments: List[Dict[str, str]] = []
    counts = {"equal": 0, "replace": 0, "insert": 0, "delete": 0}

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        counts[opcode] = counts.get(opcode, 0) + 1
        if opcode == "equal":
            continue
        segments.append(
            {
                "op": opcode,
                "base": base_text[i1:i2],
                "candidate": candidate_text[j1:j2],
            }
        )

    return segments, counts


def _normalize_variant(raw: Dict[str, Any], fallback_name: str, fallback_id: str) -> Dict[str, str]:
    text = str(raw.get("text", "") or "")
    return {
        "id": str(raw.get("id", fallback_id) or fallback_id),
        "name": str(raw.get("name", fallback_name) or fallback_name),
        "text": text,
    }


def _parse_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    baseline_raw = payload.get("baseline")
    candidates_raw = payload.get("candidates")

    # 兼容简化参数
    if not baseline_raw:
        baseline_text = payload.get("baseline_text")
        if baseline_text is not None:
            baseline_raw = {"id": "baseline", "name": "基线", "text": baseline_text}

    if not candidates_raw:
        candidate_text = payload.get("candidate_text")
        if candidate_text is not None:
            candidates_raw = [{"id": "candidate_1", "name": "候选译文", "text": candidate_text}]

    if not baseline_raw:
        variants = payload.get("variants")
        if isinstance(variants, list) and len(variants) >= 2:
            baseline_raw = variants[0]
            candidates_raw = variants[1:]

    if not isinstance(baseline_raw, dict):
        raise ValueError("baseline 缺失或格式错误")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("candidates 至少需要一个候选版本")

    baseline = _normalize_variant(baseline_raw, "基线", "baseline")
    if not baseline["text"].strip():
        raise ValueError("baseline.text 不能为空")

    candidates: List[Dict[str, str]] = []
    for i, item in enumerate(candidates_raw):
        if not isinstance(item, dict):
            continue
        candidate = _normalize_variant(item, f"候选 {i + 1}", f"candidate_{i + 1}")
        if not candidate["text"].strip():
            continue
        candidates.append(candidate)

    if not candidates:
        raise ValueError("candidates 没有可用文本")

    return {
        "session": str(payload.get("session", "") or ""),
        "page_index": payload.get("page_index"),
        "baseline": baseline,
        "candidates": candidates,
    }


def run_compare(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行对比并返回结果，自动持久化。
    """
    parsed = _parse_payload(payload)
    baseline = parsed["baseline"]
    candidates = parsed["candidates"]

    results: List[Dict[str, Any]] = []
    for item in candidates:
        segments, counts = _build_segments(baseline["text"], item["text"])
        ratio = difflib.SequenceMatcher(a=baseline["text"], b=item["text"]).ratio()
        results.append(
            {
                "id": item["id"],
                "name": item["name"],
                "text": item["text"],
                "similarity": round(ratio, 4),
                "diff_segments": segments,
                "diff_summary": counts,
            }
        )

    run = {
        "id": f"cmp_{uuid.uuid4().hex[:10]}",
        "status": "completed",
        "created_at": _now_iso(),
        "session": parsed["session"],
        "page_index": parsed["page_index"],
        "baseline": baseline,
        "candidates": results,
    }

    _ensure_dir()
    run_file = os.path.join(_RUN_DIR, f"{run['id']}.json")
    tmp_file = run_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, run_file)
    return run


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """按 ID 查询对比运行。"""
    if not validate_safe_id(run_id):
        raise ValueError("无效的 compare run id")

    _ensure_dir()
    run_file = os.path.join(_RUN_DIR, f"{run_id}.json")
    if not os.path.exists(run_file):
        return None

    try:
        with open(run_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.error(f"读取对比结果失败: {e}", exc_info=True)
        return None

