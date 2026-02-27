"""
安全校验工具

集中提供路径与标识符校验，避免各模块重复实现并出现遗漏。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple


_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Windows 非法文件名字符 + 控制字符
_INVALID_SEGMENT_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


def normalize_rel_path(path: str) -> str:
    """规范化相对路径分隔符（不做安全承诺）。"""
    return str(path).replace("\\", "/").strip("/")


def validate_safe_id(value: str) -> bool:
    """验证资源 ID（book_id/chapter_id 等）。"""
    if not isinstance(value, str):
        return False
    return _SAFE_ID_PATTERN.fullmatch(value) is not None


def validate_relative_path(
    path: str,
    *,
    allow_unicode: bool = True,
    allow_nested: bool = True,
) -> Tuple[bool, str]:
    """
    校验相对路径。

    规则：
    - 禁止空路径、绝对路径、盘符路径
    - 禁止 `.` / `..` / 空段
    - 禁止控制字符和 Windows 非法字符
    """
    if not isinstance(path, str):
        return False, "路径必须是字符串"

    raw = path.strip()
    if not raw:
        return False, "路径不能为空"

    # 绝对路径与盘符
    if raw.startswith(("/", "\\")) or ":" in raw:
        return False, "不允许绝对路径或盘符路径"

    normalized = normalize_rel_path(raw)
    if not normalized:
        return False, "路径不能为空"

    segments = normalized.split("/")
    if not allow_nested and len(segments) > 1:
        return False, "不允许多级路径"

    for segment in segments:
        if not segment or segment in {".", ".."}:
            return False, "路径包含非法片段"
        if _INVALID_SEGMENT_CHARS.search(segment):
            return False, "路径包含非法字符"
        if not allow_unicode and not re.fullmatch(r"[A-Za-z0-9._-]+", segment):
            return False, "路径仅允许字母数字下划线和连字符"

    return True, normalized


def resolve_under_base(base_dir: str, rel_path: str) -> Tuple[bool, str]:
    """
    将相对路径解析到基目录下，并保证不会越界。

    Returns:
        (True, 绝对路径) / (False, 错误信息)
    """
    ok, normalized_or_error = validate_relative_path(rel_path)
    if not ok:
        return False, normalized_or_error

    normalized = normalized_or_error
    base = Path(base_dir).resolve()
    candidate = (base / normalized).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return False, "路径超出允许目录"
    return True, str(candidate)

