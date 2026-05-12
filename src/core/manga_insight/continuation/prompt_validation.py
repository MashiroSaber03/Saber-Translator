"""
续写生图提示词与页面描述的轻量校验/清洗工具。
"""

from __future__ import annotations

import re

INVALID_IMAGE_PROMPT_PREFIXES = (
    "生成失败:",
    "提示词生成失败:",
)

CAMERA_LANGUAGE_KEYWORDS = (
    "俯视",
    "仰视",
    "平视",
    "特写",
    "远景",
    "中景",
    "近景",
    "镜头",
    "分格",
    "景深",
    "运镜",
    "视角",
    "构图",
    "焦距",
)

CAMERA_LANGUAGE_PATTERNS = (
    r"(?:建议)?(?:使用|采用)?(?:俯视|仰视|平视)(?:角度|视角)?(?:下)?",
    r"(?:建议)?(?:使用|采用)?(?:远景|中景|近景)(?:镜头|画面)?",
    r"(?:建议)?(?:使用|采用)?特写(?:镜头|画面|表情)?",
    r"(?:建议)?(?:使用|采用)?(?:镜头语言|镜头感|镜头)",
    r"(?:建议)?(?:使用|采用)?分格\d*",
    r"(?:建议)?(?:使用|采用)?景深(?:虚化)?",
    r"(?:建议)?(?:使用|采用)?运镜",
    r"(?:建议)?(?:使用|采用)?构图",
    r"(?:建议)?(?:使用|采用)?焦距",
)


def normalize_page_description(description: str) -> str:
    """
    清理页面描述中的导演/镜头语言，只保留事件、情绪、位置等核心信息。
    """
    text = str(description or "").strip()
    if not text:
        return ""

    normalized_text = re.sub(r"[\r\n]+", "，", text)
    clauses = [
        clause.strip(" ，。；、")
        for clause in re.split(r"[，。；\n]+", normalized_text)
        if clause.strip(" ，。；、")
    ]

    filtered = []
    for clause in clauses:
        cleaned_clause = clause
        for pattern in CAMERA_LANGUAGE_PATTERNS:
            cleaned_clause = re.sub(pattern, "", cleaned_clause)

        cleaned_clause = re.sub(r"^(?:以|用|并用|并采用|画面以|画面采用)", "", cleaned_clause)
        cleaned_clause = re.sub(r"\s+", " ", cleaned_clause).strip(" ，。；、:：")

        if not cleaned_clause:
            continue

        if any(keyword in cleaned_clause for keyword in CAMERA_LANGUAGE_KEYWORDS):
            continue

        filtered.append(cleaned_clause)

    result = "，".join(filtered) if filtered else text
    result = re.sub(r"\s+", " ", result).strip(" ，。；、")
    return result or text


def normalize_image_prompt_text(prompt: str, max_lines: int = 5) -> str:
    """
    规范化生图提示词，移除空行并限制为前 N 行，保持简洁结构。
    """
    lines = [
        line.strip()
        for line in str(prompt or "").splitlines()
        if line.strip()
    ]
    if not lines:
        return ""

    return "\n".join(lines[:max_lines])


def is_usable_image_prompt(prompt: str) -> bool:
    """
    判断当前提示词是否可用于生图。
    """
    text = normalize_image_prompt_text(prompt)
    if not text:
        return False

    return not any(text.startswith(prefix) for prefix in INVALID_IMAGE_PROMPT_PREFIXES)
