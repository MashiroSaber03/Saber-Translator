"""
续写功能参考图 token 与候选项工具。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

from .models import ContinuationCharacters

ORIGINAL_REFERENCE_KIND = "original"
CONTINUATION_REFERENCE_KIND = "continuation"
CHARACTER_REFERENCE_KIND = "character"


def build_original_reference_token(page_number: int) -> str:
    return f"{ORIGINAL_REFERENCE_KIND}:{int(page_number)}"


def build_continuation_reference_token(page_number: int) -> str:
    return f"{CONTINUATION_REFERENCE_KIND}:{int(page_number)}"


def build_character_reference_token(character_name: str, form_id: str) -> str:
    return f"{CHARACTER_REFERENCE_KIND}:{quote(character_name or '', safe='')}:{quote(form_id or '', safe='')}"


def list_original_manga_page_paths(book_id: str) -> List[str]:
    """列出指定书籍的全部原作页路径。"""
    import json
    from src.shared.path_helpers import resource_path
    from src.core import bookshelf_manager

    pages: List[str] = []

    try:
        book = bookshelf_manager.get_book(book_id)
        if not book:
            return pages

        chapters = book.get("chapters", [])
        for chapter in chapters:
            chapter_id = chapter.get("id")
            if not chapter_id:
                continue

            session_dir = resource_path(f"data/bookshelf/{book_id}/chapters/{chapter_id}/session")
            session_meta_path = os.path.join(session_dir, "session_meta.json")
            if not os.path.exists(session_meta_path):
                continue

            try:
                with open(session_meta_path, "r", encoding="utf-8") as handle:
                    session_data = json.load(handle)

                if "total_pages" in session_data:
                    image_count = int(session_data.get("total_pages", 0) or 0)
                else:
                    images_meta = session_data.get("images_meta", [])
                    image_count = len(images_meta)

                for index in range(image_count):
                    image_path = os.path.join(session_dir, "images", str(index), "original.png")
                    if os.path.exists(image_path):
                        pages.append(image_path)
            except Exception:
                continue
    except Exception:
        return []

    return pages


def build_original_reference_candidates(image_paths: Iterable[str]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        normalized_path = os.path.normpath(image_path) if image_path else ""
        has_image = bool(normalized_path and os.path.exists(normalized_path))
        candidates.append({
            "token": build_original_reference_token(index),
            "kind": ORIGINAL_REFERENCE_KIND,
            "page_number": index,
            "path": normalized_path if has_image else "",
            "has_image": has_image,
            "is_placeholder": not has_image,
            "label": f"第{index}页",
        })
    return candidates


def build_continuation_reference_candidates(
    total_original_pages: int,
    pages: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    total_original_pages = max(0, int(total_original_pages or 0))

    for index, page in enumerate(pages, start=1):
        continuation_page_number = int(page.get("page_number") or index)
        actual_page_number = total_original_pages + continuation_page_number
        raw_image_url = str(page.get("image_url") or "").strip()
        image_url = os.path.normpath(raw_image_url) if raw_image_url else ""
        has_image = bool(image_url and os.path.exists(image_url))
        candidates.append({
            "token": build_continuation_reference_token(continuation_page_number),
            "kind": CONTINUATION_REFERENCE_KIND,
            "page_number": actual_page_number,
            "continuation_page_number": continuation_page_number,
            "path": image_url if has_image else "",
            "has_image": has_image,
            "is_placeholder": not has_image,
            "label": f"第{actual_page_number}页",
            "status": page.get("status", "pending"),
        })

    return candidates


def build_character_reference_candidates(characters: ContinuationCharacters) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for character in characters.characters:
        if character.enabled is False:
            continue

        for form in character.forms:
            if form.enabled is False:
                continue

            image_path = os.path.normpath(form.reference_image) if form.reference_image else ""
            has_image = bool(image_path and os.path.exists(image_path))
            candidates.append({
                "token": build_character_reference_token(character.name, form.form_id),
                "kind": CHARACTER_REFERENCE_KIND,
                "page_number": 0,
                "path": image_path if has_image else "",
                "has_image": has_image,
                "is_placeholder": not has_image,
                "character_name": character.name,
                "form_id": form.form_id,
                "form_name": form.form_name,
                "label": f"{character.name} - {form.form_name}",
            })

    return candidates


def select_recent_style_reference_tokens(
    candidates: Iterable[Dict[str, Any]],
    count: int,
    current_page_number: Optional[int] = None,
) -> List[str]:
    count = max(1, int(count or 1))
    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("kind") in {ORIGINAL_REFERENCE_KIND, CONTINUATION_REFERENCE_KIND}
        and candidate.get("token")
        and candidate.get("has_image")
        and candidate.get("path")
        and not (
            current_page_number is not None
            and candidate.get("kind") == CONTINUATION_REFERENCE_KIND
            and isinstance(candidate.get("page_number"), int)
            and int(candidate.get("page_number")) >= int(current_page_number)
        )
    ]
    eligible.sort(key=lambda item: (int(item.get("page_number") or 0), str(item.get("token") or "")))
    return [candidate["token"] for candidate in eligible[-count:]]


def resolve_reference_tokens(
    tokens: Iterable[str],
    candidates: Iterable[Dict[str, Any]],
    current_page_number: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    将参考图 token 解析为 ImageGenClient 可直接消费的参考图描述。

    - 只返回已存在图片。
    - continuation 候选会在当前页及之后被过滤掉，确保滑窗只使用“之前页”的内容。
    """
    candidate_map = {
        candidate.get("token"): candidate
        for candidate in candidates
        if candidate.get("token")
    }

    resolved: List[Dict[str, Any]] = []
    current_page_number = int(current_page_number) if current_page_number is not None else None

    for token in tokens or []:
        candidate = candidate_map.get(token)
        if not candidate:
            continue

        if not candidate.get("has_image") or not candidate.get("path"):
            continue

        page_number = candidate.get("page_number")
        if (
            candidate.get("kind") == CONTINUATION_REFERENCE_KIND
            and current_page_number is not None
            and isinstance(page_number, int)
            and page_number >= current_page_number
        ):
            continue

        reference: Dict[str, Any] = {
            "path": candidate["path"],
            "type": "character" if candidate.get("kind") == CHARACTER_REFERENCE_KIND else "style",
            "token": candidate["token"],
        }

        if candidate.get("kind") == CHARACTER_REFERENCE_KIND:
            label = candidate.get("label") or candidate.get("character_name") or ""
            if label:
                reference["name"] = label

        resolved.append(reference)

    return resolved
