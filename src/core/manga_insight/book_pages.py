"""
Manga Insight 页面清单构建器

统一构建全书页面列表，作为页码映射的唯一真值来源。
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

logger = logging.getLogger("MangaInsight.BookPages")


def _natural_sort_parts(value: str) -> List[Any]:
    """将字符串拆分为自然排序所需的片段。"""
    normalized = value.replace("\\", "/")
    parts = re.split(r"(\d+)", normalized)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _find_image_file(session_dir: str, image_index: int, image_type: str = "original") -> str | None:
    """查找图片文件路径，兼容新旧存储格式。"""
    new_format_path = os.path.join(session_dir, "images", str(image_index), f"{image_type}.png")
    if os.path.exists(new_format_path):
        return new_format_path

    for ext in ("png", "jpg", "jpeg", "webp"):
        old_format_path = os.path.join(session_dir, f"image_{image_index}_{image_type}.{ext}")
        if os.path.exists(old_format_path):
            return old_format_path

    return None


def _load_page_meta(session_dir: str, image_index: int) -> Dict[str, Any]:
    """加载新格式页面元数据，缺失时返回空字典。"""
    meta_path = os.path.join(session_dir, "images", str(image_index), "meta.json")
    if not os.path.exists(meta_path):
        return {}

    try:
        with open(meta_path, "r", encoding="utf-8") as meta_file:
            data = json.load(meta_file)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("读取页面元数据失败: %s - %s", meta_path, exc)
        return {}


def _parse_total_pages(raw_total_pages: Any, chapter_id: str, session_meta_path: str) -> int:
    """解析 total_pages，非法值时返回 0 并记录警告。"""
    try:
        parsed = int(raw_total_pages or 0)
    except (TypeError, ValueError):
        logger.warning(
            "章节 %s 的 total_pages 非法(%r): %s，已跳过该章节",
            chapter_id,
            raw_total_pages,
            session_meta_path
        )
        return 0
    if parsed < 0:
        logger.warning(
            "章节 %s 的 total_pages 为负数(%r): %s，已按 0 处理",
            chapter_id,
            raw_total_pages,
            session_meta_path
        )
        return 0
    return parsed


def build_book_pages_manifest(book_id: str) -> Dict[str, Any]:
    """
    构建书籍页面清单。

    Returns:
        Dict: {
            "book_id": str,
            "title": str,
            "cover": str,
            "chapters": List[Dict],
            "total_pages": int,
            "all_images": List[Dict]  # 1-based 页码顺序
        }
    """
    try:
        from src.core import bookshelf_manager
        from src.shared.path_helpers import resource_path

        book = bookshelf_manager.get_book(book_id)
        if not book:
            return {
                "book_id": book_id,
                "title": "",
                "cover": "",
                "chapters": [],
                "total_pages": 0,
                "all_images": []
            }

        chapters = book.get("chapters", [])
        all_images: List[Dict[str, Any]] = []

        for chapter_order, chapter in enumerate(chapters):
            chapter_id = chapter.get("id") or chapter.get("chapter_id")
            if not chapter_id:
                continue

            session_dir = resource_path(f"data/bookshelf/{book_id}/chapters/{chapter_id}/session")
            session_meta_path = os.path.join(session_dir, "session_meta.json")
            if not os.path.exists(session_meta_path):
                continue

            try:
                with open(session_meta_path, "r", encoding="utf-8") as session_file:
                    session_data = json.load(session_file)
            except Exception as exc:
                logger.warning("读取 session_meta 失败: %s - %s", session_meta_path, exc)
                continue

            try:
                if "total_pages" in session_data:
                    # 新格式: images/{idx}/original.png
                    image_count = _parse_total_pages(
                        session_data.get("total_pages", 0),
                        chapter_id,
                        session_meta_path
                    )
                    for index in range(image_count):
                        image_path = _find_image_file(session_dir, index, "original")
                        if not image_path:
                            continue

                        page_meta = _load_page_meta(session_dir, index)
                        all_images.append({
                            "chapter_id": chapter_id,
                            "index": index,
                            "path": image_path,
                            "meta": page_meta,
                            "chapter_order": chapter_order
                        })
                else:
                    # 旧格式: image_{idx}_original.xxx + images_meta
                    images_meta = session_data.get("images_meta", [])
                    if not isinstance(images_meta, list):
                        logger.warning(
                            "章节 %s 的 images_meta 不是列表: %s",
                            chapter_id,
                            session_meta_path
                        )
                        continue

                    for index, legacy_meta in enumerate(images_meta):
                        image_path = _find_image_file(session_dir, index, "original")
                        if not image_path:
                            continue

                        page_meta = legacy_meta if isinstance(legacy_meta, dict) else {}
                        all_images.append({
                            "chapter_id": chapter_id,
                            "index": index,
                            "path": image_path,
                            "meta": page_meta,
                            "chapter_order": chapter_order
                        })
            except Exception as exc:
                logger.warning("解析章节页面清单失败: %s - %s", chapter_id, exc)
                continue

        def sort_key(item: Dict[str, Any]) -> tuple:
            meta = item.get("meta", {})
            sortable = meta.get("relativePath") or meta.get("fileName") or os.path.basename(item.get("path", ""))
            return (item.get("chapter_order", 0), _natural_sort_parts(sortable), item.get("index", 0))

        all_images.sort(key=sort_key)

        for page_num, image_info in enumerate(all_images, start=1):
            image_info["page_num"] = page_num
            image_info.pop("chapter_order", None)

        return {
            "book_id": book_id,
            "title": book.get("title", ""),
            "cover": book.get("cover", ""),
            "chapters": chapters,
            "total_pages": len(all_images),
            "all_images": all_images
        }
    except Exception as exc:
        logger.error("构建页面清单失败: %s", exc, exc_info=True)
        return {
            "book_id": book_id,
            "title": "",
            "cover": "",
            "chapters": [],
            "total_pages": 0,
            "all_images": []
        }
