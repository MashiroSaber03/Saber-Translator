# src/core/data_migration.py
"""
数据迁移模块 - 统一三个系统（书架、翻译、Insight）的数据存储结构

迁移目标：
- 旧：data/sessions/bookshelf/{book_id}/{chapter_id}/ → 新：data/bookshelf/{book_id}/chapters/{chapter_id}/session/
- 旧：data/manga_insight/{book_id}/ → 新：data/bookshelf/{book_id}/insight/
"""

import os
import shutil
import logging
from typing import Dict, List, Optional, Any

from src.shared.path_helpers import resource_path

logger = logging.getLogger("DataMigration")

# 旧路径常量
OLD_SESSIONS_BOOKSHELF_DIR = "data/sessions/bookshelf"
OLD_MANGA_INSIGHT_DIR = "data/manga_insight"

# 新路径常量
BOOKSHELF_DIR = "data/bookshelf"


def _get_old_session_path(book_id: str, chapter_id: str) -> str:
    """获取旧的会话路径"""
    return resource_path(os.path.join(OLD_SESSIONS_BOOKSHELF_DIR, book_id, chapter_id))


def _get_new_session_path(book_id: str, chapter_id: str) -> str:
    """获取新的会话路径"""
    return resource_path(os.path.join(BOOKSHELF_DIR, book_id, "chapters", chapter_id, "session"))


def _get_old_insight_path(book_id: str) -> str:
    """获取旧的 Insight 路径"""
    return resource_path(os.path.join(OLD_MANGA_INSIGHT_DIR, book_id))


def _get_new_insight_path(book_id: str) -> str:
    """获取新的 Insight 路径"""
    return resource_path(os.path.join(BOOKSHELF_DIR, book_id, "insight"))


def migrate_session_data(book_id: str, chapter_id: str) -> bool:
    """
    迁移单个章节的会话数据到新位置

    Args:
        book_id: 书籍ID
        chapter_id: 章节ID

    Returns:
        是否迁移成功
    """
    old_path = _get_old_session_path(book_id, chapter_id)
    new_path = _get_new_session_path(book_id, chapter_id)

    # 如果旧路径不存在，无需迁移
    if not os.path.exists(old_path):
        return True

    # 如果新路径已存在且有数据，跳过
    if os.path.exists(new_path) and os.listdir(new_path):
        logger.info(f"会话数据已在新位置，跳过: {book_id}/{chapter_id}")
        return True

    try:
        # 确保目标目录的父目录存在
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        # 移动目录
        shutil.move(old_path, new_path)
        logger.info(f"迁移会话数据成功: {old_path} -> {new_path}")

        # 清理空的父目录
        _cleanup_empty_dirs(os.path.dirname(old_path))

        return True
    except Exception as e:
        logger.error(f"迁移会话数据失败 {book_id}/{chapter_id}: {e}")
        return False


def migrate_insight_data(book_id: str) -> bool:
    """
    迁移 Insight 数据到新位置

    Args:
        book_id: 书籍ID

    Returns:
        是否迁移成功
    """
    old_path = _get_old_insight_path(book_id)
    new_path = _get_new_insight_path(book_id)

    # 如果旧路径不存在，无需迁移
    if not os.path.exists(old_path):
        return True

    # 如果新路径已存在且有数据，跳过
    if os.path.exists(new_path) and os.listdir(new_path):
        logger.info(f"Insight 数据已在新位置，跳过: {book_id}")
        return True

    try:
        # 确保目标目录的父目录存在
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        # 移动目录
        shutil.move(old_path, new_path)
        logger.info(f"迁移 Insight 数据成功: {old_path} -> {new_path}")

        return True
    except Exception as e:
        logger.error(f"迁移 Insight 数据失败 {book_id}: {e}")
        return False


def _cleanup_empty_dirs(path: str) -> None:
    """递归清理空目录"""
    try:
        while path and os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
            logger.debug(f"清理空目录: {path}")
            path = os.path.dirname(path)
    except Exception as e:
        logger.debug(f"清理空目录时出错（非致命）: {e}")


def get_books_needing_migration() -> Dict[str, Any]:
    """
    检查需要迁移的书籍

    Returns:
        包含需要迁移信息的字典
    """
    result = {
        "sessions_to_migrate": [],  # [(book_id, chapter_id), ...]
        "insights_to_migrate": [],  # [book_id, ...]
        "needs_migration": False
    }

    # 检查旧的会话目录
    old_sessions_base = resource_path(OLD_SESSIONS_BOOKSHELF_DIR)
    if os.path.exists(old_sessions_base):
        for book_id in os.listdir(old_sessions_base):
            book_path = os.path.join(old_sessions_base, book_id)
            if os.path.isdir(book_path):
                for chapter_id in os.listdir(book_path):
                    chapter_path = os.path.join(book_path, chapter_id)
                    if os.path.isdir(chapter_path):
                        # 检查新位置是否已有数据
                        new_path = _get_new_session_path(book_id, chapter_id)
                        if not os.path.exists(new_path) or not os.listdir(new_path):
                            result["sessions_to_migrate"].append((book_id, chapter_id))

    # 检查旧的 Insight 目录
    old_insight_base = resource_path(OLD_MANGA_INSIGHT_DIR)
    if os.path.exists(old_insight_base):
        for book_id in os.listdir(old_insight_base):
            insight_path = os.path.join(old_insight_base, book_id)
            if os.path.isdir(insight_path):
                # 检查新位置是否已有数据
                new_path = _get_new_insight_path(book_id)
                if not os.path.exists(new_path) or not os.listdir(new_path):
                    result["insights_to_migrate"].append(book_id)

    result["needs_migration"] = bool(result["sessions_to_migrate"]) or bool(result["insights_to_migrate"])
    return result


def check_and_migrate() -> Dict[str, Any]:
    """
    检查并自动迁移所有旧数据

    Returns:
        迁移结果统计
    """
    migration_info = get_books_needing_migration()

    if not migration_info["needs_migration"]:
        logger.info("无需数据迁移，所有数据已是新格式")
        return {
            "migrated": False,
            "message": "无需迁移",
            "sessions_migrated": 0,
            "insights_migrated": 0
        }

    logger.info("检测到旧格式数据，开始自动迁移...")

    sessions_success = 0
    sessions_failed = 0
    insights_success = 0
    insights_failed = 0

    # 迁移会话数据
    for book_id, chapter_id in migration_info["sessions_to_migrate"]:
        if migrate_session_data(book_id, chapter_id):
            sessions_success += 1
        else:
            sessions_failed += 1

    # 迁移 Insight 数据
    for book_id in migration_info["insights_to_migrate"]:
        if migrate_insight_data(book_id):
            insights_success += 1
        else:
            insights_failed += 1

    total_success = sessions_success + insights_success
    total_failed = sessions_failed + insights_failed

    logger.info(f"数据迁移完成: 成功 {total_success}, 失败 {total_failed}")

    return {
        "migrated": True,
        "message": f"迁移完成: 会话 {sessions_success}/{sessions_success + sessions_failed}, Insight {insights_success}/{insights_success + insights_failed}",
        "sessions_migrated": sessions_success,
        "sessions_failed": sessions_failed,
        "insights_migrated": insights_success,
        "insights_failed": insights_failed
    }


def get_session_path_for_chapter(book_id: str, chapter_id: str) -> str:
    """
    获取章节会话的存储路径（相对于 data 目录）

    新路径格式: bookshelf/{book_id}/chapters/{chapter_id}/session

    Args:
        book_id: 书籍ID
        chapter_id: 章节ID

    Returns:
        相对路径字符串
    """
    return os.path.join("bookshelf", book_id, "chapters", chapter_id, "session")


def get_insight_path_for_book(book_id: str) -> str:
    """
    获取书籍 Insight 数据的存储路径（绝对路径）

    新路径格式: data/bookshelf/{book_id}/insight

    Args:
        book_id: 书籍ID

    Returns:
        绝对路径字符串
    """
    return resource_path(os.path.join(BOOKSHELF_DIR, book_id, "insight"))
