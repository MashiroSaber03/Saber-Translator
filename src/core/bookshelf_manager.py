# src/core/bookshelf_manager.py
"""
书架管理模块 - 管理书籍和章节的数据存储
支持创建/编辑/删除书籍，以及书籍内的章节管理
"""

import os
import json
import logging
import time
import shutil
import uuid
from typing import Optional, List, Dict, Any
from src.shared.path_helpers import resource_path

logger = logging.getLogger("BookshelfManager")

# --- 基础配置 ---
BOOKSHELF_DIR_NAME = "bookshelf"  # 书架数据目录
BOOKS_METADATA_FILE = "books.json"  # 书籍列表元数据
BOOK_METADATA_FILE = "book_meta.json"  # 单本书的元数据
CHAPTER_METADATA_FILE = "chapter_meta.json"  # 章节元数据
COVER_FILENAME = "cover"  # 封面文件名（无扩展名）
IMAGE_DATA_EXTENSION = ".b64"  # Base64 图像数据文件扩展名
SESSIONS_DIR_NAME = "sessions"  # 会话数据目录
TAGS_METADATA_FILE = "tags.json"  # 标签元数据文件


def _get_bookshelf_dir() -> str:
    """获取书架数据目录的绝对路径"""
    base_path = resource_path(os.path.join("data", BOOKSHELF_DIR_NAME))
    os.makedirs(base_path, exist_ok=True)
    return base_path


def _get_session_dir_for_book(book_id: str) -> str:
    """获取书籍对应的会话目录路径（用于删除时清理）"""
    return resource_path(os.path.join("data", SESSIONS_DIR_NAME, "bookshelf", book_id))


def _get_session_dir_for_chapter(book_id: str, chapter_id: str) -> str:
    """获取章节对应的会话目录路径（用于删除时清理）"""
    return resource_path(os.path.join("data", SESSIONS_DIR_NAME, "bookshelf", book_id, chapter_id))


def _get_books_metadata_path() -> str:
    """获取书籍列表元数据文件路径"""
    return os.path.join(_get_bookshelf_dir(), BOOKS_METADATA_FILE)


def _get_book_dir(book_id: str) -> str:
    """获取指定书籍的目录路径"""
    return os.path.join(_get_bookshelf_dir(), book_id)


def _get_chapter_dir(book_id: str, chapter_id: str) -> str:
    """获取指定章节的目录路径"""
    return os.path.join(_get_book_dir(book_id), "chapters", chapter_id)


def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())[:8]


def _load_books_metadata() -> Dict[str, Any]:
    """加载书籍列表元数据"""
    metadata_path = _get_books_metadata_path()
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"加载书籍元数据失败: {e}")
    return {"books": []}


def _get_tags_metadata_path() -> str:
    """获取标签元数据文件路径"""
    return os.path.join(_get_bookshelf_dir(), TAGS_METADATA_FILE)


def _load_tags_metadata() -> Dict[str, Any]:
    """加载标签元数据"""
    tags_path = _get_tags_metadata_path()
    if os.path.exists(tags_path):
        try:
            with open(tags_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"加载标签元数据失败: {e}")
    return {"tags": []}


def _save_tags_metadata(metadata: Dict[str, Any]) -> bool:
    """保存标签元数据"""
    try:
        tags_path = _get_tags_metadata_path()
        with open(tags_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return True
    except IOError as e:
        logger.error(f"保存标签元数据失败: {e}")
        return False


def _save_books_metadata(metadata: Dict[str, Any]) -> bool:
    """保存书籍列表元数据"""
    try:
        metadata_path = _get_books_metadata_path()
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return True
    except IOError as e:
        logger.error(f"保存书籍元数据失败: {e}")
        return False


def _load_book_metadata(book_id: str) -> Optional[Dict[str, Any]]:
    """加载单本书的元数据"""
    book_meta_path = os.path.join(_get_book_dir(book_id), BOOK_METADATA_FILE)
    if os.path.exists(book_meta_path):
        try:
            with open(book_meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"加载书籍 {book_id} 元数据失败: {e}")
    return None


def _save_book_metadata(book_id: str, metadata: Dict[str, Any]) -> bool:
    """保存单本书的元数据"""
    try:
        book_dir = _get_book_dir(book_id)
        os.makedirs(book_dir, exist_ok=True)
        book_meta_path = os.path.join(book_dir, BOOK_METADATA_FILE)
        with open(book_meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return True
    except IOError as e:
        logger.error(f"保存书籍 {book_id} 元数据失败: {e}")
        return False


# ==================== 书籍管理 ====================

def get_all_books(search: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    获取所有书籍列表
    
    Args:
        search: 搜索关键词（可选）
        tags: 标签筛选列表（可选）
    
    Returns:
        书籍信息列表，每本书包含: id, title, cover, chapter_count, tags, created_at, updated_at
    """
    metadata = _load_books_metadata()
    books = []
    
    for book_summary in metadata.get("books", []):
        book_id = book_summary.get("id")
        if not book_id:
            continue
            
        # 加载完整的书籍元数据以获取最新信息
        book_meta = _load_book_metadata(book_id)
        if book_meta:
            book_tags = book_meta.get("tags", [])
            title = book_meta.get("title", "未命名")
            
            # 搜索过滤
            if search:
                search_lower = search.lower()
                if search_lower not in title.lower():
                    # 也搜索标签
                    tag_match = any(search_lower in tag.lower() for tag in book_tags)
                    if not tag_match:
                        continue
            
            # 标签过滤
            if tags:
                if not any(tag in book_tags for tag in tags):
                    continue
            
            books.append({
                "id": book_id,
                "title": title,
                "cover": book_meta.get("cover"),
                "chapter_count": len(book_meta.get("chapters", [])),
                "tags": book_tags,
                "created_at": book_meta.get("created_at"),
                "updated_at": book_meta.get("updated_at")
            })
    
    # 按更新时间倒序排列
    books.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return books


def create_book(title: str, cover_data: Optional[str] = None, tags: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    创建新书籍
    
    Args:
        title: 书籍标题
        cover_data: 封面图片的Base64数据（可选）
        tags: 标签列表（可选）
    
    Returns:
        创建的书籍信息，失败返回None
    """
    book_id = _generate_id()
    current_time = int(time.time() * 1000)
    
    book_meta = {
        "id": book_id,
        "title": title,
        "cover": None,
        "tags": tags or [],
        "chapters": [],
        "created_at": current_time,
        "updated_at": current_time
    }
    
    # 创建书籍目录
    book_dir = _get_book_dir(book_id)
    os.makedirs(book_dir, exist_ok=True)
    os.makedirs(os.path.join(book_dir, "chapters"), exist_ok=True)
    
    # 保存封面
    if cover_data:
        cover_saved = _save_cover(book_id, cover_data)
        if cover_saved:
            book_meta["cover"] = f"/api/bookshelf/books/{book_id}/cover"
    
    # 保存书籍元数据
    if not _save_book_metadata(book_id, book_meta):
        return None
    
    # 更新书籍列表
    books_metadata = _load_books_metadata()
    books_metadata["books"].append({"id": book_id})
    _save_books_metadata(books_metadata)
    
    logger.info(f"创建书籍成功: {title} (ID: {book_id})")
    return book_meta


def get_book(book_id: str) -> Optional[Dict[str, Any]]:
    """获取书籍详情"""
    book_meta = _load_book_metadata(book_id)
    if book_meta:
        # 确保封面路径是API路径
        if book_meta.get("cover") and not book_meta["cover"].startswith("/api"):
            book_meta["cover"] = f"/api/bookshelf/books/{book_id}/cover"
        
        # 确保有tags字段
        if "tags" not in book_meta:
            book_meta["tags"] = []
        
        # 确保每个章节都有 session_path 和 page_count
        chapters = book_meta.get("chapters", [])
        total_pages = 0
        sessions_base = resource_path("data/sessions")
        
        for chapter in chapters:
            if "session_path" not in chapter:
                chapter["session_path"] = get_chapter_session_path(book_id, chapter["id"])
            
            # 计算章节页数
            chapter_id = chapter.get("id")
            if chapter_id:
                # 构建绝对路径: data/sessions/bookshelf/{book_id}/{chapter_id}
                session_abs_path = os.path.join(sessions_base, "bookshelf", book_id, chapter_id)
                session_meta_path = os.path.join(session_abs_path, "session_meta.json")
                
                if os.path.exists(session_meta_path):
                    try:
                        with open(session_meta_path, "r", encoding="utf-8") as f:
                            session_data = json.load(f)
                        page_count = len(session_data.get("images_meta", []))
                        chapter["page_count"] = page_count
                        total_pages += page_count
                    except Exception:
                        chapter["page_count"] = 0
                else:
                    chapter["page_count"] = 0
            else:
                chapter["page_count"] = 0
        
        book_meta["total_pages"] = total_pages
    
    return book_meta


def update_book(book_id: str, title: Optional[str] = None, 
                cover_data: Optional[str] = None,
                tags: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    更新书籍信息
    
    Args:
        book_id: 书籍ID
        title: 新标题（可选）
        cover_data: 新封面Base64数据（可选）
        tags: 新标签列表（可选）
    
    Returns:
        更新后的书籍信息
    """
    book_meta = _load_book_metadata(book_id)
    if not book_meta:
        logger.error(f"书籍不存在: {book_id}")
        return None
    
    if title:
        book_meta["title"] = title
    
    if cover_data:
        if _save_cover(book_id, cover_data):
            book_meta["cover"] = f"/api/bookshelf/books/{book_id}/cover"
    
    if tags is not None:
        book_meta["tags"] = tags
    
    book_meta["updated_at"] = int(time.time() * 1000)
    
    if _save_book_metadata(book_id, book_meta):
        logger.info(f"更新书籍成功: {book_id}")
        return book_meta
    return None


def delete_book(book_id: str) -> bool:
    """删除书籍及其所有章节"""
    book_dir = _get_book_dir(book_id)
    
    if not os.path.exists(book_dir):
        logger.warning(f"书籍目录不存在: {book_id}")
        return False
    
    try:
        shutil.rmtree(book_dir)
        
        # 删除对应的会话文件（图片数据）
        session_dir = _get_session_dir_for_book(book_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            logger.info(f"已删除书籍会话目录: {session_dir}")
        
        # 从书籍列表中移除
        books_metadata = _load_books_metadata()
        books_metadata["books"] = [b for b in books_metadata["books"] if b.get("id") != book_id]
        _save_books_metadata(books_metadata)
        
        logger.info(f"删除书籍成功: {book_id}")
        return True
    except Exception as e:
        logger.error(f"删除书籍失败 {book_id}: {e}")
        return False


def _save_cover(book_id: str, cover_data: str) -> bool:
    """保存书籍封面"""
    try:
        book_dir = _get_book_dir(book_id)
        os.makedirs(book_dir, exist_ok=True)
        
        # 移除可能的Data URL前缀
        if "," in cover_data:
            cover_data = cover_data.split(",", 1)[1]
        
        cover_path = os.path.join(book_dir, f"{COVER_FILENAME}{IMAGE_DATA_EXTENSION}")
        with open(cover_path, 'w', encoding='utf-8') as f:
            f.write(cover_data)
        return True
    except Exception as e:
        logger.error(f"保存封面失败: {e}")
        return False


def get_cover(book_id: str) -> Optional[str]:
    """获取书籍封面的Base64数据"""
    cover_path = os.path.join(_get_book_dir(book_id), f"{COVER_FILENAME}{IMAGE_DATA_EXTENSION}")
    if os.path.exists(cover_path):
        try:
            with open(cover_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取封面失败: {e}")
    return None


# ==================== 章节管理 ====================

def get_chapters(book_id: str) -> List[Dict[str, Any]]:
    """获取书籍的所有章节列表"""
    book_meta = _load_book_metadata(book_id)
    if not book_meta:
        return []
    return book_meta.get("chapters", [])


def create_chapter(book_id: str, title: str, order: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    创建新章节
    
    Args:
        book_id: 书籍ID
        title: 章节标题
        order: 排序位置（可选，默认添加到末尾）
    
    Returns:
        创建的章节信息
    """
    book_meta = _load_book_metadata(book_id)
    if not book_meta:
        logger.error(f"书籍不存在: {book_id}")
        return None
    
    chapter_id = _generate_id()
    current_time = int(time.time() * 1000)
    
    chapter_info = {
        "id": chapter_id,
        "title": title,
        "image_count": 0,
        "session_path": get_chapter_session_path(book_id, chapter_id),
        "created_at": current_time,
        "updated_at": current_time
    }
    
    # 创建章节目录
    chapter_dir = _get_chapter_dir(book_id, chapter_id)
    os.makedirs(chapter_dir, exist_ok=True)
    
    # 初始化章节元数据
    chapter_meta = {
        "id": chapter_id,
        "title": title,
        "images": [],
        "created_at": current_time,
        "updated_at": current_time
    }
    _save_chapter_metadata(book_id, chapter_id, chapter_meta)
    
    # 更新书籍的章节列表
    chapters = book_meta.get("chapters", [])
    if order is not None and 0 <= order <= len(chapters):
        chapters.insert(order, chapter_info)
    else:
        chapters.append(chapter_info)
    
    book_meta["chapters"] = chapters
    book_meta["updated_at"] = current_time
    
    if _save_book_metadata(book_id, book_meta):
        logger.info(f"创建章节成功: {title} (书籍: {book_id}, 章节: {chapter_id})")
        return chapter_info
    return None


def get_chapter(book_id: str, chapter_id: str) -> Optional[Dict[str, Any]]:
    """获取章节详情（包括图片列表）"""
    chapter_meta_path = os.path.join(_get_chapter_dir(book_id, chapter_id), CHAPTER_METADATA_FILE)
    if os.path.exists(chapter_meta_path):
        try:
            with open(chapter_meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载章节元数据失败: {e}")
    return None


def update_chapter(book_id: str, chapter_id: str, title: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """更新章节信息"""
    book_meta = _load_book_metadata(book_id)
    if not book_meta:
        return None
    
    # 更新书籍中的章节摘要
    for chapter in book_meta.get("chapters", []):
        if chapter.get("id") == chapter_id:
            if title:
                chapter["title"] = title
            chapter["updated_at"] = int(time.time() * 1000)
            break
    else:
        logger.error(f"章节不存在: {chapter_id}")
        return None
    
    book_meta["updated_at"] = int(time.time() * 1000)
    _save_book_metadata(book_id, book_meta)
    
    # 更新章节自身的元数据
    chapter_meta = get_chapter(book_id, chapter_id)
    if chapter_meta and title:
        chapter_meta["title"] = title
        chapter_meta["updated_at"] = int(time.time() * 1000)
        _save_chapter_metadata(book_id, chapter_id, chapter_meta)
        return chapter_meta
    
    return None


def delete_chapter(book_id: str, chapter_id: str) -> bool:
    """删除章节"""
    book_meta = _load_book_metadata(book_id)
    if not book_meta:
        return False
    
    # 从书籍中移除章节
    chapters = book_meta.get("chapters", [])
    book_meta["chapters"] = [c for c in chapters if c.get("id") != chapter_id]
    book_meta["updated_at"] = int(time.time() * 1000)
    
    # 删除章节目录
    chapter_dir = _get_chapter_dir(book_id, chapter_id)
    if os.path.exists(chapter_dir):
        try:
            shutil.rmtree(chapter_dir)
        except Exception as e:
            logger.error(f"删除章节目录失败: {e}")
            return False
    
    # 删除对应的会话文件（图片数据）
    session_dir = _get_session_dir_for_chapter(book_id, chapter_id)
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir)
            logger.info(f"已删除章节会话目录: {session_dir}")
        except Exception as e:
            logger.warning(f"删除章节会话目录失败（非致命）: {e}")
    
    _save_book_metadata(book_id, book_meta)
    logger.info(f"删除章节成功: {chapter_id} (书籍: {book_id})")
    return True


def reorder_chapters(book_id: str, chapter_ids: List[str]) -> bool:
    """重新排序章节"""
    book_meta = _load_book_metadata(book_id)
    if not book_meta:
        return False
    
    chapters = book_meta.get("chapters", [])
    chapter_map = {c["id"]: c for c in chapters}
    
    # 按新顺序重组章节列表
    new_chapters = []
    for cid in chapter_ids:
        if cid in chapter_map:
            new_chapters.append(chapter_map[cid])
    
    # 添加不在新顺序中的章节到末尾
    for c in chapters:
        if c["id"] not in chapter_ids:
            new_chapters.append(c)
    
    book_meta["chapters"] = new_chapters
    book_meta["updated_at"] = int(time.time() * 1000)
    
    return _save_book_metadata(book_id, book_meta)


def _save_chapter_metadata(book_id: str, chapter_id: str, metadata: Dict[str, Any]) -> bool:
    """保存章节元数据"""
    try:
        chapter_dir = _get_chapter_dir(book_id, chapter_id)
        os.makedirs(chapter_dir, exist_ok=True)
        meta_path = os.path.join(chapter_dir, CHAPTER_METADATA_FILE)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"保存章节元数据失败: {e}")
        return False


def update_chapter_image_count(book_id: str, chapter_id: str, count: int) -> bool:
    """更新章节的图片数量（在翻译页面保存时调用）"""
    book_meta = _load_book_metadata(book_id)
    if not book_meta:
        return False
    
    for chapter in book_meta.get("chapters", []):
        if chapter.get("id") == chapter_id:
            chapter["image_count"] = count
            chapter["updated_at"] = int(time.time() * 1000)
            break
    
    book_meta["updated_at"] = int(time.time() * 1000)
    return _save_book_metadata(book_id, book_meta)


def get_chapter_session_path(book_id: str, chapter_id: str) -> str:
    """
    获取章节对应的会话存储路径（相对于 data/sessions 目录）
    这个路径用于与现有的session_manager集成
    返回格式: bookshelf/{book_id}/{chapter_id}
    """
    return os.path.join("bookshelf", book_id, chapter_id)


# ==================== 标签管理 ====================

def get_all_tags() -> List[Dict[str, Any]]:
    """
    获取所有标签
    
    Returns:
        标签列表，每个标签包含: name, color, book_count
    """
    tags_meta = _load_tags_metadata()
    tags = tags_meta.get("tags", [])
    
    # 统计每个标签的书籍数量
    books = get_all_books()
    tag_counts = {}
    for book in books:
        for tag in book.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # 更新标签计数
    for tag in tags:
        tag["book_count"] = tag_counts.get(tag["name"], 0)
    
    return tags


def create_tag(name: str, color: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    创建新标签
    
    Args:
        name: 标签名称
        color: 标签颜色（可选，默认为随机颜色）
    
    Returns:
        创建的标签信息
    """
    tags_meta = _load_tags_metadata()
    tags = tags_meta.get("tags", [])
    
    # 检查标签是否已存在
    if any(t["name"] == name for t in tags):
        logger.warning(f"标签已存在: {name}")
        return None
    
    # 默认颜色列表
    default_colors = [
        "#667eea", "#764ba2", "#f093fb", "#f5576c",
        "#4facfe", "#00f2fe", "#43e97b", "#38f9d7",
        "#fa709a", "#fee140", "#a8edea", "#fed6e3"
    ]
    
    if not color:
        color = default_colors[len(tags) % len(default_colors)]
    
    new_tag = {
        "name": name,
        "color": color,
        "created_at": int(time.time() * 1000)
    }
    
    tags.append(new_tag)
    tags_meta["tags"] = tags
    
    if _save_tags_metadata(tags_meta):
        logger.info(f"创建标签成功: {name}")
        return new_tag
    return None


def update_tag(old_name: str, new_name: Optional[str] = None, 
               color: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    更新标签
    
    Args:
        old_name: 原标签名称
        new_name: 新标签名称（可选）
        color: 新颜色（可选）
    
    Returns:
        更新后的标签信息
    """
    tags_meta = _load_tags_metadata()
    tags = tags_meta.get("tags", [])
    
    for tag in tags:
        if tag["name"] == old_name:
            if new_name and new_name != old_name:
                # 更新所有书籍中的标签名称
                _update_tag_in_books(old_name, new_name)
                tag["name"] = new_name
            if color:
                tag["color"] = color
            
            tags_meta["tags"] = tags
            if _save_tags_metadata(tags_meta):
                logger.info(f"更新标签成功: {old_name} -> {new_name or old_name}")
                return tag
            return None
    
    logger.error(f"标签不存在: {old_name}")
    return None


def delete_tag(name: str) -> bool:
    """
    删除标签
    
    Args:
        name: 标签名称
    
    Returns:
        是否删除成功
    """
    tags_meta = _load_tags_metadata()
    tags = tags_meta.get("tags", [])
    
    original_count = len(tags)
    tags = [t for t in tags if t["name"] != name]
    
    if len(tags) == original_count:
        logger.warning(f"标签不存在: {name}")
        return False
    
    # 从所有书籍中移除该标签
    _remove_tag_from_books(name)
    
    tags_meta["tags"] = tags
    if _save_tags_metadata(tags_meta):
        logger.info(f"删除标签成功: {name}")
        return True
    return False


def _update_tag_in_books(old_name: str, new_name: str) -> None:
    """更新所有书籍中的标签名称"""
    metadata = _load_books_metadata()
    
    for book_summary in metadata.get("books", []):
        book_id = book_summary.get("id")
        if not book_id:
            continue
        
        book_meta = _load_book_metadata(book_id)
        if book_meta:
            book_tags = book_meta.get("tags", [])
            if old_name in book_tags:
                book_tags = [new_name if t == old_name else t for t in book_tags]
                book_meta["tags"] = book_tags
                _save_book_metadata(book_id, book_meta)


def _remove_tag_from_books(tag_name: str) -> None:
    """从所有书籍中移除指定标签"""
    metadata = _load_books_metadata()
    
    for book_summary in metadata.get("books", []):
        book_id = book_summary.get("id")
        if not book_id:
            continue
        
        book_meta = _load_book_metadata(book_id)
        if book_meta:
            book_tags = book_meta.get("tags", [])
            if tag_name in book_tags:
                book_tags = [t for t in book_tags if t != tag_name]
                book_meta["tags"] = book_tags
                _save_book_metadata(book_id, book_meta)


# ==================== 批量操作 ====================

def delete_books_batch(book_ids: List[str]) -> Dict[str, Any]:
    """
    批量删除书籍
    
    Args:
        book_ids: 要删除的书籍ID列表
    
    Returns:
        包含成功和失败信息的字典
    """
    success_ids = []
    failed_ids = []
    
    for book_id in book_ids:
        if delete_book(book_id):
            success_ids.append(book_id)
        else:
            failed_ids.append(book_id)
    
    logger.info(f"批量删除书籍完成: 成功 {len(success_ids)}, 失败 {len(failed_ids)}")
    return {
        "success_count": len(success_ids),
        "failed_count": len(failed_ids),
        "success_ids": success_ids,
        "failed_ids": failed_ids
    }


def add_tags_to_books_batch(book_ids: List[str], tags: List[str]) -> Dict[str, Any]:
    """
    批量为书籍添加标签
    
    Args:
        book_ids: 书籍ID列表
        tags: 要添加的标签列表
    
    Returns:
        操作结果
    """
    success_count = 0
    
    for book_id in book_ids:
        book_meta = _load_book_metadata(book_id)
        if book_meta:
            current_tags = book_meta.get("tags", [])
            # 添加新标签（去重）
            for tag in tags:
                if tag not in current_tags:
                    current_tags.append(tag)
            book_meta["tags"] = current_tags
            book_meta["updated_at"] = int(time.time() * 1000)
            if _save_book_metadata(book_id, book_meta):
                success_count += 1
    
    logger.info(f"批量添加标签完成: {success_count}/{len(book_ids)}")
    return {
        "success_count": success_count,
        "total_count": len(book_ids)
    }


def remove_tags_from_books_batch(book_ids: List[str], tags: List[str]) -> Dict[str, Any]:
    """
    批量从书籍移除标签
    
    Args:
        book_ids: 书籍ID列表
        tags: 要移除的标签列表
    
    Returns:
        操作结果
    """
    success_count = 0
    
    for book_id in book_ids:
        book_meta = _load_book_metadata(book_id)
        if book_meta:
            current_tags = book_meta.get("tags", [])
            # 移除标签
            current_tags = [t for t in current_tags if t not in tags]
            book_meta["tags"] = current_tags
            book_meta["updated_at"] = int(time.time() * 1000)
            if _save_book_metadata(book_id, book_meta):
                success_count += 1
    
    logger.info(f"批量移除标签完成: {success_count}/{len(book_ids)}")
    return {
        "success_count": success_count,
        "total_count": len(book_ids)
    }
