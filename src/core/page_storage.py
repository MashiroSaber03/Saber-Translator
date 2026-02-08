# src/core/page_storage.py
"""
单页存储管理模块

新的存储结构：每张图片独立保存，支持增量更新。

存储结构：
data/sessions/{session_path}/
├── session_meta.json          # 会话元数据
├── images/
│   ├── 0/                     # 第1页
│   │   ├── original.png       # 原图
│   │   ├── clean.png          # 干净背景
│   │   ├── translated.png     # 译图
│   │   └── meta.json          # 该页的元数据
│   ├── 1/                     # 第2页
│   └── ...
"""

import os
import json
import base64
import logging
import time
import threading
from typing import Optional, Dict, Any, List

from src.shared.path_helpers import resource_path

logger = logging.getLogger("PageStorage")

# 常量
SESSION_BASE_DIR = "sessions"
BOOKSHELF_BASE_DIR = "bookshelf"  # 新增：书架数据目录
SESSION_META_FILENAME = "session_meta.json"
PAGE_META_FILENAME = "meta.json"
IMAGES_DIR = "images"

# 线程锁：防止并发写入冲突
_locks: Dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    """获取指定键的锁"""
    with _locks_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _get_session_base_dir() -> str:
    """获取会话基础目录（独立会话）"""
    base_path = resource_path(os.path.join("data", SESSION_BASE_DIR))
    os.makedirs(base_path, exist_ok=True)
    return base_path


def _is_bookshelf_path(session_path: str) -> bool:
    """判断是否是书架路径（新格式或旧格式）"""
    # 新格式书架路径: bookshelf/{book_id}/chapters/{chapter_id}/session
    # 旧格式书架路径: bookshelf/{book_id}/{chapter_id}
    # 规范化路径分隔符后再检查
    normalized = session_path.replace("\\", "/")
    return normalized.startswith("bookshelf/")


def _convert_old_to_new_path(session_path: str) -> str:
    """
    将旧格式路径转换为新格式路径（如果新格式存在）
    旧格式: bookshelf/{book_id}/{chapter_id}
    新格式: bookshelf/{book_id}/chapters/{chapter_id}/session
    """
    normalized = session_path.replace("\\", "/")
    parts = normalized.split("/")

    # 检查是否是旧格式: bookshelf/{book_id}/{chapter_id}（3段且不含 chapters）
    if len(parts) == 3 and parts[0] == "bookshelf" and "chapters" not in normalized:
        book_id, chapter_id = parts[1], parts[2]
        new_format = f"bookshelf/{book_id}/chapters/{chapter_id}/session"
        new_full_path = resource_path(os.path.join("data", new_format))

        # 如果新格式路径存在，使用新格式
        if os.path.isdir(new_full_path):
            return new_format

    return session_path


def _get_session_path(session_path: str) -> str:
    """
    获取完整的会话目录路径

    - 书架路径（新格式）: data/bookshelf/{book_id}/chapters/{chapter_id}/session
    - 书架路径（旧格式）: data/bookshelf/{book_id}/{chapter_id}
    - 独立会话路径: data/sessions/{session_path}
    """
    if _is_bookshelf_path(session_path):
        # 尝试转换旧格式到新格式
        converted_path = _convert_old_to_new_path(session_path)
        # 书架路径：直接使用 data/ 作为基础
        full_path = resource_path(os.path.join("data", converted_path))

        # 如果转换后的路径不存在，回退到原路径
        if not os.path.isdir(full_path):
            full_path = resource_path(os.path.join("data", session_path))
    else:
        # 独立会话：使用 data/sessions/ 作为基础
        base_dir = _get_session_base_dir()
        full_path = os.path.join(base_dir, session_path)
    return full_path


def _get_page_dir(session_path: str, page_index: int) -> str:
    """获取单页目录路径"""
    session_dir = _get_session_path(session_path)
    return os.path.join(session_dir, IMAGES_DIR, str(page_index))


def _safe_write_json(filepath: str, data: dict) -> bool:
    """安全写入 JSON 文件（使用临时文件 + 原子替换）"""
    try:
        temp_path = filepath + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, filepath)
        return True
    except Exception as e:
        logger.error(f"写入 JSON 失败: {filepath} - {e}")
        # 清理临时文件
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass
        return False


def _safe_read_json(filepath: str) -> Optional[dict]:
    """安全读取 JSON 文件"""
    try:
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return None
            return json.loads(content)
    except Exception as e:
        logger.error(f"读取 JSON 失败: {filepath} - {e}")
        return None


# ============================================================
# 会话元数据操作
# ============================================================

def save_session_meta(session_path: str, metadata: dict) -> dict:
    """
    保存会话元数据
    
    Args:
        session_path: 会话路径（如 bookshelf/book123/chapter1）
        metadata: 元数据字典，包含：
            - ui_settings: UI 设置
            - total_pages: 总页数
            - currentImageIndex: 当前图片索引
    
    Returns:
        {"success": True} 或 {"success": False, "error": "..."}
    """
    session_dir = _get_session_path(session_path)
    lock = _get_lock(f"session_meta:{session_path}")
    
    try:
        with lock:
            os.makedirs(session_dir, exist_ok=True)
            
            meta_to_save = {
                "metadata": {
                    "name": session_path,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "3.0.0"  # 新版本
                },
                "ui_settings": metadata.get("ui_settings", {}),
                "total_pages": metadata.get("total_pages", 0),
                "currentImageIndex": metadata.get("currentImageIndex", 0)
            }
            
            filepath = os.path.join(session_dir, SESSION_META_FILENAME)
            if _safe_write_json(filepath, meta_to_save):
                logger.info(f"会话元数据已保存: {session_path}")
                return {"success": True}
            else:
                return {"success": False, "error": "写入失败"}
    except Exception as e:
        logger.error(f"保存会话元数据失败: {e}")
        return {"success": False, "error": str(e)}


def load_session_meta(session_path: str) -> Optional[dict]:
    """
    加载会话元数据
    
    Returns:
        元数据字典，或 None（如果不存在）
    """
    session_dir = _get_session_path(session_path)
    filepath = os.path.join(session_dir, SESSION_META_FILENAME)
    
    data = _safe_read_json(filepath)
    if data:
        logger.info(f"会话元数据已加载: {session_path}")
    return data


# ============================================================
# 单页图片操作
# ============================================================

def save_page_image(session_path: str, page_index: int, image_type: str, base64_data: str) -> dict:
    """
    保存单页图片
    
    Args:
        session_path: 会话路径
        page_index: 页面索引（0-based）
        image_type: 图片类型：'original' | 'clean' | 'translated'
        base64_data: Base64 编码的图片数据（可带或不带 data: 前缀）
    
    Returns:
        {"success": True} 或 {"success": False, "error": "..."}
    """
    if image_type not in ('original', 'clean', 'translated'):
        return {"success": False, "error": f"无效的图片类型: {image_type}"}
    
    if not base64_data:
        return {"success": True}  # 空数据，跳过
    
    page_dir = _get_page_dir(session_path, page_index)
    lock = _get_lock(f"page:{session_path}:{page_index}")
    
    try:
        with lock:
            os.makedirs(page_dir, exist_ok=True)
            
            # 处理 base64 数据
            if ',' in base64_data:
                base64_data = base64_data.split(',', 1)[1]
            
            # 解码并保存
            image_bytes = base64.b64decode(base64_data)
            filepath = os.path.join(page_dir, f"{image_type}.png")
            
            # 使用临时文件 + 原子替换
            temp_path = filepath + '.tmp'
            with open(temp_path, 'wb') as f:
                f.write(image_bytes)
            os.replace(temp_path, filepath)
            
            logger.debug(f"页面图片已保存: {session_path}/page_{page_index}/{image_type}")
            return {"success": True}
    except Exception as e:
        logger.error(f"保存页面图片失败: {e}")
        return {"success": False, "error": str(e)}


def load_page_image(session_path: str, page_index: int, image_type: str) -> Optional[bytes]:
    """
    加载单页图片
    
    Returns:
        图片二进制数据，或 None（如果不存在）
    """
    if image_type not in ('original', 'clean', 'translated'):
        return None
    
    page_dir = _get_page_dir(session_path, page_index)
    filepath = os.path.join(page_dir, f"{image_type}.png")
    
    try:
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"加载页面图片失败: {e}")
        return None


def page_image_exists(session_path: str, page_index: int, image_type: str) -> bool:
    """检查页面图片是否存在"""
    page_dir = _get_page_dir(session_path, page_index)
    filepath = os.path.join(page_dir, f"{image_type}.png")
    return os.path.exists(filepath)


# ============================================================
# 单页元数据操作
# ============================================================

def save_page_meta(session_path: str, page_index: int, meta: dict) -> dict:
    """
    保存单页元数据
    
    Args:
        session_path: 会话路径
        page_index: 页面索引
        meta: 页面元数据（气泡坐标、文字等）
    
    Returns:
        {"success": True} 或 {"success": False, "error": "..."}
    """
    page_dir = _get_page_dir(session_path, page_index)
    lock = _get_lock(f"page:{session_path}:{page_index}")
    
    try:
        with lock:
            os.makedirs(page_dir, exist_ok=True)
            filepath = os.path.join(page_dir, PAGE_META_FILENAME)
            
            if _safe_write_json(filepath, meta):
                logger.debug(f"页面元数据已保存: {session_path}/page_{page_index}")
                return {"success": True}
            else:
                return {"success": False, "error": "写入失败"}
    except Exception as e:
        logger.error(f"保存页面元数据失败: {e}")
        return {"success": False, "error": str(e)}


def load_page_meta(session_path: str, page_index: int) -> Optional[dict]:
    """
    加载单页元数据
    
    Returns:
        元数据字典，或 None（如果不存在）
    """
    page_dir = _get_page_dir(session_path, page_index)
    filepath = os.path.join(page_dir, PAGE_META_FILENAME)
    return _safe_read_json(filepath)


# ============================================================
# 批量操作
# ============================================================

def presave_all_pages(session_path: str, images: List[dict], ui_settings: dict) -> dict:
    """
    预保存所有页面（批量保存 = 循环调用单页保存）
    
    Args:
        session_path: 会话路径
        images: 图片列表，每个元素包含：
            - originalDataURL: 原图 base64
            - translatedDataURL: 译图 base64（可选）
            - cleanImageData: 干净背景 base64（可选）
            - 其他元数据字段
        ui_settings: UI 设置
    
    Returns:
        {"success": True, "saved": n} 或 {"success": False, "error": "..."}
    """
    logger.info(f"开始批量保存: {session_path}, 共 {len(images)} 页")
    
    try:
        # 1. 保存会话元数据
        result = save_session_meta(session_path, {
            "ui_settings": ui_settings,
            "total_pages": len(images),
            "currentImageIndex": 0
        })
        if not result.get("success"):
            return result
        
        saved_count = 0
        
        # 2. 逐页保存（批量保存就是循环单页保存）
        for page_index, img in enumerate(images):
            # 保存原图（每次都保存，确保数据完整）
            original = img.get("originalDataURL")
            if original and isinstance(original, str) and original.startswith("data:"):
                result = save_page_image(session_path, page_index, "original", original)
                if not result.get("success"):
                    logger.warning(f"保存原图失败: page_{page_index}")
            
            # 保存译图
            translated = img.get("translatedDataURL")
            if translated and isinstance(translated, str) and translated.startswith("data:"):
                result = save_page_image(session_path, page_index, "translated", translated)
                if not result.get("success"):
                    logger.warning(f"保存译图失败: page_{page_index}")
            
            # 保存干净背景
            clean = img.get("cleanImageData")
            if clean and isinstance(clean, str) and not clean.startswith("/api/"):
                # cleanImageData 可能是纯 base64（不带 data: 前缀）
                if not clean.startswith("data:"):
                    clean = f"data:image/png;base64,{clean}"
                result = save_page_image(session_path, page_index, "clean", clean)
                if not result.get("success"):
                    logger.warning(f"保存干净背景失败: page_{page_index}")
            
            # 保存页面元数据
            page_meta = {}
            skip_keys = {"originalDataURL", "translatedDataURL", "cleanImageData"}
            for key, value in img.items():
                if key not in skip_keys:
                    page_meta[key] = value
            
            # 添加状态标记
            page_meta["hasOriginal"] = page_image_exists(session_path, page_index, "original")
            page_meta["hasTranslated"] = page_image_exists(session_path, page_index, "translated")
            page_meta["hasClean"] = page_image_exists(session_path, page_index, "clean")
            
            save_page_meta(session_path, page_index, page_meta)
            saved_count += 1
            
            logger.debug(f"页面 {page_index} 已保存")
        
        logger.info(f"批量保存完成: {session_path}, 共保存 {saved_count} 页")
        return {"success": True, "saved": saved_count}
    
    except Exception as e:
        logger.error(f"批量保存失败: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def save_translated_page(session_path: str, page_index: int, 
                         translated_data: Optional[str] = None,
                         clean_data: Optional[str] = None,
                         page_meta: Optional[dict] = None) -> dict:
    """
    保存翻译完成的页面（只更新译图和干净背景）
    
    Args:
        session_path: 会话路径
        page_index: 页面索引
        translated_data: 译图 base64（可选）
        clean_data: 干净背景 base64（可选）
        page_meta: 页面元数据更新（可选）
    
    Returns:
        {"success": True} 或 {"success": False, "error": "..."}
    """
    try:
        # 保存译图
        if translated_data:
            result = save_page_image(session_path, page_index, "translated", translated_data)
            if not result.get("success"):
                return result
        
        # 保存干净背景
        if clean_data:
            result = save_page_image(session_path, page_index, "clean", clean_data)
            if not result.get("success"):
                return result
        
        # 更新页面元数据
        if page_meta:
            # 合并现有元数据
            existing = load_page_meta(session_path, page_index) or {}
            existing.update(page_meta)
            existing["hasTranslated"] = page_image_exists(session_path, page_index, "translated")
            existing["hasClean"] = page_image_exists(session_path, page_index, "clean")
            save_page_meta(session_path, page_index, existing)
        
        logger.debug(f"翻译页面已保存: {session_path}/page_{page_index}")
        return {"success": True}
    
    except Exception as e:
        logger.error(f"保存翻译页面失败: {e}")
        return {"success": False, "error": str(e)}


def load_session(session_path: str) -> Optional[dict]:
    """
    加载完整会话（用于前端恢复）
    
    Returns:
        {
            "ui_settings": {...},
            "total_pages": n,
            "currentImageIndex": 0,
            "pages": [
                {
                    "index": 0,
                    "hasOriginal": True,
                    "hasTranslated": True,
                    "hasClean": True,
                    "originalUrl": "/api/sessions/page/...",
                    "translatedUrl": "/api/sessions/page/...",
                    "cleanUrl": "/api/sessions/page/...",
                    ... 其他元数据 ...
                },
                ...
            ]
        }
    """
    session_meta = load_session_meta(session_path)
    if not session_meta:
        return None
    
    total_pages = session_meta.get("total_pages", 0)
    pages = []
    
    for page_index in range(total_pages):
        page_meta = load_page_meta(session_path, page_index) or {}
        
        # 检查各类型图片是否存在
        has_original = page_image_exists(session_path, page_index, "original")
        has_translated = page_image_exists(session_path, page_index, "translated")
        has_clean = page_image_exists(session_path, page_index, "clean")
        
        page_data = {
            "index": page_index,
            "hasOriginal": has_original,
            "hasTranslated": has_translated,
            "hasClean": has_clean,
            **page_meta
        }
        
        # 生成图片 URL
        if has_original:
            page_data["originalUrl"] = f"/api/sessions/page/{session_path}/{page_index}/original"
        if has_translated:
            page_data["translatedUrl"] = f"/api/sessions/page/{session_path}/{page_index}/translated"
        if has_clean:
            page_data["cleanUrl"] = f"/api/sessions/page/{session_path}/{page_index}/clean"
        
        pages.append(page_data)
    
    return {
        "ui_settings": session_meta.get("ui_settings", {}),
        "total_pages": total_pages,
        "currentImageIndex": session_meta.get("currentImageIndex", 0),
        "pages": pages
    }


def get_page_count(session_path: str) -> int:
    """获取会话的页面数量"""
    session_meta = load_session_meta(session_path)
    if session_meta:
        return session_meta.get("total_pages", 0)
    
    # 如果元数据不存在，尝试从目录结构推断
    session_dir = _get_session_path(session_path)
    images_dir = os.path.join(session_dir, IMAGES_DIR)
    
    if not os.path.exists(images_dir):
        return 0
    
    count = 0
    for item in os.listdir(images_dir):
        if item.isdigit():
            count = max(count, int(item) + 1)
    
    return count
