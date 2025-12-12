"""
Manga Insight 数据 API

处理分析结果数据的获取和导出。
"""

import logging
import asyncio
from flask import request, jsonify, Response

from . import manga_insight_bp
from src.core.manga_insight.storage import AnalysisStorage
from src.core.manga_insight.features.timeline import TimelineBuilder

logger = logging.getLogger("MangaInsight.API.Data")


def run_async(coro):
    """在同步上下文中运行异步函数"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ==================== 概述数据 ====================

@manga_insight_bp.route('/<book_id>/overview', methods=['GET'])
def get_overview(book_id: str):
    """获取全书概述"""
    try:
        storage = AnalysisStorage(book_id)
        overview = run_async(storage.load_overview())
        
        return jsonify({
            "success": True,
            "overview": overview
        })
        
    except Exception as e:
        logger.error(f"获取概述失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== 页面数据 ====================

@manga_insight_bp.route('/<book_id>/pages', methods=['GET'])
def list_pages(book_id: str):
    """获取已分析的页面列表"""
    try:
        storage = AnalysisStorage(book_id)
        pages = run_async(storage.list_pages())
        
        return jsonify({
            "success": True,
            "pages": pages,
            "count": len(pages)
        })
        
    except Exception as e:
        logger.error(f"获取页面列表失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/pages/<int:page_num>', methods=['GET'])
def get_page_analysis(book_id: str, page_num: int):
    """获取单页分析结果"""
    try:
        storage = AnalysisStorage(book_id)
        analysis = run_async(storage.load_page_analysis(page_num))
        
        if not analysis:
            return jsonify({
                "success": False,
                "error": f"未找到第 {page_num} 页的分析结果"
            }), 404
        
        return jsonify({
            "success": True,
            "analysis": analysis
        })
        
    except Exception as e:
        logger.error(f"获取页面分析失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/thumbnail/<int:page_num>', methods=['GET'])
def get_page_thumbnail(book_id: str, page_num: int):
    """获取页面缩略图（自动生成小尺寸版本）"""
    import os
    import io
    import json
    from flask import send_file
    from PIL import Image
    from src.shared.path_helpers import resource_path
    from src.core import bookshelf_manager
    
    THUMB_WIDTH = 150  # 缩略图宽度
    THUMB_QUALITY = 70  # JPEG 质量
    
    try:
        # 从书架系统获取书籍信息
        book = bookshelf_manager.get_book(book_id)
        if not book:
            return Response(status=404)
        
        # 查找对应页面的图片
        chapters = book.get("chapters", [])
        sessions_base = resource_path("data/sessions/bookshelf")
        
        # 缩略图缓存目录
        thumb_cache_dir = resource_path(f"data/manga_insight/{book_id}/thumbnails")
        os.makedirs(thumb_cache_dir, exist_ok=True)
        thumb_cache_path = os.path.join(thumb_cache_dir, f"page_{page_num}.jpg")
        
        # 检查缓存
        if os.path.exists(thumb_cache_path):
            return send_file(thumb_cache_path, mimetype='image/jpeg')
        
        current_page = 0
        for chapter in chapters:
            chapter_id = chapter.get("id")
            if not chapter_id:
                continue
            
            # 从 session_meta.json 获取图片信息
            session_dir = os.path.join(sessions_base, book_id, chapter_id)
            session_meta_path = os.path.join(session_dir, "session_meta.json")
            
            if os.path.exists(session_meta_path):
                try:
                    with open(session_meta_path, "r", encoding="utf-8") as f:
                        session_data = json.load(f)
                    
                    images_meta = session_data.get("images_meta", [])
                    image_count = len(images_meta)
                    
                    for i in range(image_count):
                        current_page += 1
                        if current_page == page_num:
                            # 找到了目标页面，尝试多种图片格式
                            for ext in ['png', 'jpg', 'jpeg', 'webp']:
                                image_path = os.path.join(session_dir, f"image_{i}_original.{ext}")
                                if os.path.exists(image_path):
                                    # 生成缩略图
                                    try:
                                        with Image.open(image_path) as img:
                                            # 计算等比例缩放高度
                                            ratio = THUMB_WIDTH / img.width
                                            thumb_height = int(img.height * ratio)
                                            
                                            # 缩放并转换为 RGB（处理 RGBA/P 模式）
                                            thumb = img.resize((THUMB_WIDTH, thumb_height), Image.Resampling.LANCZOS)
                                            if thumb.mode in ('RGBA', 'P'):
                                                thumb = thumb.convert('RGB')
                                            
                                            # 保存到缓存
                                            thumb.save(thumb_cache_path, 'JPEG', quality=THUMB_QUALITY)
                                            
                                            # 返回缩略图
                                            return send_file(thumb_cache_path, mimetype='image/jpeg')
                                    except Exception as e:
                                        logger.warning(f"生成缩略图失败: {image_path}, {e}")
                                        # 降级：直接返回原图
                                        return send_file(image_path, mimetype='image/jpeg')
                            
                            return Response(status=404)
                except Exception as e:
                    logger.warning(f"读取 session_meta 失败: {session_meta_path}, {e}")
                    continue
        
        # 未找到图片
        return Response(status=404)
        
    except Exception as e:
        logger.error(f"获取缩略图失败: {e}", exc_info=True)
        return Response(status=500)


@manga_insight_bp.route('/<book_id>/page-image/<int:page_num>', methods=['GET'])
def get_page_image(book_id: str, page_num: int):
    """获取页面原图"""
    import os
    import json
    from flask import send_file
    from src.shared.path_helpers import resource_path
    from src.core import bookshelf_manager
    
    try:
        # 从书架系统获取书籍信息
        book = bookshelf_manager.get_book(book_id)
        if not book:
            return Response(status=404)
        
        # 查找对应页面的图片
        chapters = book.get("chapters", [])
        sessions_base = resource_path("data/sessions/bookshelf")
        
        current_page = 0
        for chapter in chapters:
            chapter_id = chapter.get("id")
            if not chapter_id:
                continue
            
            # 从 session_meta.json 获取图片信息
            session_dir = os.path.join(sessions_base, book_id, chapter_id)
            session_meta_path = os.path.join(session_dir, "session_meta.json")
            
            if os.path.exists(session_meta_path):
                try:
                    with open(session_meta_path, "r", encoding="utf-8") as f:
                        session_data = json.load(f)
                    
                    images_meta = session_data.get("images_meta", [])
                    image_count = len(images_meta)
                    
                    for i in range(image_count):
                        current_page += 1
                        if current_page == page_num:
                            # 找到了目标页面，尝试多种图片格式
                            for ext in ['png', 'jpg', 'jpeg', 'webp']:
                                image_path = os.path.join(session_dir, f"image_{i}_original.{ext}")
                                if os.path.exists(image_path):
                                    # 确定 MIME 类型
                                    mime_types = {
                                        'png': 'image/png',
                                        'jpg': 'image/jpeg',
                                        'jpeg': 'image/jpeg',
                                        'webp': 'image/webp'
                                    }
                                    return send_file(image_path, mimetype=mime_types.get(ext, 'image/jpeg'))
                            
                            return Response(status=404)
                except Exception as e:
                    logger.warning(f"读取 session_meta 失败: {session_meta_path}, {e}")
                    continue
        
        # 未找到图片
        return Response(status=404)
        
    except Exception as e:
        logger.error(f"获取页面图片失败: {e}", exc_info=True)
        return Response(status=500)


# ==================== 章节数据 ====================

@manga_insight_bp.route('/<book_id>/chapters', methods=['GET'])
def list_chapters(book_id: str):
    """获取已分析的章节列表"""
    try:
        storage = AnalysisStorage(book_id)
        chapters = run_async(storage.list_chapters())
        
        return jsonify({
            "success": True,
            "chapters": chapters
        })
        
    except Exception as e:
        logger.error(f"获取章节列表失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/chapters/<chapter_id>', methods=['GET'])
def get_chapter_analysis(book_id: str, chapter_id: str):
    """获取章节分析结果"""
    try:
        storage = AnalysisStorage(book_id)
        analysis = run_async(storage.load_chapter_analysis(chapter_id))
        
        if not analysis:
            return jsonify({
                "success": False,
                "error": f"未找到章节 {chapter_id} 的分析结果"
            }), 404
        
        return jsonify({
            "success": True,
            "analysis": analysis
        })
        
    except Exception as e:
        logger.error(f"获取章节分析失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== 时间线数据 ====================

@manga_insight_bp.route('/<book_id>/timeline', methods=['GET'])
def get_timeline(book_id: str):
    """
    获取剧情时间线（从缓存加载，不自动构建）
    
    时间线只在以下情况下构建：
    1. 分析完成后自动构建
    2. 用户点击刷新按钮时
    """
    try:
        storage = AnalysisStorage(book_id)
        
        # 直接从缓存加载，不自动构建
        timeline_data = run_async(storage.load_timeline())
        
        if timeline_data:
            return jsonify({
                "success": True,
                "cached": True,
                **timeline_data
            })
        else:
            # 没有缓存，返回空结果
            return jsonify({
                "success": True,
                "cached": False,
                "groups": [],
                "events": [],
                "stats": {
                    "total_events": 0,
                    "total_groups": 0,
                    "total_batches": 0,
                    "total_pages": 0
                },
                "message": "时间线尚未生成，请先完成漫画分析"
            })
        
    except Exception as e:
        logger.error(f"获取时间线失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== 导出 ====================

@manga_insight_bp.route('/<book_id>/export', methods=['GET'])
def export_analysis(book_id: str):
    """导出分析数据"""
    try:
        format_type = request.args.get('format', 'markdown')
        storage = AnalysisStorage(book_id)
        data = run_async(storage.export_all())
        
        if format_type == 'json':
            return jsonify({
                "success": True,
                "data": data
            })
        else:
            # 默认导出 Markdown
            content = _generate_markdown_report(data)
            return jsonify({
                "success": True,
                "markdown": content
            })
        
    except Exception as e:
        logger.error(f"导出分析数据失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def _generate_markdown_report(data: dict) -> str:
    """生成 Markdown 格式报告"""
    lines = []
    
    # 标题
    title = data.get("overview", {}).get("title", data.get("book_id", "漫画"))
    lines.append(f"# {title} 分析报告\n")
    
    # 概述
    overview = data.get("overview", {})
    if overview:
        lines.append("## 概述\n")
        if overview.get("summary"):
            lines.append(overview["summary"])
            lines.append("")
    
    # 时间线 - 新格式：从批量分析中提取
    batches = data.get("batches", [])
    if batches:
        lines.append("## 剧情时间线\n")
        for batch in batches:
            page_range = batch.get("page_range", {})
            start = page_range.get("start", "?")
            end = page_range.get("end", "?")
            
            lines.append(f"### 第 {start}-{end} 页")
            
            # 批次摘要
            batch_summary = batch.get("batch_summary", "")
            if batch_summary:
                lines.append(f"\n{batch_summary}\n")
            
            # 关键事件
            events = batch.get("key_events", [])
            if events:
                lines.append("**关键事件：**")
                for event in events:
                    if event:
                        lines.append(f"- {event}")
            lines.append("")
    
    lines.append(f"\n---\n导出时间: {data.get('exported_at', '')}")
    
    return "\n".join(lines)


# ==================== 重新生成 API ====================

@manga_insight_bp.route('/<book_id>/regenerate/overview', methods=['POST'])
def regenerate_overview(book_id: str):
    """重新生成概述"""
    try:
        from src.core.manga_insight.analyzer import MangaAnalyzer
        from src.core.manga_insight.config_utils import load_insight_config
        
        config = load_insight_config()
        analyzer = MangaAnalyzer(book_id, config)
        
        overview = run_async(analyzer.generate_overview())
        
        return jsonify({
            "success": True,
            "message": "概述已重新生成",
            "overview": overview
        })
        
    except Exception as e:
        logger.error(f"重新生成概述失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/rebuild-embeddings', methods=['POST'])
def rebuild_embeddings(book_id: str):
    """重新构建向量嵌入"""
    try:
        from src.core.manga_insight.analyzer import MangaAnalyzer
        from src.core.manga_insight.config_utils import load_insight_config
        from src.core.manga_insight.vector_store import MangaVectorStore
        
        config = load_insight_config()
        
        # 检查 Embedding 是否已配置
        if not config.embedding.api_key:
            return jsonify({
                "success": False,
                "error": "未配置 Embedding API Key，请先在设置中配置向量模型"
            }), 400
        
        # 1. 重新构建向量（build_embeddings 内部会先清除现有向量）
        analyzer = MangaAnalyzer(book_id, config)
        result = run_async(analyzer.build_embeddings())
        
        # 2. 获取最新统计
        vector_store = MangaVectorStore(book_id)
        stats = vector_store.get_stats()
        
        if result.get("success"):
            return jsonify({
                "success": True,
                "message": f"向量嵌入重建完成: {result.get('pages_count', 0)} 页面, {result.get('events_count', 0)} 事件",
                "stats": stats,
                "build_result": result
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get("error", "向量嵌入构建失败，可能是向量存储不可用或没有分析数据")
            }), 500
        
    except Exception as e:
        logger.error(f"重建向量嵌入失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/regenerate/timeline', methods=['POST'])
def regenerate_timeline(book_id: str):
    """
    重新生成时间线（构建并保存到缓存）
    
    支持两种模式：
    - enhanced: 增强模式，使用 LLM 进行智能整合（默认）
    - simple: 简单模式，仅提取事件列表
    
    请求体参数：
    - mode: "enhanced" 或 "simple"
    """
    try:
        # 获取模式参数
        mode = "enhanced"
        if request.is_json and request.json:
            mode = request.json.get("mode", "enhanced")
        
        storage = AnalysisStorage(book_id)
        
        if mode == "enhanced":
            # 增强模式：使用 LLM 智能整合
            from src.core.manga_insight.features.timeline_enhanced import EnhancedTimelineBuilder
            from src.core.manga_insight.config_utils import load_insight_config
            
            config = load_insight_config()
            builder = EnhancedTimelineBuilder(book_id, config)
            timeline_data = run_async(builder.build(mode="enhanced"))
        else:
            # 简单模式：使用原有逻辑
            builder = TimelineBuilder(book_id)
            timeline_data = run_async(builder.build_timeline_grouped())
            timeline_data["mode"] = "simple"
        
        # 保存到缓存
        run_async(storage.save_timeline(timeline_data))
        
        stats = timeline_data.get("stats", {})
        actual_mode = timeline_data.get("mode", mode)
        
        # 根据模式生成消息
        if actual_mode == "enhanced":
            message = f"增强时间线已生成: {stats.get('total_events', 0)} 个事件, {stats.get('total_arcs', 0)} 个剧情弧, {stats.get('total_characters', 0)} 个角色"
        else:
            message = f"时间线已生成: {stats.get('total_events', 0)} 个事件"
        
        return jsonify({
            "success": True,
            "cached": True,
            "message": message,
            **timeline_data
        })
        
    except Exception as e:
        logger.error(f"重新生成时间线失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


