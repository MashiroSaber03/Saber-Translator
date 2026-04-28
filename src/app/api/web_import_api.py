"""
网页漫画导入 API

提供网页漫画导入功能的 REST API 接口：
- POST /api/web-import/extract - 提取漫画图片 (支持双引擎)
- POST /api/web-import/download - 下载图片 (支持双引擎)
- GET /api/web-import/check-support - 检查 URL 是否支持 gallery-dl
- GET /api/web-import/proxy-image - 代理图片请求 (解决防盗链)
- POST /api/web-import/test-firecrawl - 测试 Firecrawl 连接
- POST /api/web-import/test-agent - 测试 AI Agent 连接
"""

import logging
import json
import os
import httpx
import threading
import time
from pathlib import Path
from datetime import datetime
from flask import Blueprint, request, jsonify, Response, stream_with_context, send_from_directory

from src.core.web_import import MangaScraperAgent, GalleryDLRunner, check_gallery_dl_support
from src.shared import constants
from src.shared.ai_providers import normalize_provider_id
from src.shared.config_loader import get_config_path, load_json_config, save_json_config

logger = logging.getLogger("WebImportAPI")

web_import_bp = Blueprint('web_import', __name__, url_prefix='/api/web-import')

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _normalize_web_import_settings_payload(payload: dict | None) -> dict:
    payload = payload if isinstance(payload, dict) else {}
    settings = payload.get('settings', {})
    provider_configs = payload.get('providerConfigs', {})

    normalized_settings = settings.copy() if isinstance(settings, dict) else {}
    agent_settings = normalized_settings.get('agent')
    if isinstance(agent_settings, dict) and agent_settings.get('provider') is not None:
        normalized_settings['agent'] = {
            **agent_settings,
            'provider': normalize_provider_id(agent_settings.get('provider'))
        }

    normalized_agent_configs = {}
    raw_agent_configs = provider_configs.get('agent', {}) if isinstance(provider_configs, dict) else {}
    if isinstance(raw_agent_configs, dict):
        for provider, config in raw_agent_configs.items():
            canonical_provider = normalize_provider_id(provider)
            if not canonical_provider or not isinstance(config, dict):
                continue
            normalized_agent_configs[canonical_provider] = {
                'apiKey': config.get('apiKey', '') or '',
                'modelName': config.get('modelName', '') or '',
                'customBaseUrl': config.get('customBaseUrl', '') or '',
            }

    return {
        'settings': normalized_settings,
        'providerConfigs': {
            'agent': normalized_agent_configs
        }
    }


@web_import_bp.route('/settings', methods=['GET'])
def get_web_import_settings():
    has_stored_settings = os.path.exists(get_config_path(constants.WEB_IMPORT_SETTINGS_FILE))
    payload = load_json_config(
        constants.WEB_IMPORT_SETTINGS_FILE,
        default_value={'settings': {}, 'providerConfigs': {'agent': {}}}
    )
    normalized = _normalize_web_import_settings_payload(payload)
    return jsonify({'success': True, 'hasStoredSettings': has_stored_settings, **normalized})


@web_import_bp.route('/settings', methods=['POST'])
def save_web_import_settings():
    data = request.get_json() or {}
    settings = data.get('settings')
    provider_configs = data.get('providerConfigs')

    if not isinstance(settings, dict) or not isinstance(provider_configs, dict):
        return jsonify({'success': False, 'error': '缺少设置数据'}), 400

    normalized_payload = _normalize_web_import_settings_payload({
        'settings': settings,
        'providerConfigs': provider_configs
    })

    success = save_json_config(constants.WEB_IMPORT_SETTINGS_FILE, normalized_payload)
    if not success:
        return jsonify({'success': False, 'error': '保存设置失败'}), 500

    return jsonify({'success': True})


@web_import_bp.route('/static/temp/gallery_dl/<path:filename>', methods=['GET'])
def serve_gallery_dl_temp(filename):
    """
    提供 gallery-dl 临时文件的静态访问
    """
    temp_dir = PROJECT_ROOT / "data" / "temp" / "gallery_dl"
    return send_from_directory(str(temp_dir), filename)


@web_import_bp.route('/static/temp/ai_agent/<path:filename>', methods=['GET'])
def serve_ai_agent_temp(filename):
    """
    提供 AI Agent 临时文件的静态访问
    """
    temp_dir = PROJECT_ROOT / "data" / "temp" / "ai_agent"
    return send_from_directory(str(temp_dir), filename)


@web_import_bp.route('/gallery-dl-images', methods=['GET'])
def get_gallery_dl_images():
    """
    获取 gallery-dl 临时目录中的所有图片
    返回 base64 编码的图片数据，供前端直接导入
    """
    import base64
    
    temp_dir = PROJECT_ROOT / "data" / "temp" / "gallery_dl"
    
    if not temp_dir.exists():
        return jsonify({'success': False, 'error': '临时目录不存在', 'images': []})
    
    images = []
    all_files = list(temp_dir.glob('*'))
    all_files.sort(key=lambda p: p.name)
    
    for file_path in all_files:
        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']:
            try:
                with open(file_path, 'rb') as f:
                    img_data = f.read()
                ext = file_path.suffix.lower().lstrip('.')
                if ext == 'jpg':
                    ext = 'jpeg'
                base64_data = base64.b64encode(img_data).decode('utf-8')
                images.append({
                    'filename': file_path.name,
                    'data': f'data:image/{ext};base64,{base64_data}'
                })
            except Exception as e:
                logger.warning(f"读取图片失败 {file_path}: {e}")
    
    return jsonify({'success': True, 'images': images, 'total': len(images)})


@web_import_bp.route('/check-support', methods=['GET'])
def check_support():
    """
    检查 URL 是否支持 gallery-dl
    
    Query Parameters:
        url: 要检查的 URL
    
    Response:
        {
            "available": true,   // gallery-dl 是否可用
            "supported": true    // URL 是否支持
        }
    """
    url = request.args.get('url', '').strip()
    
    if not url:
        return jsonify({'available': False, 'supported': False})
    
    try:
        result = check_gallery_dl_support(url)
        return jsonify(result)
    except Exception as e:
        logger.exception("检查 gallery-dl 支持失败")
        return jsonify({'available': False, 'supported': False, 'error': str(e)})


@web_import_bp.route('/proxy-image', methods=['GET'])
def proxy_image():
    """
    代理图片请求，解决防盗链 403 问题
    
    Query Parameters:
        url: 图片 URL
        referer: Referer 头 (可选)
    
    Response:
        图片二进制数据
    """
    url = request.args.get('url', '').strip()
    referer = request.args.get('referer', '').strip()
    
    if not url:
        return "No URL provided", 400
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }
    
    if referer:
        headers["Referer"] = referer
    
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            
            content_type = resp.headers.get('content-type', 'image/jpeg')
            return Response(resp.content, mimetype=content_type)
    except httpx.HTTPStatusError as e:
        logger.warning(f"代理图片失败: {e.response.status_code} - {url}")
        return f"HTTP Error: {e.response.status_code}", e.response.status_code
    except Exception as e:
        logger.exception(f"代理图片异常: {url}")
        return f"Proxy Error: {str(e)}", 502


@web_import_bp.route('/extract', methods=['POST'])
def extract_images():
    """
    提取漫画图片 (支持双引擎)
    
    Request Body:
        {
            "url": "漫画网页URL",
            "engine": "auto" | "gallery-dl" | "ai-agent",  // 引擎选择
            "config": { ... WebImportSettings }
        }
    
    Response (SSE Stream):
        event: log
        data: {"timestamp": "...", "type": "info", "message": "..."}
        
        event: result
        data: {"success": true, "comicTitle": "...", "engine": "gallery-dl", ...}
    """
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        engine = data.get('engine', 'auto')  # auto | gallery-dl | ai-agent
        config = data.get('config', {})
        
        if not url:
            return jsonify({'success': False, 'error': '请输入网址'}), 400
        
        # 确定使用哪个引擎
        use_gallery_dl = False
        
        if engine == 'gallery-dl':
            use_gallery_dl = True
        elif engine == 'ai-agent':
            use_gallery_dl = False
        else:  # auto
            # 自动检测：优先使用 gallery-dl
            support_info = check_gallery_dl_support(url)
            use_gallery_dl = support_info.get('available') and support_info.get('supported')
        
        if use_gallery_dl:
            # 使用 Gallery-DL 引擎
            return _extract_with_gallery_dl(url, config)
        else:
            # 使用 AI Agent 引擎
            return _extract_with_ai_agent(url, config)
        
    except Exception as e:
        logger.exception("提取 API 错误")
        return jsonify({'success': False, 'error': str(e)}), 500



def _extract_with_gallery_dl(url: str, config: dict):
    """使用 Gallery-DL 引擎提取"""
    
    def generate():
        try:
            # 发送开始日志
            log_data = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'info',
                'message': f'使用 Gallery-DL 引擎提取: {url}'
            }
            yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"
            
            # 创建运行器 - 无超时限制，等待下载完成
            runner_config = {
                **config,
                'timeout': 0  # 0 表示无超时限制
            }
            runner = GalleryDLRunner(runner_config)
            
            # 发送提取中日志
            log_data = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'tool_call',
                'message': '调用 gallery-dl 开始下载图片...'
            }
            yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"
            
            # 发送等待提示
            log_data = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'info',
                'message': '⏳ 开始下载，图片将实时显示...'
            }
            yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"
            
            # 定义进度回调：每下载一张图片就推送
            pages_yielded = []  # 用于收集已推送的页面
            
            def on_page_downloaded(page_data):
                """每发现一张新图片的回调"""
                pages_yielded.append(page_data)
            
            # 执行提取（带进度回调）
            logger.info(f"开始 gallery-dl 提取: {url}")
            
            # 由于回调无法直接在生成器中 yield，我们需要使用轮询方式
            result_container = {'result': None, 'error': None}
            
            def extraction_thread():
                try:
                    result_container['result'] = runner.extract_metadata(url, on_progress=on_page_downloaded)
                except Exception as e:
                    result_container['error'] = e
            
            # 启动提取线程
            thread = threading.Thread(target=extraction_thread, daemon=True)
            thread.start()
            
            # 轮询检查新页面并推送
            last_count = 0
            while thread.is_alive() or last_count < len(pages_yielded):
                # 推送新发现的页面
                while last_count < len(pages_yielded):
                    page_data = pages_yielded[last_count]
                    page_event = {
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'type': 'info',
                        'message': f'📥 第 {page_data["pageNumber"]} 张图片已下载'
                    }
                    yield f"event: log\ndata: {json.dumps(page_event, ensure_ascii=False)}\n\n"
                    
                    # 推送图片数据
                    yield f"event: page\ndata: {json.dumps(page_data, ensure_ascii=False)}\n\n"
                    
                    last_count += 1
                
                # 等待一小段时间再检查
                time.sleep(0.1)
            
            # 等待线程完成
            thread.join(timeout=1)
            
            # 检查是否有错误
            if result_container['error']:
                raise result_container['error']
            
            result = result_container['result']
            logger.info(f"gallery-dl 提取完成: success={result.success}, pages={result.total_pages}")
            
            # 发送完成日志
            if result.success:
                log_data = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': 'tool_result',
                    'message': f'✅ 提取完成: 共 {result.total_pages} 张图片'
                }
            else:
                log_data = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': 'error',
                    'message': f'提取失败: {result.error}'
                }
            yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"
            
            # 发送最终结果
            result_data = {
                'success': result.success,
                'comicTitle': result.comic_title,
                'chapterTitle': result.chapter_title,
                'pages': result.pages,
                'totalPages': result.total_pages,
                'sourceUrl': result.source_url,
                'referer': result.referer,
                'engine': 'gallery-dl',
                'error': result.error
            }
            yield f"event: result\ndata: {json.dumps(result_data, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.exception("Gallery-DL 提取过程发生错误")
            error_log = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'error',
                'message': f'异常: {str(e)}'
            }
            yield f"event: log\ndata: {json.dumps(error_log, ensure_ascii=False)}\n\n"
            
            error_data = {
                'success': False,
                'engine': 'gallery-dl',
                'error': str(e)
            }
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


def _extract_with_ai_agent(url: str, config: dict):
    """使用 AI Agent 引擎提取"""
    
    # 检查必要的配置
    firecrawl_key = config.get('firecrawl', {}).get('apiKey', '')
    agent_key = config.get('agent', {}).get('apiKey', '')
    
    if not firecrawl_key:
        return jsonify({'success': False, 'error': '请配置 Firecrawl API Key'}), 400
    if not agent_key:
        return jsonify({'success': False, 'error': '请配置 AI Agent API Key'}), 400
    
    # 临时目录（与 gallery-dl 使用相同的结构，便于统一处理）
    temp_dir = Path("data/temp/ai_agent")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 每次提取前清理旧文件
    for old_file in temp_dir.glob("*"):
        try:
            old_file.unlink()
        except:
            pass
    
    def generate():
        try:
            # 创建 Agent
            agent = MangaScraperAgent(config)
            
            # 收集日志的包装器
            collected_logs = []
            def collect_log(log):
                collected_logs.append(log)
            
            # 直接调用同步版本的 extract 方法
            result = agent.extract(url, collect_log)
            
            # 发送收集的日志
            for log in collected_logs:
                log_data = {
                    'timestamp': log.timestamp,
                    'type': log.type,
                    'message': log.message
                }
                yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"
            
            # 如果提取成功，下载图片到临时目录
            if result.success and result.pages:
                yield f"event: log\ndata: {json.dumps({'timestamp': datetime.now().strftime('%H:%M:%S'), 'type': 'info', 'message': f'开始下载 {len(result.pages)} 张图片...'}, ensure_ascii=False)}\n\n"
                
                # 从原始 URL 提取 Referer
                from urllib.parse import urlparse
                parsed = urlparse(url)
                referer = f"{parsed.scheme}://{parsed.netloc}/"
                
                # 下载每张图片
                downloaded_pages = []
                for i, page in enumerate(result.pages):
                    page_num = page.get('pageNumber', i + 1)
                    image_url = page.get('imageUrl', '')
                    
                    if not image_url or image_url.startswith('blob:'):
                        continue
                    
                    try:
                        # 下载图片
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Referer': referer
                        }
                        
                        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                            resp = client.get(image_url, headers=headers)
                            resp.raise_for_status()
                            
                            # 确定文件扩展名
                            content_type = resp.headers.get('content-type', '')
                            if 'webp' in content_type:
                                ext = '.webp'
                            elif 'png' in content_type:
                                ext = '.png'
                            elif 'gif' in content_type:
                                ext = '.gif'
                            else:
                                ext = '.jpg'
                            
                            # 保存到临时目录
                            filename = f"{page_num:04d}{ext}"
                            filepath = temp_dir / filename
                            filepath.write_bytes(resp.content)
                            
                            # 更新页面数据，使用本地静态路径
                            local_url = f"/api/web-import/static/temp/ai_agent/{filename}"
                            downloaded_pages.append({
                                'pageNumber': page_num,
                                'imageUrl': local_url,
                                'originalUrl': image_url  # 保留原始 URL 以备用
                            })
                            
                            # 推送进度
                            log_msg = {
                                'timestamp': datetime.now().strftime('%H:%M:%S'),
                                'type': 'info',
                                'message': f'📥 第 {page_num} 张图片已下载 ({i + 1}/{len(result.pages)})'
                            }
                            yield f"event: log\ndata: {json.dumps(log_msg, ensure_ascii=False)}\n\n"
                            
                            # 推送页面数据（与 gallery-dl 相同格式）
                            page_data = {
                                'pageNumber': page_num,
                                'imageUrl': local_url,
                                'originalUrl': image_url
                            }
                            yield f"event: page\ndata: {json.dumps(page_data, ensure_ascii=False)}\n\n"
                            
                    except Exception as e:
                        logger.warning(f"下载图片失败 {image_url}: {e}")
                        # 下载失败时使用原始 URL
                        downloaded_pages.append({
                            'pageNumber': page_num,
                            'imageUrl': image_url
                        })
                
                # 更新结果中的 pages
                if downloaded_pages:
                    result.pages = downloaded_pages
                    result.total_pages = len(downloaded_pages)
                
                yield f"event: log\ndata: {json.dumps({'timestamp': datetime.now().strftime('%H:%M:%S'), 'type': 'tool_result', 'message': f'✅ 图片下载完成: {len(downloaded_pages)} 张'}, ensure_ascii=False)}\n\n"
            
            # 发送最终结果
            result_data = {
                'success': result.success,
                'comicTitle': result.comic_title,
                'chapterTitle': result.chapter_title,
                'pages': result.pages,
                'totalPages': result.total_pages,
                'sourceUrl': result.source_url,
                'referer': '',
                'engine': 'ai-agent',
                'error': result.error
            }
            yield f"event: result\ndata: {json.dumps(result_data, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.exception("AI Agent 提取过程发生错误")
            error_data = {
                'success': False,
                'engine': 'ai-agent',
                'error': str(e)
            }
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@web_import_bp.route('/download', methods=['POST'])
def download_images():
    """
    下载图片 (支持双引擎)
    
    Request Body:
        {
            "pages": [{"pageNumber": 1, "imageUrl": "..."}, ...],
            "sourceUrl": "来源页面URL",
            "engine": "gallery-dl" | "ai-agent",  // 使用的引擎
            "config": { ... WebImportSettings }
        }
    
    Response:
        {
            "success": true,
            "images": [{"index": 0, "filename": "...", "dataUrl": "...", "size": 123}, ...],
            "failedCount": 0
        }
    """
    try:
        data = request.get_json()
        pages = data.get('pages', [])
        source_url = data.get('sourceUrl', '')
        engine = data.get('engine', 'ai-agent')
        config = data.get('config', {})
        
        if not pages:
            return jsonify({'success': False, 'error': '没有要下载的图片'}), 400
        
        # 根据引擎选择下载方式
        if engine == 'gallery-dl':
            # 使用 Gallery-DL 托管下载
            return _download_with_gallery_dl(source_url, pages, config)
        else:
            # AI Agent 模式：检查图片是否已在临时目录
            # 如果 imageUrl 以 /api/web-import/static/temp/ai_agent/ 开头，说明已下载
            return _download_with_ai_agent_cache(pages, source_url, config)
        
    except Exception as e:
        logger.exception("下载 API 错误")
        return jsonify({'success': False, 'error': str(e)}), 500


def _download_with_ai_agent_cache(pages: list, source_url: str, config: dict):
    """
    使用 AI Agent 缓存下载图片
    如果图片已在临时目录，直接读取；否则重新下载
    """
    import base64
    
    temp_dir = Path("data/temp/ai_agent")
    images = []
    failed_count = 0
    
    # 获取图片预处理配置
    preprocess = config.get('imagePreprocess', {})
    
    for i, page in enumerate(pages):
        page_num = page.get('pageNumber', i + 1)
        image_url = page.get('imageUrl', '')
        original_url = page.get('originalUrl', '')  # 原始远程 URL
        
        try:
            # 检查是否是本地缓存文件
            if image_url.startswith('/api/web-import/static/temp/ai_agent/'):
                # 从本地读取
                filename = image_url.split('/')[-1]
                filepath = temp_dir / filename
                
                if filepath.exists():
                    # 读取本地文件
                    image_data = filepath.read_bytes()
                    
                    # 应用图片预处理（如果需要）
                    if preprocess.get('enabled'):
                        from src.core.web_import.image_processor import ImageProcessor
                        processor = ImageProcessor(preprocess)
                        image_data = processor.process(image_data)
                    
                    # 转换为 base64
                    ext = filepath.suffix.lower()
                    mime_type = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.webp': 'image/webp',
                        '.gif': 'image/gif'
                    }.get(ext, 'image/jpeg')
                    
                    data_url = f"data:{mime_type};base64,{base64.b64encode(image_data).decode()}"
                    
                    images.append({
                        'index': page_num - 1,
                        'filename': f"page_{page_num:04d}{ext}",
                        'dataUrl': data_url,
                        'size': len(image_data)
                    })
                    logger.debug(f"从缓存读取图片: {filename}")
                    continue
            
            # 不是本地缓存，使用 ImageDownloader 下载
            # 这种情况一般是下载失败回退到原始 URL
            download_url = original_url if original_url else image_url
            if download_url and not download_url.startswith('blob:'):
                from src.core.web_import.image_downloader import ImageDownloader
                downloader = ImageDownloader(config.get('download', {}))
                result = downloader.download_single_sync(download_url, source_url)
                
                if result.get('success'):
                    images.append({
                        'index': page_num - 1,
                        'filename': f"page_{page_num:04d}.jpg",
                        'dataUrl': result.get('dataUrl', ''),
                        'size': result.get('size', 0)
                    })
                    continue
            
            # 下载失败
            failed_count += 1
            logger.warning(f"图片下载失败: {image_url}")
            
        except Exception as e:
            failed_count += 1
            logger.warning(f"处理图片失败 {image_url}: {e}")
    
    return jsonify({
        'success': True,
        'images': images,
        'failedCount': failed_count
    })


def _download_with_gallery_dl(source_url: str, pages: list, config: dict):
    """使用 Gallery-DL 托管下载图片"""
    # 获取选中的页码
    selected_indices = [p.get('pageNumber', i + 1) for i, p in enumerate(pages)]
    
    # 构建配置
    runner_config = {
        'timeout': config.get('download', {}).get('timeout', 600),
        'imagePreprocess': config.get('imagePreprocess', {})
    }
    
    # 创建运行器
    runner = GalleryDLRunner(runner_config)
    
    # 执行下载
    results = runner.download(source_url, selected_indices)
    
    # 转换结果
    images = []
    failed_count = 0
    
    for result in results:
        if result.get('success', False):
            images.append({
                'index': result.get('index', 0),
                'filename': result.get('filename', ''),
                'dataUrl': result.get('dataUrl', ''),
                'size': result.get('size', 0)
            })
        else:
            failed_count += 1
            logger.warning(f"Gallery-DL 下载失败: {result.get('error', 'Unknown error')}")
    
    return jsonify({
        'success': True,
        'images': images,
        'failedCount': failed_count
    })


@web_import_bp.route('/test-firecrawl', methods=['POST'])
def test_firecrawl_connection():
    """
    测试 Firecrawl 连接
    
    Request Body:
        {
            "apiKey": "Firecrawl API Key"
        }
    
    Response:
        {
            "success": true,
            "message": "连接成功"
        }
    """
    try:
        data = request.get_json()
        api_key = data.get('apiKey', '').strip()
        
        if not api_key:
            return jsonify({'success': False, 'error': '请输入 API Key'}), 400
        
        import httpx
        
        # 测试 API 连接
        response = httpx.get(
            'https://api.firecrawl.dev/v1/scrape',
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=10
        )
        
        # 401 表示 API Key 无效，但连接成功
        # 其他状态码可能表示服务可用
        if response.status_code in (401, 403):
            return jsonify({'success': False, 'error': 'API Key 无效'}), 400
        
        return jsonify({'success': True, 'message': '连接成功'})
        
    except httpx.TimeoutException:
        return jsonify({'success': False, 'error': '连接超时'}), 500
    except Exception as e:
        logger.exception("测试 Firecrawl 连接失败")
        return jsonify({'success': False, 'error': str(e)}), 500


@web_import_bp.route('/test-agent', methods=['POST'])
def test_agent_connection():
    """
    测试 AI Agent 连接
    
    Request Body:
        {
            "provider": "服务商",
            "apiKey": "API Key",
            "customBaseUrl": "自定义地址",
            "modelName": "模型名称"
        }
    
    Response:
        {
            "success": true,
            "message": "连接成功"
        }
    """
    try:
        data = request.get_json()
        provider = data.get('provider', 'openai')
        api_key = data.get('apiKey', '').strip()
        base_url = data.get('customBaseUrl', '').strip()
        model_name = data.get('modelName', 'gpt-4o-mini')
        
        # 日志：接收到的参数
        logger.info(f"[Agent测试] 服务商: {provider}")
        logger.info(f"[Agent测试] 模型名称: {model_name}")
        logger.info(f"[Agent测试] 自定义Base URL: {base_url if base_url else '(未设置)'}")
        logger.info(f"[Agent测试] API Key前缀: {api_key[:15]}..." if len(api_key) > 15 else f"[Agent测试] API Key: {api_key}")
        
        if not api_key:
            return jsonify({'success': False, 'error': '请输入 API Key'}), 400
        
        from src.shared.ai_providers import (
            CHAT_CAPABILITY,
            WEB_IMPORT_AGENT_CAPABILITY,
            normalize_provider_id,
            provider_supports_capability,
            resolve_provider_base_url_for_capability,
        )
        from src.shared.openai_helpers import create_openai_client
        
        normalized_provider = normalize_provider_id(provider)
        if normalized_provider and not provider_supports_capability(normalized_provider, WEB_IMPORT_AGENT_CAPABILITY):
            return jsonify({'success': False, 'error': f'不支持的服务商: {provider}'}), 400
        if normalized_provider == 'openai':
            final_base_url = None
        else:
            final_base_url = resolve_provider_base_url_for_capability(normalized_provider, CHAT_CAPABILITY, base_url)
        
        # 日志：最终使用的配置
        logger.info(f"[Agent测试] 最终Base URL: {final_base_url if final_base_url else '(使用OpenAI默认)'}")
        logger.info(f"[Agent测试] 开始调用 {model_name} 模型...")
        
        # 创建客户端
        client = create_openai_client(
            api_key=api_key,
            base_url=final_base_url,
            timeout=30,
        )
        
        # 发送测试请求
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        
        logger.info(f"[Agent测试] ✅ 连接成功！响应: {response.choices[0].message.content if response.choices else '(无内容)'}")
        return jsonify({'success': True, 'message': '连接成功'})
        
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        # 详细的错误日志
        logger.error(f"[Agent测试] ❌ 连接失败")
        logger.error(f"[Agent测试] 错误类型: {error_type}")
        logger.error(f"[Agent测试] 错误信息: {error_msg}")
        
        # 如果有更多错误细节
        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            logger.error(f"[Agent测试] HTTP状态码: {e.response.status_code}")
            logger.error(f"[Agent测试] 响应体: {e.response.text if hasattr(e.response, 'text') else '(无)'}")
        
        # 解析具体错误
        if 'authentication' in error_msg.lower() or '401' in error_msg:
            return jsonify({'success': False, 'error': 'API Key 无效'}), 400
        elif '403' in error_msg or 'permission' in error_msg.lower():
            return jsonify({'success': False, 'error': f'权限错误(403): {error_msg}'}), 400
        elif 'not found' in error_msg.lower() or '404' in error_msg:
            return jsonify({'success': False, 'error': f'模型或端点不存在(404): {model_name}'}), 400
        
        logger.exception("测试 Agent 连接失败")
        return jsonify({'success': False, 'error': error_msg}), 500
