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
import httpx
from pathlib import Path
from flask import Blueprint, request, jsonify, Response, stream_with_context, send_from_directory

from src.core.web_import import MangaScraperAgent, ImageDownloader, GalleryDLRunner, check_gallery_dl_support

logger = logging.getLogger("WebImportAPI")

web_import_bp = Blueprint('web_import', __name__, url_prefix='/api/web-import')

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@web_import_bp.route('/static/temp/gallery_dl/<path:filename>', methods=['GET'])
def serve_gallery_dl_temp(filename):
    """
    提供 gallery-dl 临时文件的静态访问
    """
    temp_dir = PROJECT_ROOT / "data" / "temp" / "gallery_dl"
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
    from datetime import datetime
    import sys
    
    def generate():
        try:
            # 发送开始日志
            log_data = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'info',
                'message': f'使用 Gallery-DL 引擎提取: {url}'
            }
            yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"
            
            # 创建运行器 - gallery-dl 元数据提取需要较长时间
            runner_config = {
                **config,
                'timeout': 600  # 固定10分钟超时，某些站点需要很长时间
            }
            runner = GalleryDLRunner(runner_config)
            
            # 发送提取中日志
            log_data = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'tool_call',
                'message': '调用 gallery-dl 下载预览图片（前20张）...'
            }
            yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"
            
            # 发送等待提示
            log_data = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'info',
                'message': '⏳ 正在下载预览图片，请稍候...'
            }
            yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"
            
            # 执行提取
            logger.info(f"开始 gallery-dl 提取: {url}")
            result = runner.extract_metadata(url)
            logger.info(f"gallery-dl 提取完成: success={result.success}, pages={result.total_pages}")
            
            # 发送结果日志
            if result.success:
                log_data = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': 'tool_result',
                    'message': f'提取成功: 发现 {result.total_pages} 张图片'
                }
            else:
                log_data = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': 'error',
                    'message': f'提取失败: {result.error}'
                }
            yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"
            
            # 发送结果
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
            
            # 发送结果
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
            # 使用 ImageDownloader 下载
            return _download_with_image_downloader(pages, source_url, config)
        
    except Exception as e:
        logger.exception("下载 API 错误")
        return jsonify({'success': False, 'error': str(e)}), 500


def _download_with_image_downloader(pages: list, source_url: str, config: dict):
    """使用 ImageDownloader 下载图片"""
    # 构建下载器配置
    download_config = {
        **config.get('download', {}),
        'customCookie': config.get('advanced', {}).get('customCookie', ''),
        'customHeaders': config.get('advanced', {}).get('customHeaders', ''),
        'bypassProxy': config.get('advanced', {}).get('bypassProxy', False),
        'imagePreprocess': config.get('imagePreprocess', {})
    }
    
    # 创建下载器
    downloader = ImageDownloader(download_config)
    
    # 执行下载
    results = downloader.download_all_sync(pages, source_url)
    
    # 转换结果
    images = []
    failed_count = 0
    
    for result in results:
        if result.success:
            images.append({
                'index': result.index,
                'filename': result.filename,
                'dataUrl': result.data_url,
                'size': result.size
            })
        else:
            failed_count += 1
            logger.warning(f"下载失败: {result.error}")
    
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
        
        if not api_key:
            return jsonify({'success': False, 'error': '请输入 API Key'}), 400
        
        from openai import OpenAI
        
        # 获取 base_url
        provider_urls = {
            'openai': None,
            'siliconflow': 'https://api.siliconflow.cn/v1',
            'deepseek': 'https://api.deepseek.com/v1',
            'volcano': 'https://ark.cn-beijing.volces.com/api/v3',
            'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai/',
            'custom_openai': base_url or None
        }
        
        final_base_url = base_url if base_url else provider_urls.get(provider)
        
        # 创建客户端
        client = OpenAI(
            api_key=api_key,
            base_url=final_base_url,
            timeout=30
        )
        
        # 发送测试请求
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        
        return jsonify({'success': True, 'message': '连接成功'})
        
    except Exception as e:
        error_msg = str(e)
        if 'authentication' in error_msg.lower() or '401' in error_msg:
            return jsonify({'success': False, 'error': 'API Key 无效'}), 400
        logger.exception("测试 Agent 连接失败")
        return jsonify({'success': False, 'error': error_msg}), 500
