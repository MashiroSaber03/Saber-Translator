"""
网页漫画导入 API

提供网页漫画导入功能的 REST API 接口：
- POST /api/web-import/extract - 提取漫画图片
- POST /api/web-import/download - 下载图片
- GET /api/web-import/test-connection - 测试连接
"""

import logging
import json
from flask import Blueprint, request, jsonify, Response, stream_with_context

from src.core.web_import import MangaScraperAgent, ImageDownloader

logger = logging.getLogger("WebImportAPI")

web_import_bp = Blueprint('web_import', __name__, url_prefix='/api/web-import')


@web_import_bp.route('/extract', methods=['POST'])
def extract_images():
    """
    提取漫画图片
    
    Request Body:
        {
            "url": "漫画网页URL",
            "config": { ... WebImportSettings }
        }
    
    Response (SSE Stream):
        event: log
        data: {"timestamp": "...", "type": "info", "message": "..."}
        
        event: result
        data: {"success": true, "comicTitle": "...", ...}
    """
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        config = data.get('config', {})
        
        if not url:
            return jsonify({'success': False, 'error': '请输入网址'}), 400
        
        # 检查必要的配置
        firecrawl_key = config.get('firecrawl', {}).get('apiKey', '')
        agent_key = config.get('agent', {}).get('apiKey', '')
        
        if not firecrawl_key:
            return jsonify({'success': False, 'error': '请配置 Firecrawl API Key'}), 400
        if not agent_key:
            return jsonify({'success': False, 'error': '请配置 AI Agent API Key'}), 400
        
        def generate():
            """SSE 生成器"""
            logs = []
            result = None
            
            def on_log(log):
                logs.append(log)
                log_data = {
                    'timestamp': log.timestamp,
                    'type': log.type,
                    'message': log.message
                }
                yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"
            
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
                    'error': result.error
                }
                yield f"event: result\ndata: {json.dumps(result_data, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                logger.exception("提取过程发生错误")
                error_data = {
                    'success': False,
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
        
    except Exception as e:
        logger.exception("提取 API 错误")
        return jsonify({'success': False, 'error': str(e)}), 500


@web_import_bp.route('/download', methods=['POST'])
def download_images():
    """
    下载图片
    
    Request Body:
        {
            "pages": [{"pageNumber": 1, "imageUrl": "..."}, ...],
            "sourceUrl": "来源页面URL",
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
        config = data.get('config', {})
        
        if not pages:
            return jsonify({'success': False, 'error': '没有要下载的图片'}), 400
        
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
        
    except Exception as e:
        logger.exception("下载 API 错误")
        return jsonify({'success': False, 'error': str(e)}), 500


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
