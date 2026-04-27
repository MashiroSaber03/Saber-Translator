"""
连接测试相关API

包含所有与服务连接测试相关的API端点：
- Ollama连接测试
- Sakura连接测试
- LAMA修复测试
- 百度OCR连接测试
- AI视觉OCR连接测试
- 百度翻译连接测试
- 有道翻译连接测试
"""

import os
import io
import base64
import time
import threading
import logging
from typing import List, Dict, Any
from flask import request, jsonify
from PIL import Image, ImageDraw, ImageFont
import requests

from . import system_bp
from src.shared.path_helpers import get_debug_dir, resource_path
from src.interfaces.lama_interface import clean_image_with_lama, LAMA_AVAILABLE
from src.interfaces.baidu_ocr_interface import test_baidu_ocr_connection
from src.interfaces.vision_interface import test_ai_vision_ocr
from src.interfaces.baidu_translate_interface import baidu_translate
from src.shared import constants
from src.shared.ai_adapters import fetch_local_models, test_caiyun_connection as adapter_test_caiyun_connection
from src.shared.ai_providers import (
    CONNECTION_TEST_CAPABILITY,
    MODEL_FETCH_CAPABILITY,
    VISION_OCR_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_supports_capability,
)
from src.shared.ai_transport import (
    OpenAICompatibleChatTransport,
    ProviderConnectionTestRequest,
    ProviderModelListRequest,
)

logger = logging.getLogger("SystemAPI.Tests")
_chat_transport = OpenAICompatibleChatTransport()

# 全局变量，用于存储Sakura服务状态和模型列表
SAKURA_STATUS = {
    'available': False,
    'models': [
        "sakura-7b-qwen2.5-v1.0",
        "sakura-14b-qwen2.5-v1.0",
        "sakura-32b-qwen2beta-v0.9"
    ],
    'last_check_time': 0
}


def _request_value(data: Dict[str, Any], *keys: str, default=None):
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return default


@system_bp.route('/test_ollama_connection', methods=['POST'])
def test_ollama_connection():
    """
    测试Ollama连接状态
    
    返回:
        {
            'success': True,
            'version': '版本号',
            'models': ['model1', 'model2', ...]
        }
    """
    try:
        # 先检查Ollama服务是否可用
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code != 200:
                return jsonify({
                    'success': False,
                    'message': f'Ollama服务响应异常，状态码: {response.status_code}'
                }), 500
                
            version_info = response.json()
            version = version_info.get('version', '未知')
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'无法连接到Ollama服务，请确认服务是否启动: {str(e)}'
            }), 500
        
        # 获取已安装的模型列表
        try:
            models_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if models_response.status_code != 200:
                return jsonify({
                    'success': False,
                    'message': f'获取模型列表失败，状态码: {models_response.status_code}'
                }), 500
                
            models_data = models_response.json()
            models = models_data.get('models', [])
            model_names = [m.get('name', '') for m in models]
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'获取模型列表时出错: {str(e)}'
            }), 500
        
        logger.info(f"Ollama连接成功，版本: {version}，模型数量: {len(model_names)}")
        
        return jsonify({
            'success': True,
            'version': version,
            'models': model_names
        })
        
    except Exception as e:
        logger.error(f"测试Ollama连接时发生错误: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'测试Ollama连接时发生错误: {str(e)}'
        }), 500


@system_bp.route('/test_sakura_connection', methods=['POST'])
def test_sakura_connection():
    """
    测试Sakura服务连接状态
    
    URL参数:
        force: 是否强制刷新缓存 (true/false)
    
    返回:
        {
            'success': True,
            'models': ['model1', 'model2', ...],
            'cached': False
        }
    """
    try:
        global SAKURA_STATUS
        
        # 检查是否需要强制刷新模型列表
        force_refresh = request.args.get('force', 'false').lower() == 'true'
        current_time = time.time()
        
        # 如果上次检查时间在30秒内且不是强制刷新，则使用缓存的结果
        if not force_refresh and current_time - SAKURA_STATUS['last_check_time'] < 30 and SAKURA_STATUS['available']:
            logger.info(f"使用缓存的Sakura模型列表: {len(SAKURA_STATUS['models'])}个模型")
            return jsonify({
                'success': True,
                'models': SAKURA_STATUS['models'],
                'cached': True
            })
        
        # 增加重试次数和超时时间
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                logger.info(f"尝试连接Sakura服务 ({retry+1}/{max_retries})...")
                response = requests.get("http://localhost:8080/v1/models", timeout=10)
                
                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get('data', [])
                    model_names = [m.get('id', '') for m in models]
                    
                    # 如果没有获取到模型列表，则使用默认的模型列表
                    if not model_names:
                        model_names = SAKURA_STATUS['models']
                    
                    # 更新全局状态
                    SAKURA_STATUS['available'] = True
                    SAKURA_STATUS['models'] = model_names
                    SAKURA_STATUS['last_check_time'] = current_time
                    
                    logger.info(f"成功连接到Sakura服务，获取到 {len(model_names)} 个模型")
                    return jsonify({
                        'success': True,
                        'models': model_names,
                        'cached': False
                    })
                else:
                    # 如果不是最后一次重试，等待后继续
                    if retry < max_retries - 1:
                        logger.warning(f"Sakura服务响应异常，状态码: {response.status_code}，将在2秒后重试")
                        time.sleep(2)
                    else:
                        SAKURA_STATUS['available'] = False
                        SAKURA_STATUS['last_check_time'] = current_time
                        return jsonify({
                            'success': False,
                            'message': f'Sakura服务响应异常，状态码: {response.status_code}'
                        }), 500
                    
            except Exception as e:
                logger.warning(f"连接Sakura尝试 {retry+1}/{max_retries} 失败: {e}")
                if retry < max_retries - 1:
                    time.sleep(2)
        
        # 所有重试都失败
        SAKURA_STATUS['available'] = False
        SAKURA_STATUS['last_check_time'] = current_time
        return jsonify({
            'success': False,
            'message': '无法连接到Sakura服务，请确认服务是否启动'
        }), 500
            
    except Exception as e:
        logger.error(f"测试Sakura连接时发生错误: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'测试Sakura连接时发生错误: {str(e)}'
        }), 500


@system_bp.route('/test_lama_repair', methods=['GET', 'POST'])
def test_lama_repair():
    """
    测试LAMA修复功能
    
    创建测试图像和掩码，执行LAMA修复，返回修复结果
    
    返回:
        {
            'success': True,
            'message': 'LAMA修复成功',
            'result_image': 'base64_image_data'
        }
    """
    try:
        debug_dir = get_debug_dir()
        test_img_path = os.path.join(debug_dir, "result_image.png")
        
        if os.path.exists(test_img_path):
            image = Image.open(test_img_path).convert("RGB")
            logger.info(f"开始LAMA修复功能测试，使用现有测试图: {test_img_path}")
        else:
            logger.info("未找到现有测试图，创建自包含的 LAMA 测试图像")
            image = Image.new("RGB", (256, 256), color=(255, 255, 255))
            image_draw = ImageDraw.Draw(image)
            image_draw.rectangle([(64, 96), (192, 160)], fill=(0, 0, 0))
            image_draw.text((78, 110), "LAMA", fill=(255, 255, 255))
            image.save(test_img_path)
            logger.info(f"已创建测试图像：{test_img_path}")

        # 创建一个简单的掩码。clean_image_with_lama 期望黑色区域为修复区。
        mask = Image.new("L", image.size, color=255)
        draw = ImageDraw.Draw(mask)
        
        # 在图像中央绘制一个黑色矩形作为掩码
        width, height = image.size
        rect_width, rect_height = width // 3, height // 3
        left = (width - rect_width) // 2
        top = (height - rect_height) // 2
        draw.rectangle(
            [(left, top), (left + rect_width, top + rect_height)],
            fill=0
        )
        
        # 保存掩码供检查
        mask_path = os.path.join(debug_dir, "test_mask.png")
        mask.save(mask_path)
        logger.info(f"保存掩码图像：{mask_path}")
        
        # 确认LAMA可用
        if not LAMA_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'LAMA功能不可用',
                'LAMA_AVAILABLE': LAMA_AVAILABLE
            }), 503
        
        # 使用LAMA执行修复
        logger.info("开始使用LAMA进行修复")
        try:
            repaired_image = clean_image_with_lama(image, mask)
            if repaired_image is None:
                return jsonify({
                    'success': False,
                    'error': 'LAMA 修复返回空结果'
                }), 500
            
            # 保存修复后的图像
            result_path = os.path.join(debug_dir, "test_lama_web_result.png")
            repaired_image.save(result_path)
            logger.info(f"成功保存修复结果：{result_path}")
            
            # 转换图像为base64
            buffered = io.BytesIO()
            repaired_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'message': 'LAMA修复成功',
                'result_image': img_str
            })
        except Exception as e:
            logger.error(f"LAMA修复失败：{e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'LAMA修复失败：{str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"测试端点出错：{e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'测试端点出错：{str(e)}'
        }), 500


@system_bp.route('/test_baidu_ocr_connection', methods=['POST'])
def test_baidu_ocr_connection_api():
    """
    测试百度OCR连接状态
    
    请求体:
        {
            'api_key': 'xxx',
            'secret_key': 'xxx'
        }
    
    返回:
        {
            'success': True,
            'message': '连接成功'
        }
    """
    try:
        data = request.json
        api_key = data.get('api_key')
        secret_key = data.get('secret_key')
        
        if not api_key or not secret_key:
            return jsonify({
                'success': False,
                'message': '请提供API Key和Secret Key'
            }), 400
            
        # 调用测试函数
        result = test_baidu_ocr_connection(api_key, secret_key)
        
        return jsonify({
            'success': result.get('success', False),
            'message': result.get('message', '未知结果')
        })
        
    except Exception as e:
        logger.error(f"测试百度OCR连接时发生错误: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'测试百度OCR连接时发生错误: {str(e)}'
        }), 500


@system_bp.route('/test_ai_vision_ocr', methods=['POST'])
def test_ai_vision_ocr_api():
    """
    测试AI视觉OCR连接状态
    
    请求体:
        {
            'provider': 'siliconflow' | 'volcano' | 'gemini' | 'custom',
            'api_key': 'xxx',
            'model_name': 'xxx',
            'prompt': 'xxx',
            'custom_ai_vision_base_url': 'xxx'  # 可选，自定义服务需要
        }
    
    返回:
        {
            'success': True,
            'message': '连接成功，识别结果: xxx',
            'test_image_path': '/path/to/test_image.png'
        }
    """
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据为空'
            }), 400
            
        provider = normalize_provider_id(_request_value(data, 'provider'))
        api_key = _request_value(data, 'api_key', 'apiKey')
        model_name = _request_value(data, 'model_name', 'model', 'modelName')
        prompt = _request_value(data, 'prompt')
        custom_ai_vision_base_url = _request_value(data, 'custom_ai_vision_base_url', 'base_url', 'baseUrl', 'customBaseUrl')
        
        # 检查必要参数
        missing = []
        if not provider: missing.append('provider')
        if not api_key: missing.append('api_key')
        if not model_name: missing.append('model_name')
        if not provider_supports_capability(provider, VISION_OCR_CAPABILITY):
            return jsonify({
                'success': False,
                'message': f'不支持的AI视觉服务商: {provider}'
            }), 400
        manifest = get_provider_manifest(provider)
        if manifest.requires_base_url and not custom_ai_vision_base_url:
            missing.append('custom_ai_vision_base_url (当选择自定义服务时)')
        
        if missing:
            return jsonify({
                'success': False,
                'message': f'缺少必要参数: {", ".join(missing)}'
            }), 400
            
        # 获取或创建测试图片
        debug_dir = get_debug_dir()
        test_img_path = None
        possible_imgs = ['result_image.png', 'test_lama_web_result.png']
        
        for img_name in possible_imgs:
            path = os.path.join(debug_dir, img_name)
            if os.path.exists(path):
                test_img_path = path
                break
        
        # 如果没有现成的测试图片，创建一个简单的测试图像
        if not test_img_path:
            logger.info("未找到现有测试图片，创建简单测试图像")
            test_img_path = os.path.join(debug_dir, "ai_vision_test.png")
            test_img = Image.new('RGB', (300, 100), color=(255, 255, 255))
            draw = ImageDraw.Draw(test_img)
            try:
                font = ImageFont.truetype(resource_path(constants.DEFAULT_FONT_RELATIVE_PATH), 20)
            except:
                font = ImageFont.load_default()
            draw.text((10, 40), "AI视觉OCR测试文本", fill=(0, 0, 0), font=font)
            test_img.save(test_img_path)
        
        logger.info(f"使用测试图片: {test_img_path}")
        
        # 调用测试函数
        success, result_message = test_ai_vision_ocr(
            test_img_path,
            provider,
            api_key,
            model_name,
            prompt,
            custom_base_url=custom_ai_vision_base_url
        )
        
        return jsonify({
            'success': success,
            'message': result_message,
            'test_image_path': test_img_path
        })
        
    except Exception as e:
        logger.error(f"测试AI视觉OCR连接时发生错误: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'测试AI视觉OCR连接时发生错误: {str(e)}'
        }), 500


@system_bp.route('/test_baidu_translate_connection', methods=['POST'])
def test_baidu_translate_connection_api():
    """
    测试百度翻译API连接状态
    
    请求体:
        {
            'app_id': 'xxx',
            'app_key': 'xxx'
        }
    
    返回:
        {
            'success': True,
            'message': '连接成功'
        }
    """
    try:
        data = request.json
        app_id = data.get('app_id')
        app_key = data.get('app_key')
        
        if not app_id or not app_key:
            return jsonify({
                'success': False,
                'message': '请提供App ID和App Key'
            }), 400
            
        # 设置百度翻译接口的认证信息
        baidu_translate.set_credentials(app_id, app_key)
        
        # 调用测试连接方法
        success, message = baidu_translate.test_connection()
        
        return jsonify({
            'success': success,
            'message': message
        })
        
    except Exception as e:
        logger.error(f"测试百度翻译API连接时发生错误: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'测试百度翻译API连接时发生错误: {str(e)}'
        }), 500


@system_bp.route('/test_youdao_translate', methods=['POST'])
def test_youdao_translate():
    """
    测试有道翻译连接
    
    请求体:
        {
            'appKey': 'xxx',
            'appSecret': 'xxx'
        }
    
    返回:
        {
            'success': True,
            'message': '连接成功！测试翻译结果: xxx'
        }
    """
    data = request.get_json()
    app_key = data.get('appKey')
    app_secret = data.get('appSecret')
    
    if not app_key or not app_secret:
        return jsonify({
            'success': False,
            'message': '请提供有效的AppKey和AppSecret'
        }), 400
    
    try:
        from src.interfaces.youdao_translate_interface import YoudaoTranslateInterface
        
        # 创建接口实例
        translator = YoudaoTranslateInterface(app_key, app_secret)
        
        # 尝试翻译一个简单的测试文本
        test_text = "Hello, this is a test."
        result = translator.translate(test_text, from_lang="en", to_lang="zh-CHS")
        
        if result and result != test_text:
            return jsonify({
                'success': True,
                'message': f'连接成功！测试翻译结果: {result}'
            })
        else:
            return jsonify({
                'success': False,
                'message': '连接失败：未获得预期的翻译结果'
            }), 500
    except Exception as e:
        error_msg = str(e)
        logger.error(f"测试有道翻译失败: {error_msg}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'连接失败：{error_msg}'
        }), 500


@system_bp.route('/test_ai_translate_connection', methods=['POST'])
def test_ai_translate_connection():
    """
    测试AI翻译服务连接（通用接口）
    
    支持: SiliconFlow, DeepSeek, 火山引擎, Gemini, 彩云小译, 自定义OpenAI兼容服务
    
    请求体:
        {
            'provider': 'siliconflow' | 'deepseek' | 'volcano' | 'gemini' | 'caiyun' | 'custom',
            'api_key': 'xxx',
            'model_name': 'xxx',  // 可选，彩云小译不需要
            'base_url': 'xxx'     // 仅自定义服务需要
        }
    
    返回:
        {
            'success': True,
            'message': '连接成功，测试翻译结果: xxx'
        }
    """
    try:
        data = request.json
        provider = normalize_provider_id(_request_value(data, 'provider', default=''))
        api_key = (_request_value(data, 'api_key', 'apiKey', default='') or '').strip()
        model_name = (_request_value(data, 'model_name', 'model', 'modelName', default='') or '').strip()
        base_url = (_request_value(data, 'base_url', 'custom_base_url', 'baseUrl', 'customBaseUrl', default='') or '').strip()
        
        if not api_key:
            return jsonify({
                'success': False,
                'message': '请提供API Key'
            }), 400
        
        # 彩云小译特殊处理
        if provider == 'caiyun':
            success, result = adapter_test_caiyun_connection(api_key)
            if success:
                return jsonify({
                    'success': True,
                    'message': f'连接成功! 测试翻译: Hello → {result}'
                })
            return jsonify({'success': False, 'message': f'连接失败: {result}'}), 500
        
        # 其他服务需要模型名称
        if not model_name:
            return jsonify({
                'success': False,
                'message': '请提供模型名称'
            }), 400
        
        # 根据服务商确定 base_url
        if not provider_supports_capability(provider, CONNECTION_TEST_CAPABILITY):
            return jsonify({
                'success': False,
                'message': f'不支持的服务商: {provider}'
            }), 400
        manifest = get_provider_manifest(provider)
        if manifest.requires_base_url and not base_url:
            return jsonify({
                'success': False,
                'message': '自定义服务需要提供Base URL'
            }), 400

        success, result = _chat_transport.test_connection(
            ProviderConnectionTestRequest(
                provider=provider,
                api_key=api_key,
                model=model_name,
                base_url=base_url or None,
            )
        )
        if not success:
            raise ValueError(result)
        
        logger.info(f"AI翻译服务测试成功 ({provider}): Hello -> {result}")
        
        return jsonify({
            'success': True,
            'message': f'连接成功! 测试翻译: Hello → {result}'
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"测试AI翻译服务连接失败 ({data.get('provider', 'unknown')}): {error_msg}", exc_info=True)
        
        # 提取更友好的错误信息
        if 'authentication' in error_msg.lower() or 'api key' in error_msg.lower():
            friendly_msg = 'API Key 无效或已过期'
        elif 'model' in error_msg.lower() and 'not found' in error_msg.lower():
            friendly_msg = f'模型 {model_name} 不存在或无权访问'
        elif 'timeout' in error_msg.lower():
            friendly_msg = '连接超时，请检查网络'
        elif 'connection' in error_msg.lower():
            friendly_msg = '无法连接到服务器'
        else:
            friendly_msg = error_msg
        
        return jsonify({
            'success': False,
            'message': f'连接失败: {friendly_msg}'
        }), 500


@system_bp.route('/fetch_models', methods=['POST'])
def fetch_models():
    """
    获取可用模型列表
    
    请求体:
        {
            'provider': 'siliconflow' | 'deepseek' | 'volcano' | 'gemini' | 'custom',
            'api_key': 'xxx',
            'base_url': 'xxx'  // 仅自定义服务需要
        }
    
    返回:
        {
            'success': True,
            'models': [
                {'id': 'model-id', 'name': 'Model Name'},
                ...
            ]
        }
    """
    try:
        data = request.json
        provider = normalize_provider_id(_request_value(data, 'provider', default=''))
        api_key = (_request_value(data, 'api_key', 'apiKey', default='') or '').strip()
        base_url = (_request_value(data, 'base_url', 'baseUrl', 'customBaseUrl', default='') or '').strip()
        
        if not api_key:
            return jsonify({
                'success': False,
                'message': '请提供API Key'
            }), 400
        
        models = []
        
        if not provider_supports_capability(provider, MODEL_FETCH_CAPABILITY):
            return jsonify({
                'success': False,
                'message': f'不支持的服务商: {provider}'
            }), 400

        manifest = get_provider_manifest(provider)
        if manifest.kind == 'local':
            models = fetch_local_models(provider)
        elif manifest.requires_base_url and not base_url:
            return jsonify({
                'success': False,
                'message': '自定义服务需要提供Base URL'
            }), 400
        else:
            models = _chat_transport.list_models(
                ProviderModelListRequest(
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url or None,
                )
            )
        
        return jsonify({
            'success': True,
            'models': models
        })
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'获取模型列表失败: {str(e)}'
        }), 500


# ============ Sakura服务监控线程 ============

def check_services_availability():
    """后台定期检查Sakura服务可用性"""
    global SAKURA_STATUS
    
    service_logger = logging.getLogger("SakuraServiceChecker")
    service_logger.info("启动Sakura服务监控线程")
    
    while True:
        try:
            response = requests.get("http://localhost:8080/v1/models", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('data', [])
                model_names = [m.get('id', '') for m in models]
                
                if not model_names:
                    model_names = [
                        "sakura-7b-qwen2.5-v1.0",
                        "sakura-14b-qwen2.5-v1.0",
                        "sakura-32b-qwen2beta-v0.9"
                    ]
                
                was_available = SAKURA_STATUS['available']
                SAKURA_STATUS['available'] = True
                SAKURA_STATUS['models'] = model_names
                SAKURA_STATUS['last_check_time'] = time.time()
                
                if not was_available:
                    service_logger.info(f"Sakura服务已连接，可用模型: {', '.join(model_names)}")
            else:
                if SAKURA_STATUS['available']:
                    service_logger.warning(f"Sakura服务响应异常，状态码: {response.status_code}")
                    SAKURA_STATUS['available'] = False
        except Exception as e:
            if SAKURA_STATUS['available']:
                service_logger.warning(f"Sakura服务连接中断: {e}")
                SAKURA_STATUS['available'] = False
        
        # 每30秒检查一次
        time.sleep(30)


def start_service_monitor():
    """启动定期检查服务可用性的后台线程"""
    service_check_thread = threading.Thread(target=check_services_availability, daemon=True)
    service_check_thread.start()
    logger.info("Sakura服务监控线程已启动")
    return service_check_thread
