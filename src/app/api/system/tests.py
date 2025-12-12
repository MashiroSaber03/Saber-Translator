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
from typing import Tuple, List, Dict, Any
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

logger = logging.getLogger("SystemAPI.Tests")

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


@system_bp.route('/test_ollama_connection', methods=['GET'])
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


@system_bp.route('/test_sakura_connection', methods=['GET'])
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


@system_bp.route('/test_lama_repair', methods=['GET'])
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
        
        if not os.path.exists(test_img_path):
            return jsonify({
                'success': False,
                'error': '测试图像不存在',
                'path': test_img_path
            }), 404
        
        logger.info("开始LAMA修复功能测试")
        
        # 加载测试图像
        image = Image.open(test_img_path).convert("RGB")
        
        # 创建一个简单的掩码
        mask = Image.new("RGB", image.size, color=(0, 0, 0))
        draw = ImageDraw.Draw(mask)
        
        # 在图像中央绘制一个白色矩形作为掩码
        width, height = image.size
        rect_width, rect_height = width // 3, height // 3
        left = (width - rect_width) // 2
        top = (height - rect_height) // 2
        draw.rectangle(
            [(left, top), (left + rect_width, top + rect_height)],
            fill=(255, 255, 255)
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
            
        provider = data.get('provider')
        api_key = data.get('api_key')
        model_name = data.get('model_name')
        prompt = data.get('prompt')
        custom_ai_vision_base_url = data.get('custom_ai_vision_base_url')
        
        # 检查必要参数
        missing = []
        if not provider: missing.append('provider')
        if not api_key: missing.append('api_key')
        if not model_name: missing.append('model_name')
        if provider == constants.CUSTOM_AI_VISION_PROVIDER_ID and not custom_ai_vision_base_url:
            missing.append('custom_ai_vision_base_url (当选择自定义服务时)')
        
        if missing:
            return jsonify({
                'success': False,
                'message': f'缺少必要参数: {", ".join(missing)}'
            }), 400
        
        # 检查provider是否受支持
        if provider not in constants.SUPPORTED_AI_VISION_PROVIDERS:
            return jsonify({
                'success': False,
                'message': f'不支持的AI视觉服务商: {provider}，'
                           f'支持的服务商: {", ".join(constants.SUPPORTED_AI_VISION_PROVIDERS.keys())}'
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
            'provider': 'siliconflow' | 'deepseek' | 'volcano' | 'gemini' | 'caiyun' | 'custom_openai',
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
        provider = data.get('provider', '').lower()
        api_key = data.get('api_key', '').strip()
        model_name = data.get('model_name', '').strip()
        base_url = data.get('base_url', '').strip()
        
        if not api_key:
            return jsonify({
                'success': False,
                'message': '请提供API Key'
            }), 400
        
        # 彩云小译特殊处理
        if provider == 'caiyun':
            return test_caiyun_connection(api_key)
        
        # 其他服务需要模型名称
        if not model_name:
            return jsonify({
                'success': False,
                'message': '请提供模型名称'
            }), 400
        
        # 根据服务商确定 base_url
        provider_urls = {
            'siliconflow': 'https://api.siliconflow.cn/v1',
            'deepseek': 'https://api.deepseek.com',
            'volcano': 'https://ark.cn-beijing.volces.com/api/v3',
        }
        
        if provider == 'custom_openai':
            if not base_url:
                return jsonify({
                    'success': False,
                    'message': '自定义服务需要提供Base URL'
                }), 400
            api_base_url = base_url
        elif provider == 'gemini':
            # Gemini 使用特殊处理
            return test_gemini_connection(api_key, model_name)
        elif provider in provider_urls:
            api_base_url = provider_urls[provider]
        else:
            return jsonify({
                'success': False,
                'message': f'不支持的服务商: {provider}'
            }), 400
        
        # 使用 OpenAI 兼容 API 进行测试
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key, base_url=api_base_url)
        
        # 发送简短翻译请求
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a translator. Translate to Chinese."},
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=50,
            timeout=30
        )
        
        result = response.choices[0].message.content.strip()
        
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


def test_caiyun_connection(token: str) -> tuple:
    """测试彩云小译连接"""
    try:
        response = requests.post(
            'https://api.interpreter.caiyunai.com/v1/translator',
            headers={
                'Content-Type': 'application/json',
                'X-Authorization': f'token {token}'
            },
            json={
                'source': ['Hello'],
                'trans_type': 'en2zh',
                'request_id': 'test',
                'detect': True
            },
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('target') and len(data['target']) > 0:
                result = data['target'][0]
                logger.info(f"彩云小译测试成功: Hello -> {result}")
                return jsonify({
                    'success': True,
                    'message': f'连接成功! 测试翻译: Hello → {result}'
                })
        
        # 处理错误响应
        error_data = response.json() if response.text else {}
        error_msg = error_data.get('message', response.text or f'HTTP {response.status_code}')
        
        return jsonify({
            'success': False,
            'message': f'连接失败: {error_msg}'
        }), 500
        
    except Exception as e:
        logger.error(f"测试彩云小译连接失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'连接失败: {str(e)}'
        }), 500


@system_bp.route('/fetch_models', methods=['POST'])
def fetch_models():
    """
    获取可用模型列表
    
    请求体:
        {
            'provider': 'siliconflow' | 'deepseek' | 'volcano' | 'gemini' | 'custom_openai',
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
        provider = data.get('provider', '').lower()
        api_key = data.get('api_key', '').strip()
        base_url = data.get('base_url', '').strip()
        
        if not api_key:
            return jsonify({
                'success': False,
                'message': '请提供API Key'
            }), 400
        
        models = []
        
        # 根据服务商获取模型列表
        if provider == 'siliconflow':
            models = fetch_siliconflow_models(api_key)
        elif provider == 'deepseek':
            models = fetch_openai_compatible_models(api_key, 'https://api.deepseek.com/v1')
        elif provider == 'volcano':
            models = fetch_openai_compatible_models(api_key, 'https://ark.cn-beijing.volces.com/api/v3')
        elif provider == 'gemini':
            models = fetch_gemini_models(api_key)
        elif provider == 'openai':
            models = fetch_openai_compatible_models(api_key, 'https://api.openai.com/v1')
        elif provider == 'qwen':
            models = fetch_openai_compatible_models(api_key, 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        elif provider == 'custom_openai':
            if not base_url:
                return jsonify({
                    'success': False,
                    'message': '自定义服务需要提供Base URL'
                }), 400
            models = fetch_openai_compatible_models(api_key, base_url)
        else:
            return jsonify({
                'success': False,
                'message': f'不支持的服务商: {provider}'
            }), 400
        
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


def fetch_siliconflow_models(api_key: str) -> List[Dict[str, str]]:
    """获取 SiliconFlow 可用模型列表"""
    try:
        response = requests.get(
            'https://api.siliconflow.cn/v1/models',
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get('data', []):
                model_id = model.get('id', '')
                # 只返回文本模型（排除图像、音频等）
                if model_id and ('chat' in model_id.lower() or 'llm' in model_id.lower() 
                                or 'qwen' in model_id.lower() or 'deepseek' in model_id.lower()
                                or 'glm' in model_id.lower() or 'yi-' in model_id.lower()
                                or 'internlm' in model_id.lower() or 'gemma' in model_id.lower()):
                    models.append({
                        'id': model_id,
                        'name': model_id
                    })
            return sorted(models, key=lambda x: x['id'])
        else:
            logger.warning(f"SiliconFlow API返回错误: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取SiliconFlow模型列表失败: {e}")
        return []


def fetch_openai_compatible_models(api_key: str, base_url: str) -> List[Dict[str, str]]:
    """获取 OpenAI 兼容服务的模型列表"""
    try:
        # 确保 base_url 格式正确
        if not base_url.endswith('/v1') and not base_url.endswith('/v1/'):
            if not base_url.endswith('/'):
                base_url += '/'
            base_url += 'v1'
        
        models_url = base_url.rstrip('/') + '/models'
        
        response = requests.get(
            models_url,
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get('data', []):
                model_id = model.get('id', '')
                if model_id:
                    models.append({
                        'id': model_id,
                        'name': model_id
                    })
            return sorted(models, key=lambda x: x['id'])
        else:
            logger.warning(f"API返回错误 ({base_url}): {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取模型列表失败 ({base_url}): {e}")
        return []


def fetch_gemini_models(api_key: str) -> List[Dict[str, str]]:
    """获取 Google Gemini 可用模型列表"""
    try:
        response = requests.get(
            f'https://generativelanguage.googleapis.com/v1beta/models?key={api_key}',
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get('models', []):
                model_name = model.get('name', '')
                # 提取模型ID (去掉 'models/' 前缀)
                model_id = model_name.replace('models/', '') if model_name.startswith('models/') else model_name
                display_name = model.get('displayName', model_id)
                
                # 只返回支持内容生成的模型
                supported_methods = model.get('supportedGenerationMethods', [])
                if 'generateContent' in supported_methods:
                    models.append({
                        'id': model_id,
                        'name': display_name
                    })
            return sorted(models, key=lambda x: x['id'])
        else:
            logger.warning(f"Gemini API返回错误: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取Gemini模型列表失败: {e}")
        return []


def test_gemini_connection(api_key: str, model_name: str) -> tuple:
    """测试 Google Gemini 连接"""
    try:
        # Gemini 使用 OpenAI 兼容模式
        from openai import OpenAI
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a translator. Translate to Chinese."},
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=50,
            timeout=30
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"Gemini测试成功: Hello -> {result}")
        
        return jsonify({
            'success': True,
            'message': f'连接成功! 测试翻译: Hello → {result}'
        })
        
    except Exception as e:
        logger.error(f"测试Gemini连接失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'连接失败: {str(e)}'
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
