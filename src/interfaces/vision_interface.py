"""
AI视觉OCR服务接口模块：用于调用不同服务商的视觉API进行OCR识别
"""

import logging
import base64
import requests
import json
import time
from io import BytesIO
from PIL import Image
from src.shared.openai_helpers import create_openai_client  # 导入辅助函数
from src.shared.http_retry import post_with_retry

from src.shared import constants
from src.shared.image_helpers import image_to_base64

# 设置日志
logger = logging.getLogger("VisionInterface")


def _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt, base_url_to_use, service_friendly_name, start_time):
    """
    通用的 OpenAI 兼容视觉 API 调用函数。
    """
    logger.info(f"开始调用 {service_friendly_name} 视觉API (通过 OpenAI SDK)，模型: {model_name}, BaseURL: {base_url_to_use}")
    try:
        if not base_url_to_use: 
            logger.error(f"调用 {service_friendly_name} 失败：未提供 Base URL。")
            return ""

        # ✨ 使用辅助函数创建客户端，自动处理代理与反爬伪装
        client = create_openai_client(api_key=api_key, base_url=base_url_to_use)

        payload_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=payload_messages
        )

        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            logger.info(f"{service_friendly_name} 视觉OCR识别成功，耗时: {elapsed_time:.2f}秒")
            logger.info(f"识别结果 (前100字符): {content[:100]}")
            return content.strip()
        else:
            logger.error(f"{service_friendly_name} 响应格式异常或无有效结果, 响应: {response}")
            return ""
    except Exception as e:
        logger.error(f"调用 {service_friendly_name} 视觉API ({base_url_to_use}) 时发生异常: {e}", exc_info=True)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                logger.error(f"{service_friendly_name} API 错误详情: {error_detail}")
            except json.JSONDecodeError:
                logger.error(f"{service_friendly_name} API 原始错误响应 (状态码 {e.response.status_code}): {e.response.text}")
        return ""


def call_ai_vision_ocr_service(image_pil, provider='siliconflow', api_key=None, model_name=None, prompt=None, custom_base_url=None):
    if not image_pil:
        logger.error("未提供有效图像")
        return ""
    if not api_key:
        logger.error(f"未提供 {provider} 的API密钥")
        return ""
    if not model_name:
        logger.error(f"未提供 {provider} 的模型名称")
        return ""
    if not prompt:
        prompt = constants.DEFAULT_AI_VISION_OCR_PROMPT
        logger.info(f"使用默认AI视觉OCR提示词")

    start_time = time.time()
    try:
        image_base64 = image_to_base64(image_pil)
    except Exception as e:
        logger.error(f"图像转Base64失败: {e}")
        return ""

    try:
        provider_lower = provider.lower()
        if provider_lower == 'siliconflow':
            return call_siliconflow_vision_api(image_base64, api_key, model_name, prompt, start_time)
        elif provider_lower == 'volcano':
            return _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt,
                                                   "https://ark.cn-beijing.volces.com/api/v3",
                                                   "火山引擎", start_time)
        elif provider_lower == 'gemini':
            return _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt,
                                                   "https://generativelanguage.googleapis.com/v1beta/openai/",
                                                   "Gemini Vision", start_time)
        elif provider_lower == constants.CUSTOM_AI_VISION_PROVIDER_ID:
            if not custom_base_url:
                logger.error(f"未提供自定义AI视觉OCR服务的Base URL (provider: {provider})")
                return ""
            return _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt,
                                                   custom_base_url,
                                                   "自定义OpenAI兼容视觉服务", start_time)
        else:
            logger.error(f"不支持的AI视觉OCR服务提供商: {provider}")
            return ""
    except Exception as e:
        logger.error(f"调用AI视觉OCR服务 ({provider}) 时发生顶层异常: {e}", exc_info=True)
        return ""


def call_siliconflow_vision_api(image_base64, api_key, model_name, prompt, start_time):
    """调用SiliconFlow的视觉API进行OCR识别"""
    logger.info(f"开始调用SiliconFlow视觉API进行OCR识别，模型: {model_name}")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        # ✨ 新增：为原生 requests 请求加上防拦截伪装头
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        response = post_with_retry(
            "https://api.siliconflow.cn/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=(10, 120),
            max_retries=3,
            backoff_base=1.0,
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                elapsed_time = time.time() - start_time
                logger.info(f"SiliconFlow视觉OCR识别成功，耗时: {elapsed_time:.2f}秒")
                logger.info(f"识别结果: {content}")
                return content.strip()
            else:
                logger.error(f"SiliconFlow响应格式异常: {result}")
                return ""
        else:
            logger.error(f"SiliconFlow API请求失败: HTTP {response.status_code}, {response.text}")
            return ""
            
    except requests.exceptions.Timeout:
        logger.error("SiliconFlow API请求超时")
        return ""
    except requests.exceptions.ConnectionError:
        logger.error("SiliconFlow API连接错误")
        return ""
    except Exception as e:
        logger.error(f"调用SiliconFlow视觉API时发生异常: {e}")
        return ""


def test_ai_vision_ocr(image_path, provider, api_key, model_name, prompt=None, custom_base_url=None):
    try:
        with Image.open(image_path) as img:
            result = call_ai_vision_ocr_service( 
                img,
                provider,
                api_key,
                model_name,
                prompt,
                custom_base_url=custom_base_url 
            )

            if result:
                logger.info(f"测试成功，服务商: {provider}, 模型: {model_name}, 识别结果 (部分): {result[:100]}...")
                return True, f"识别成功 (部分结果: {result[:50]}...)"
            else:
                logger.error(f"测试失败，服务商: {provider}, 模型: {model_name}, 未返回有效识别结果")
                return False, "OCR识别失败，未返回有效结果"
    except FileNotFoundError:
        logger.error(f"测试图片未找到: {image_path}")
        return False, f"测试图片未找到: {image_path}"
    except Exception as e:
        logger.error(f"测试过程中发生错误 (服务商: {provider}, 模型: {model_name}): {e}", exc_info=True)
        return False, f"测试出错: {str(e)}"
