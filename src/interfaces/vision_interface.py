"""
AI视觉OCR服务接口模块：用于调用不同服务商的视觉API进行OCR识别
"""

import logging
import requests
import json
import time
from io import BytesIO
from PIL import Image

from src.shared import constants
from src.shared.ai_providers import normalize_provider_id, provider_supports_capability
from src.shared.ai_transport import OpenAICompatibleChatTransport, UnifiedVisionRequest
from src.shared.image_helpers import image_to_base64

# 设置日志
logger = logging.getLogger("VisionInterface")
_transport = OpenAICompatibleChatTransport()

# VVVVVV 新增：通用的 OpenAI 兼容视觉 API 调用函数 VVVVVV
def _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt, base_url_to_use, service_friendly_name, start_time, use_json_format=False):
    """
    通用的 OpenAI 兼容视觉 API 调用函数。
    """
    logger.info(f"开始调用 {service_friendly_name} 视觉API (通过 OpenAI SDK)，模型: {model_name}, BaseURL: {base_url_to_use}")
    try:
        if not base_url_to_use: # 增加对 base_url_to_use 的检查
            logger.error(f"调用 {service_friendly_name} 失败：未提供 Base URL。")
            return ""

        content = _transport.complete_vision(
            UnifiedVisionRequest(
                provider=service_friendly_name,
                api_key=api_key,
                model=model_name,
                prompt=prompt,
                image_base64=image_base64,
                base_url=base_url_to_use,
                timeout=120.0,
                use_json_format=use_json_format,
            )
        )
        if content:
            elapsed_time = time.time() - start_time
            logger.info(f"{service_friendly_name} 视觉OCR识别成功，耗时: {elapsed_time:.2f}秒")
            logger.info(f"识别结果 (前100字符): {content[:100]}")
            return content.strip()
        else:
            logger.error(f"{service_friendly_name} 响应格式异常或无有效结果")
            return ""
    except Exception as e:
        logger.error(f"调用 {service_friendly_name} 视觉API ({base_url_to_use}) 时发生异常: {e}", exc_info=True)
        # 记录更详细的错误响应 (如果可用)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                logger.error(f"{service_friendly_name} API 错误详情: {error_detail}")
            except json.JSONDecodeError:
                logger.error(f"{service_friendly_name} API 原始错误响应 (状态码 {e.response.status_code}): {e.response.text}")
        return ""
# ^^^^^^ 结束新增辅助函数 ^^^^^^


def call_ai_vision_ocr_service(image_pil, provider='siliconflow', api_key=None, model_name=None, prompt=None,
                               prompt_mode: str = 'normal', use_json_format: bool = False,
                               # VVVVVV 新增 custom_base_url 参数 VVVVVV
                               custom_base_url=None):
                               # ^^^^^^ 结束新增 ^^^^^^
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
        provider_lower = normalize_provider_id(provider)
        if not provider_supports_capability(provider_lower, "vision_ocr"):
            logger.error(f"不支持的AI视觉OCR服务提供商: {provider}")
            return ""
        logger.info(
            "[AI视觉OCR-请求] provider=%s, model=%s, prompt_mode=%s, json_mode=%s, base_url=%s",
            provider_lower,
            model_name,
            prompt_mode,
            use_json_format,
            custom_base_url if provider_lower == 'custom' else '(provider default)',
        )
        logger.info("[AI视觉OCR-请求] 实际提示词开始\n%s\n[AI视觉OCR-请求] 实际提示词结束", prompt)
        if provider_lower == 'siliconflow':
            return _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt,
                                                   "https://api.siliconflow.cn/v1",
                                                   "siliconflow", start_time, use_json_format)
        elif provider_lower == 'volcano':
            return _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt,
                                                   "https://ark.cn-beijing.volces.com/api/v3",
                                                   "volcano", start_time, use_json_format)
        elif provider_lower == 'gemini':
            return _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt,
                                                   "https://generativelanguage.googleapis.com/v1beta/openai/",
                                                   "gemini", start_time, use_json_format)
        elif provider_lower == 'custom':
            if not custom_base_url: # 检查 custom_base_url
                logger.error(f"未提供自定义AI视觉OCR服务的Base URL (provider: {provider})")
                return ""
            return _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt,
                                                    custom_base_url, # <<< 使用传入的自定义 Base URL
                                                    "custom", start_time, use_json_format)
        else:
            logger.error(f"不支持的AI视觉OCR服务提供商: {provider}")
            return ""
    except Exception as e: # 捕获 call_xxx 函数可能抛出的其他未预料错误
        logger.error(f"调用AI视觉OCR服务 ({provider}) 时发生顶层异常: {e}", exc_info=True)
        return ""

# 更新 test_ai_vision_ocr 函数签名和内部调用
def test_ai_vision_ocr(image_path, provider, api_key, model_name, prompt=None,
                       # VVVVVV 新增 custom_base_url 参数 VVVVVV
                       custom_base_url=None):
                       # ^^^^^^ 结束新增 ^^^^^^
    try:
        # 加载图片
        with Image.open(image_path) as img:
            # 调用OCR服务
            result = call_ai_vision_ocr_service( # 调用更新后的主服务函数
                img,
                provider,
                api_key,
                model_name,
                prompt,
                prompt_mode='normal',
                use_json_format=False,
                custom_base_url=custom_base_url # <<< 传递自定义 Base URL
            )

            if result:
                logger.info(f"测试成功，服务商: {provider}, 模型: {model_name}, 识别结果 (部分): {result[:100]}...")
                return True, f"识别成功 (部分结果: {result[:50]}...)" # 返回更简洁的消息给前端
            else:
                logger.error(f"测试失败，服务商: {provider}, 模型: {model_name}, 未返回有效识别结果")
                return False, "OCR识别失败，未返回有效结果"
    except FileNotFoundError:
        logger.error(f"测试图片未找到: {image_path}")
        return False, f"测试图片未找到: {image_path}"
    except Exception as e:
        logger.error(f"测试过程中发生错误 (服务商: {provider}, 模型: {model_name}): {e}", exc_info=True)
        return False, f"测试出错: {str(e)}"
