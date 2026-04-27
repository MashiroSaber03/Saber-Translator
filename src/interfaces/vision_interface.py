"""
AI视觉OCR服务接口模块：用于调用不同服务商的视觉API进行OCR识别
"""

import logging
import time
from PIL import Image

from src.shared import constants
from src.shared.ai_providers import (
    VISION_OCR_CAPABILITY,
    normalize_provider_id,
    provider_supports_capability,
    resolve_provider_base_url,
)
from src.shared.ai_transport import OpenAICompatibleChatTransport, UnifiedVisionRequest
from src.shared.image_helpers import image_to_base64

# 设置日志
logger = logging.getLogger("VisionInterface")
_transport = OpenAICompatibleChatTransport()


def call_ai_vision_ocr_service(image_pil, provider='siliconflow', api_key=None, model_name=None, prompt=None,
                               prompt_mode: str = 'normal', use_json_format: bool = False,
                               custom_base_url=None):
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
        if not provider_supports_capability(provider_lower, VISION_OCR_CAPABILITY):
            logger.error(f"不支持的AI视觉OCR服务提供商: {provider}")
            return ""

        resolved_base_url = resolve_provider_base_url(provider_lower, custom_base_url)
        if not resolved_base_url:
            logger.error(f"未提供 {provider_lower} 的 Base URL")
            return ""

        logger.info(
            "[AI视觉OCR-请求] provider=%s, model=%s, prompt_mode=%s, json_mode=%s, base_url=%s",
            provider_lower,
            model_name,
            prompt_mode,
            use_json_format,
            resolved_base_url,
        )
        logger.info("[AI视觉OCR-请求] 实际提示词开始\n%s\n[AI视觉OCR-请求] 实际提示词结束", prompt)
        content = _transport.complete_vision(
            UnifiedVisionRequest(
                provider=provider_lower,
                api_key=api_key,
                model=model_name,
                prompt=prompt,
                image_base64=image_base64,
                base_url=custom_base_url if provider_lower == 'custom' else None,
                timeout=120.0,
                use_json_format=use_json_format,
            )
        )
        if not content:
            logger.error(f"{provider_lower} 响应格式异常或无有效结果")
            return ""

        elapsed_time = time.time() - start_time
        logger.info(f"{provider_lower} 视觉OCR识别成功，耗时: {elapsed_time:.2f}秒")
        logger.info(f"识别结果 (前100字符): {content[:100]}")
        return content.strip()
    except Exception as e:
        logger.error(f"调用AI视觉OCR服务 ({provider}) 时发生顶层异常: {e}", exc_info=True)
        return ""

def test_ai_vision_ocr(image_path, provider, api_key, model_name, prompt=None,
                       custom_base_url=None):
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
