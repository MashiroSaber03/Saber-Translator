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
from openai import OpenAI  # 添加OpenAI库导入，用于火山引擎调用
from src.shared.openai_helpers import create_openai_client  # 导入辅助函数

from src.shared import constants
from src.shared.image_helpers import image_to_base64

# 设置日志
logger = logging.getLogger("VisionInterface")


def _log_llm_usage(provider: str, model: str, usage) -> None:
    if usage is None:
        logger.info(f"[LLM usage] provider={provider} model={model} usage=missing")
        return

    try:
        if hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
        elif isinstance(usage, dict):
            usage_dict = usage
        else:
            usage_dict = getattr(usage, "__dict__", {}) or {}

        prompt_tokens = usage_dict.get("prompt_tokens")
        completion_tokens = usage_dict.get("completion_tokens")
        total_tokens = usage_dict.get("total_tokens")

        prompt_details = usage_dict.get("prompt_tokens_details") or {}
        if hasattr(prompt_details, "model_dump"):
            prompt_details = prompt_details.model_dump()
        elif not isinstance(prompt_details, dict):
            prompt_details = getattr(prompt_details, "__dict__", {}) or {}

        cached_tokens = prompt_details.get("cached_tokens")
        if cached_tokens is None and isinstance(prompt_details, dict):
            cached_tokens = 0

        parts = [
            f"provider={provider}",
            f"model={model}",
        ]
        if prompt_tokens is not None:
            parts.append(f"prompt={int(prompt_tokens)}")
        parts.append(f"cached={int(cached_tokens) if cached_tokens is not None else 'n/a'}")
        if completion_tokens is not None:
            parts.append(f"completion={int(completion_tokens)}")
        if total_tokens is not None:
            parts.append(f"total={int(total_tokens)}")

        logger.info("[LLM usage] " + " ".join(parts))
    except Exception:
        return

# VVVVVV 新增：通用的 OpenAI 兼容视觉 API 调用函数 VVVVVV
def _call_generic_openai_vision_api(
    image_base64,
    api_key,
    model_name,
    prompt,
    base_url_to_use,
    service_friendly_name,
    start_time,
    reasoning_effort=None,
    image_detail=None
):
    """
    通用的 OpenAI 兼容视觉 API 调用函数。
    """
    logger.info(f"开始调用 {service_friendly_name} 视觉API (通过 OpenAI SDK)，模型: {model_name}, BaseURL: {base_url_to_use}")
    try:
        if not base_url_to_use: # 增加对 base_url_to_use 的检查
            logger.error(f"调用 {service_friendly_name} 失败：未提供 Base URL。")
            return ""

        max_retries = 0 if "自定义" in (service_friendly_name or "") else None
        client = create_openai_client(api_key=api_key, base_url=base_url_to_use, max_retries=max_retries)

        image_url_payload = {"url": f"data:image/png;base64,{image_base64}"}
        if image_detail and image_detail != 'default':
            image_url_payload["detail"] = image_detail

        payload_messages = [ # payload 结构保持一致
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_url_payload}
                ]
            }
        ]
        # (可选) 调试日志，与 Gemini 部分类似
        # debug_payload = { "model": model_name, "messages": [...] }
        # logger.debug(f"{service_friendly_name} API 请求体 (无图): {json.dumps(debug_payload, ensure_ascii=False)}")

        request_kwargs = dict(
            model=model_name,
            messages=payload_messages
        )
        if reasoning_effort and reasoning_effort != 'default':
            request_kwargs["reasoning_effort"] = reasoning_effort

        response = client.chat.completions.create(
            **request_kwargs
        )
        _log_llm_usage(service_friendly_name, model_name, getattr(response, "usage", None))

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
                               # VVVVVV 新增 custom_base_url 参数 VVVVVV
                               custom_base_url=None, reasoning_effort=None, image_detail=None):
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
        provider_lower = provider.lower()
        if provider_lower == 'siliconflow':
            return call_siliconflow_vision_api(image_base64, api_key, model_name, prompt, start_time)
        elif provider_lower == 'volcano':
            # VVVVVV 修改为调用通用函数 VVVVVV
            return _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt,
                                                   "https://ark.cn-beijing.volces.com/api/v3",
                                                   "火山引擎", start_time,
                                                   reasoning_effort=reasoning_effort,
                                                   image_detail=image_detail)
            # ^^^^^^ 结束修改 ^^^^^^
        elif provider_lower == 'gemini':
            # VVVVVV 修改为调用通用函数 VVVVVV
            return _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt,
                                                   "https://generativelanguage.googleapis.com/v1beta/openai/",
                                                   "Gemini Vision", start_time)
            # ^^^^^^ 结束修改 ^^^^^^
        # VVVVVV 新增对自定义服务商的处理 VVVVVV
        elif provider_lower == constants.CUSTOM_AI_VISION_PROVIDER_ID: # 使用后端常量
            if not custom_base_url: # 检查 custom_base_url
                logger.error(f"未提供自定义AI视觉OCR服务的Base URL (provider: {provider})")
                return ""
            return _call_generic_openai_vision_api(image_base64, api_key, model_name, prompt,
                                                   custom_base_url, # <<< 使用传入的自定义 Base URL
                                                   "自定义OpenAI兼容视觉服务", start_time)
        # ^^^^^^ 结束新增 ^^^^^^
        else:
            logger.error(f"不支持的AI视觉OCR服务提供商: {provider}")
            return ""
    except Exception as e: # 捕获 call_xxx 函数可能抛出的其他未预料错误
        logger.error(f"调用AI视觉OCR服务 ({provider}) 时发生顶层异常: {e}", exc_info=True)
        return ""

def call_siliconflow_vision_api(image_base64, api_key, model_name, prompt, start_time):
    """
    调用SiliconFlow的视觉API进行OCR识别
    
    Args:
        image_base64 (str): Base64编码的图片数据
        api_key (str): SiliconFlow API密钥
        model_name (str): 模型名称 (如 'silicon-llava2-34b')
        prompt (str): 提示词
        start_time (float): 计时起点
    
    Returns:
        str: 识别结果文本
    """
    logger.info(f"开始调用SiliconFlow视觉API进行OCR识别，模型: {model_name}")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # 构建请求数据
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
        # 发送请求
        response = requests.post(
            "https://api.siliconflow.cn/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            _log_llm_usage("SiliconFlow", model_name, result.get("usage"))
            
            # 提取识别结果
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                # 计算耗时
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

# 更新 test_ai_vision_ocr 函数签名和内部调用
def test_ai_vision_ocr(image_path, provider, api_key, model_name, prompt=None,
                       # VVVVVV 新增 custom_base_url 参数 VVVVVV
                       custom_base_url=None, reasoning_effort=None, image_detail=None):
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
                custom_base_url=custom_base_url, # <<< 传递自定义 Base URL
                reasoning_effort=reasoning_effort,
                image_detail=image_detail
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
