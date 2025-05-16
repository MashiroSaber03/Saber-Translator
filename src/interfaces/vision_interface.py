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

from src.shared import constants
from src.shared.image_helpers import image_to_base64

# 设置日志
logger = logging.getLogger("VisionInterface")

def call_ai_vision_ocr_service(image_pil, provider='siliconflow', api_key=None, model_name=None, prompt=None):
    """
    调用AI视觉OCR服务识别图片中的文字
    
    Args:
        image_pil (PIL.Image): 要识别的图片(PIL图像对象)
        provider (str): 服务提供商 ('siliconflow', 'volcano', 'gemini', 等)
        api_key (str): API密钥
        model_name (str): 模型名称
        prompt (str): 提示词
        
    Returns:
        str: 识别结果文本，识别失败则返回空字符串
    """
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

    # 测量开始时间，用于计算耗时
    start_time = time.time()
    
    # 转换图片为Base64
    try:
        image_base64 = image_to_base64(image_pil)
    except Exception as e:
        logger.error(f"图像转Base64失败: {e}")
        return ""
    
    try:
        # 根据不同的提供商调用不同的API
        if provider.lower() == 'siliconflow':
            return call_siliconflow_vision_api(image_base64, api_key, model_name, prompt, start_time)
        elif provider.lower() == 'volcano':
            return call_volcano_vision_api(image_base64, api_key, model_name, prompt, start_time)
        elif provider.lower() == 'gemini':
            return call_gemini_vision_api(image_base64, api_key, model_name, prompt, start_time)
        # 未来可以添加其他提供商的支持
        # elif provider.lower() == 'openai':
        #    return call_openai_vision_api(image_base64, api_key, model_name, prompt, start_time)
        else:
            logger.error(f"不支持的AI视觉OCR服务提供商: {provider}")
            return ""
    except Exception as e:
        logger.error(f"调用AI视觉OCR服务失败: {e}")
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
            json=payload,
            timeout=60  # 设置超时为60秒
        )
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            
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

def call_volcano_vision_api(image_base64, api_key, model_name, prompt, start_time):
    """
    调用火山引擎的视觉API进行OCR识别
    
    Args:
        image_base64 (str): Base64编码的图片数据
        api_key (str): 火山引擎 API密钥
        model_name (str): 模型名称
        prompt (str): 提示词
        start_time (float): 计时起点
    
    Returns:
        str: 识别结果文本
    """
    logger.info(f"开始调用火山引擎视觉API进行OCR识别，模型: {model_name}")
    
    try:
        # 使用OpenAI客户端，但指定火山引擎的API基础URL
        client = OpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")
        
        # 构建请求负载
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]}
            ],
            timeout=60
        )
        
        # 提取识别结果
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            # 计算耗时
            elapsed_time = time.time() - start_time
            logger.info(f"火山引擎视觉OCR识别成功，耗时: {elapsed_time:.2f}秒")
            logger.info(f"识别结果: {content}")
            return content.strip()
        else:
            logger.error(f"火山引擎响应格式异常")
            return ""
    except Exception as e:
        logger.error(f"调用火山引擎视觉API时发生异常: {e}")
        return ""

def call_gemini_vision_api(image_base64, api_key, model_name, prompt, start_time):
    """
    调用 Google Gemini Vision API (通过OpenAI兼容接口)进行OCR识别
    """
    logger.info(f"开始调用 Gemini Vision API进行OCR识别，模型: {model_name}")
    
    if not model_name:
        logger.error("调用 Gemini Vision API 失败: 未提供模型名称。")
        return ""

    try:
        # 根据教程，使用 OpenAI() client 和指定的 base_url
        client = OpenAI(
            api_key=api_key,
            # 教程截图中的 base_url，确保末尾有 /
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/" 
        )
        
        payload_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ]
        
        # 打印将要发送的请求体（不含图片数据，用于调试）
        debug_payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}} # 隐藏实际图片数据
                    ]
                }
            ]
        }
        logger.debug(f"Gemini Vision API 请求体 (无图): {json.dumps(debug_payload, ensure_ascii=False)}")

        response = client.chat.completions.create(
            model=model_name, # 例如 "gemini-1.5-flash-latest" 或教程中的 "gemini-2.0-flash"
            messages=payload_messages,
            timeout=60 
        )
        
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            logger.info(f"Gemini Vision OCR识别成功，模型: {model_name}，耗时: {elapsed_time:.2f}秒")
            # 仅记录部分结果以避免日志过长
            logger.info(f"Gemini Vision OCR 识别结果 (前100字符): {content[:100]}")
            return content.strip()
        else:
            logger.error(f"Gemini Vision API响应格式异常或无有效结果，模型: {model_name}, 响应: {response}")
            return ""
            
    except Exception as e:
        logger.error(f"调用 Gemini Vision API 时发生异常 (模型: {model_name}): {e}", exc_info=True)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                logger.error(f"Gemini API 错误详情: {error_detail}")
            except json.JSONDecodeError:
                logger.error(f"Gemini API 原始错误响应 (状态码 {e.response.status_code}): {e.response.text}")
        return ""

# 测试函数
def test_ai_vision_ocr(image_path, provider, api_key, model_name, prompt=None):
    """
    测试AI视觉OCR功能
    
    Args:
        image_path (str): 图片文件路径
        provider (str): 服务提供商
        api_key (str): API密钥
        model_name (str): 模型名称
        prompt (str, optional): 提示词，如果不提供则使用默认提示词
    
    Returns:
        bool: 测试是否成功
    """
    try:
        # 加载图片
        with Image.open(image_path) as img:
            # 调用OCR服务
            result = call_ai_vision_ocr_service(
                img, provider, api_key, model_name, prompt
            )
            
            if result:
                logger.info(f"测试成功，识别结果: {result}")
                return True, result
            else:
                logger.error("测试失败，未返回有效识别结果")
                return False, "OCR识别失败，未返回有效结果"
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        return False, f"测试出错: {str(e)}" 