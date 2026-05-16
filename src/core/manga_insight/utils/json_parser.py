# src/core/manga_insight/utils/json_parser.py
"""
LLM JSON 解析工具

统一处理 LLM 返回的 JSON 响应，消除重复代码。
"""

import json
import logging
from typing import Any, Dict, Optional

from src.shared.openai_execution import OpenAICompatibleBusinessRetryableError, parse_json_block_from_text

logger = logging.getLogger("MangaInsight.Utils.JsonParser")


def parse_llm_json(response: str, default: Optional[Dict] = None) -> Any:
    """
    解析 LLM 返回的 JSON 响应

    自动处理以下情况：
    - ```json ... ``` 代码块
    - ``` ... ``` 代码块
    - 纯 JSON 文本
    - 前后空白字符

    Args:
        response: LLM 的原始响应文本
        default: 解析失败时返回的默认值，默认为空字典

    Returns:
        解析后的字典，解析失败则返回 default
    """
    if default is None:
        default = {}

    if not response or not isinstance(response, str):
        return default

    try:
        return parse_json_block_from_text(response.strip())
    except OpenAICompatibleBusinessRetryableError as e:
        logger.warning(f"JSON 解析失败: {e}")
        logger.debug(f"原始文本: {str(response)[:200]}...")
        return default
def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    安全的 JSON 加载，失败时返回默认值

    Args:
        text: JSON 文本
        default: 解析失败时的默认值

    Returns:
        解析结果或默认值
    """
    if not text:
        return default

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_json_objects(text: str) -> list:
    """
    从文本中提取所有 JSON 对象

    用于处理 LLM 返回多个 JSON 对象的情况

    Args:
        text: 可能包含多个 JSON 对象的文本

    Returns:
        解析出的 JSON 对象列表
    """
    results = []

    # 尝试匹配所有 {...} 结构
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match)
            results.append(obj)
        except json.JSONDecodeError:
            continue

    return results
