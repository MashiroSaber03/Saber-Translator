# src/core/manga_insight/utils/json_parser.py
"""
LLM JSON 解析工具

统一处理 LLM 返回的 JSON 响应，消除重复代码。
"""

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
