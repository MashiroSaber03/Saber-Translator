# src/core/manga_insight/utils/text_formatter.py
"""
文本格式化工具

统一处理分析结果的格式化，消除重复代码。
"""

from typing import Dict, List, Optional


def format_page_range(start: int, end: int) -> str:
    """
    格式化页码范围

    Args:
        start: 起始页码
        end: 结束页码

    Returns:
        格式化的页码范围字符串
    """
    if start == end:
        return f"第{start}页"
    return f"第{start}-{end}页"


def truncate_text(text: str, max_length: int = 600, suffix: str = "...") -> str:
    """
    截断文本到指定长度

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后的后缀

    Returns:
        截断后的文本
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def format_batch_results(results: List[Dict], max_batches: int = 5) -> str:
    """
    格式化批量分析结果列表

    Args:
        results: 批量分析结果列表
        max_batches: 最大包含的批次数

    Returns:
        格式化的文本
    """
    if not results:
        return ""

    parts = []

    for idx, result in enumerate(results[:max_batches]):
        formatted = format_single_batch_result(result, idx + 1, len(results))
        if formatted:
            parts.append(formatted)

    return "\n\n".join(parts)


def format_single_batch_result(result: Dict, batch_num: int = 1, total_batches: int = 1) -> str:
    """
    格式化单个批次分析结果

    Args:
        result: 批次分析结果
        batch_num: 当前批次号
        total_batches: 总批次数

    Returns:
        格式化的文本
    """
    if not result or result.get("parse_error"):
        return ""

    parts = []

    # 页面范围
    page_range = result.get("page_range", {})
    if page_range:
        start = page_range.get("start", "?")
        end = page_range.get("end", "?")
        if total_batches > 1:
            parts.append(f"【前第{total_batches - batch_num + 1}批：第{start}-{end}页】")
        else:
            parts.append(f"【第{start}-{end}页】")

    # 批次摘要
    batch_summary = result.get("batch_summary", "")
    if batch_summary:
        batch_summary = truncate_text(batch_summary, 600)
        parts.append(f"剧情: {batch_summary}")

    # 关键事件
    key_events = result.get("key_events", [])
    if key_events:
        valid_events = [str(e) for e in key_events[:3] if e]
        if valid_events:
            parts.append(f"事件: {'; '.join(valid_events)}")

    return "\n".join(parts)


def format_segment_summary(segment: Dict) -> str:
    """
    格式化段落摘要

    Args:
        segment: 段落摘要数据

    Returns:
        格式化的文本
    """
    if not segment:
        return ""

    parts = []

    page_range = segment.get("page_range", {})
    if page_range:
        range_text = format_page_range(
            page_range.get("start", 0),
            page_range.get("end", 0)
        )
        parts.append(f"【{range_text}】")

    summary = segment.get("summary", "")
    if summary:
        parts.append(summary)

    themes = segment.get("themes", [])
    if themes:
        parts.append(f"主题: {', '.join(themes[:3])}")

    return "\n".join(parts)


def format_chapter_summaries(summaries: List[str], max_items: int = 5) -> str:
    """
    格式化章节摘要列表用于 LLM 输入

    Args:
        summaries: 摘要列表
        max_items: 最大项数

    Returns:
        格式化的文本
    """
    if not summaries:
        return ""

    return "\n".join(summaries[:max_items])


def build_context_text(previous_results: List[Dict]) -> str:
    """
    构建上下文文本（用于批量分析的上文参考）

    Args:
        previous_results: 前几批的分析结果

    Returns:
        格式化的上下文文本
    """
    if not previous_results:
        return ""

    return format_batch_results(previous_results, max_batches=len(previous_results))
