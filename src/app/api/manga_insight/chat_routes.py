"""
Manga Insight 问答 API

处理基于 RAG 的问答请求。
"""

import logging
from flask import request

from . import manga_insight_bp
from .async_helpers import run_async
from .response_builder import success_response, error_response
from src.core.manga_insight.qa import MangaQA

logger = logging.getLogger("MangaInsight.API.Chat")


@manga_insight_bp.route('/<book_id>/chat', methods=['POST'])
def chat(book_id: str):
    """
    智能问答（单轮对话模式）
    
    Request Body:
        {
            "question": "问题内容",
            "use_parent_child": false,  // 是否使用父子块检索模式
            "use_reasoning": false,  // 是否使用推理检索（分解问题后并行检索）
            "use_reranker": null,  // 是否使用重排序 (null=使用配置默认值，默认启用)
            "top_k": 5,  // 返回的最大结果数
            "threshold": 0.0,  // 相关性阈值（0-1）
            "use_global_context": false  // 是否使用全局摘要模式（适合总结性问题）
        }
    
    两种模式说明：
    - 精确模式（默认）：使用 RAG 检索相关片段，适合具体问题如"第15页发生了什么"
    - 全局模式：使用压缩后的全文摘要，适合总结性问题如"故事的主题是什么"
    """
    try:
        data = request.json or {}
        question = data.get("question", "").strip()
        use_parent_child = data.get("use_parent_child", False)
        use_reasoning = data.get("use_reasoning", False)
        # use_reranker: None 表示使用配置默认值（默认启用）
        use_reranker = data.get("use_reranker")  # 不设默认值，让 qa.py 决定
        top_k = int(data.get("top_k", 5))
        threshold = float(data.get("threshold", 0.0))
        use_global_context = data.get("use_global_context", False)  # 新增：全局摘要模式
        
        # 参数校验
        top_k = max(1, min(20, top_k))  # 限制 1-20
        threshold = max(0.0, min(1.0, threshold))  # 限制 0-1
        
        if not question:
            return error_response("问题不能为空", 400)
        
        qa = MangaQA(book_id)
        response = run_async(qa.answer(
            question, 
            use_parent_child, 
            use_reasoning, 
            use_reranker, 
            top_k, 
            threshold,
            use_global_context  # 传递全局模式参数
        ))
        
        # 格式化引用
        citations = []
        for c in response.citations:
            citations.append({
                "page": c.page,
                "score": c.score,
                "summary": c.summary
            })
        
        return success_response(
            data={
                "answer": response.text,
                "citations": citations,
                "suggested_questions": response.suggested_questions,
                "mode": "global" if use_global_context else "precise"
            }
        )
        
    except Exception as e:
        logger.error(f"问答失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/chat/status', methods=['GET'])
def chat_status(book_id: str):
    """
    获取问答功能状态
    
    返回当前可用的问答模式信息
    """
    try:
        qa = MangaQA(book_id)
        has_global = run_async(qa.has_global_context())
        
        return success_response(data={
            "has_global_context": has_global,
            "modes": {
                "precise": {
                    "available": True,
                    "description": "精确模式：使用 RAG 检索相关片段，适合具体问题"
                },
                "global": {
                    "available": has_global,
                    "description": "全局模式：使用全文摘要，适合总结性问题（需先生成概述）"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"获取问答状态失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/search', methods=['POST'])
def search(book_id: str):
    """
    语义搜索
    
    Request Body:
        {
            "query": "搜索内容",
            "top_k": 10
        }
    """
    try:
        data = request.json or {}
        query = data.get("query", "").strip()
        top_k = data.get("top_k", 10)
        
        if not query:
            return error_response("搜索内容不能为空", 400)
        
        qa = MangaQA(book_id)
        results = run_async(qa.search(query, top_k))
        
        return success_response(data={"results": results})

    except Exception as e:
        logger.error(f"搜索失败: {e}", exc_info=True)
        return error_response(str(e), 500)
