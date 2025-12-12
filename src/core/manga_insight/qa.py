"""
Manga Insight 问答系统

基于 RAG 的智能问答。
"""

import logging
import json
import re
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .config_utils import load_insight_config
from .config_models import DEFAULT_QA_SYSTEM_PROMPT, DEFAULT_QUESTION_DECOMPOSE_PROMPT
from .storage import AnalysisStorage
from .vector_store import MangaVectorStore
from .embedding_client import EmbeddingClient, ChatClient
from .reranker_client import RerankerClient
from .query_preprocessor import get_preprocessor

logger = logging.getLogger("MangaInsight.QA")


@dataclass
class Citation:
    """引用"""
    page: int
    score: float
    summary: str = ""
    thumbnail: str = ""


@dataclass
class QAResponse:
    """问答响应"""
    text: str
    citations: List[Citation] = field(default_factory=list)
    suggested_questions: List[str] = field(default_factory=list)


class MangaQA:
    """漫画智能问答"""
    
    def __init__(self, book_id: str):
        self.book_id = book_id
        self.config = load_insight_config()
        self.storage = AnalysisStorage(book_id)
        self.vector_store = MangaVectorStore(book_id)
        
        # 初始化客户端
        self.embedding_client = None
        if self.config.embedding.api_key:
            self.embedding_client = EmbeddingClient(self.config.embedding)
        
        # 对话模型
        if self.config.chat_llm.use_same_as_vlm:
            self.chat_client = ChatClient(self.config.vlm)
        else:
            self.chat_client = ChatClient(self.config.chat_llm)
        
        # 重排序模型（可选）
        self.reranker = None
        if self.config.reranker.enabled and self.config.reranker.api_key:
            self.reranker = RerankerClient(self.config.reranker)
        
        # 查询预处理器
        self.preprocessor = get_preprocessor()
    
    async def answer(
        self,
        question: str,
        use_parent_child: bool = False,
        use_reasoning: bool = False,
        use_reranker: bool = None,  # 改为 None，默认从配置读取
        top_k: int = 5,
        threshold: float = 0.0
    ) -> QAResponse:
        """
        回答用户问题（单轮对话模式）
        
        Args:
            question: 用户问题
            use_parent_child: 是否使用父子块检索模式
            use_reasoning: 是否使用推理检索（分解问题后并行检索）
            use_reranker: 是否使用重排序 (None=使用配置默认值)
            top_k: 返回的最大结果数
            threshold: 相关性阈值（0-1），低于此值的结果将被过滤
        
        Returns:
            QAResponse: 回答结果
        """
        
        # 默认启用 Reranker（如果配置了）
        if use_reranker is None:
            use_reranker = self.config.reranker.enabled and self.reranker is not None
        
        # 查询预处理（照搬 RAGFlow）
        processed = self.preprocessor.preprocess(question)
        logger.info(f"问答请求: '{question}' -> 关键词: {processed['keywords']}")
        
        # 1. 检索相关内容（使用混合检索）
        if use_reasoning:
            relevant_pages = await self._reasoning_retrieve(question, processed, top_k)
        else:
            relevant_pages = await self._hybrid_retrieve(question, processed, top_k)
        
        logger.info(f"检索后结果数量: {len(relevant_pages)}")
        
        # 1.5 阈值过滤（使用 hybrid_score 或 score）
        if threshold > 0:
            before_filter = len(relevant_pages)
            relevant_pages = [
                p for p in relevant_pages 
                if p.get("hybrid_score", p.get("score", 0)) >= threshold
            ]
            logger.info(f"阈值过滤: {before_filter} -> {len(relevant_pages)} (阈值={threshold})")
        
        # 2. 重排序（默认启用）
        if use_reranker and self.reranker and relevant_pages:
            before_rerank = len(relevant_pages)
            try:
                # 使用清理后的查询进行重排序
                rerank_query = processed["clean_query"] or question
                relevant_pages = await self.reranker.rerank(
                    query=rerank_query,
                    documents=relevant_pages,
                    top_k=max(top_k, self.config.reranker.top_k)
                )
                logger.info(f"重排序后结果数量: {len(relevant_pages)}")
            except Exception as e:
                logger.error(f"重排序失败: {e}")
        
        # 3. 限制结果数量
        relevant_pages = relevant_pages[:top_k]
        logger.info(f"最终结果数量: {len(relevant_pages)}")
        
        # 4. 检查是否有检索结果
        if not relevant_pages:
            return QAResponse(
                text="抱歉，没有找到与您问题相关的内容。请尝试换一种方式提问，或降低相关性阈值。",
                citations=[],
                suggested_questions=["这本漫画讲了什么故事？", "主角是谁？"]
            )
        
        # 5. 构建上下文
        context = await self._build_context(relevant_pages, use_parent_child)
        
        # 6. 生成回答
        prompt = self._build_qa_prompt(question, context, use_parent_child)
        # 优先使用用户自定义提示词，否则使用默认
        system = self.config.prompts.qa_response if self.config.prompts.qa_response else DEFAULT_QA_SYSTEM_PROMPT
        
        answer_text = await self.chat_client.generate(prompt, system)
        
        # 7. 提取引用（从实际使用的结果中提取）
        citations = self._extract_citations(relevant_pages)
        
        # 8. 生成推荐问题
        follow_ups = await self._generate_follow_up_questions(question, answer_text)
        
        return QAResponse(
            text=answer_text,
            citations=citations,
            suggested_questions=follow_ups
        )
    
    async def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        语义搜索
        
        Args:
            query: 搜索查询
            top_k: 返回数量
        
        Returns:
            List[Dict]: 搜索结果
        """
        if not self.embedding_client or not self.vector_store.is_available():
            return []
        
        query_embedding = await self.embedding_client.embed(query)
        
        # 搜索页面摘要（作为RAG检索的最小单位）
        page_results = await self.vector_store.search_pages(query_embedding, top_k=top_k)
        
        results = []
        for item in page_results:
            results.append({
                "type": "page",
                "page_num": item.get("metadata", {}).get("page_num"),
                "content": item.get("document", ""),
                "score": item.get("score", 0)
            })
        
        return results
    
    async def _hybrid_retrieve(
        self, 
        question: str, 
        processed: Dict,
        top_k: int = 5
    ) -> List[Dict]:
        """
        混合检索 - 照搬 RAGFlow 的双路检索 + 分数融合
        
        Args:
            question: 原始问题
            processed: 预处理结果
            top_k: 返回数量
            
        Returns:
            List[Dict]: 检索结果
        """
        # 检查向量存储可用性
        if not self.embedding_client:
            logger.warning("Embedding 客户端不可用，使用回退检索")
            return await self._fallback_retrieval()
        
        if not self.vector_store.is_available():
            logger.warning("向量存储不可用，使用回退检索")
            return await self._fallback_retrieval()
        
        # 检查向量存储统计
        stats = self.vector_store.get_stats()
        logger.debug(f"向量存储统计: {stats}")
        
        pages_count = stats.get("pages_count", 0)
        events_count = stats.get("events_count", 0)
        
        if pages_count == 0 and events_count == 0:
            logger.warning("向量存储中没有数据，使用回退检索")
            return await self._fallback_retrieval()
        
        try:
            # 1. 获取查询向量 (使用清理后的查询)
            query_text = processed["clean_query"] or question
            query_embedding = await self.embedding_client.embed(query_text)
            logger.debug(f"问题向量维度: {len(query_embedding)}")
            
            # 2. 多检索一些供后续筛选 (参考 RAGFlow)
            initial_top_k = max(20, top_k * 4)
            
            # 3. 并行检索: 页面 + 事件 (使用 asyncio.gather 同时执行)
            search_tasks = []
            task_types = []  # 记录任务类型，用于结果分配
            
            if pages_count > 0:
                search_tasks.append(self.vector_store.search_pages(query_embedding, top_k=initial_top_k))
                task_types.append("pages")
            
            if events_count > 0:
                search_tasks.append(self.vector_store.search_events(query_embedding, top_k=initial_top_k))
                task_types.append("events")
            
            # 并行执行所有检索任务
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # 解析结果
            pages_results = []
            events_results = []
            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    logger.warning(f"检索任务失败: {task_types[i]} - {result}")
                    continue
                if task_types[i] == "pages":
                    pages_results = result
                elif task_types[i] == "events":
                    events_results = result
            
            # 4. 合并结果并去重
            all_results = []
            seen_ids = set()
            
            for result in pages_results + events_results:
                result_id = result.get("id", "")
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    all_results.append(result)
            
            # 5. 计算混合分数 (照搬 RAGFlow hybrid_similarity)
            keywords = processed["keywords"]
            tokens_with_weight = processed["tokens_with_weight"]
            
            for result in all_results:
                vector_score = result.get("score", 0)
                content = result.get("document", "")
                
                # 关键词匹配分数 (照搬 RAGFlow)
                keyword_score = self.preprocessor.compute_keyword_score(
                    content, keywords, tokens_with_weight
                )
                
                # 混合分数: 向量 70% + 关键词 30% (参考 RAGFlow 0.7:0.3)
                hybrid_score = vector_score * 0.7 + keyword_score * 0.3
                
                result["vector_score"] = round(vector_score, 4)
                result["keyword_score"] = round(keyword_score, 4)
                result["hybrid_score"] = round(hybrid_score, 4)
            
            # 6. 按混合分数排序
            all_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
            
            logger.info(
                f"混合检索: 页面={len(pages_results)}, 事件={len(events_results)}, "
                f"合并后={len(all_results)}"
            )
            
            return all_results[:initial_top_k]  # 返回较多结果供 reranker 筛选
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}", exc_info=True)
            return await self._fallback_retrieval()
    
    async def _retrieve_relevant_content(self, question: str, top_k: int = 5) -> List[Dict]:
        """检索相关内容（兼容旧接口）"""
        processed = self.preprocessor.preprocess(question)
        return await self._hybrid_retrieve(question, processed, top_k)
    
    async def _decompose_question(self, question: str) -> List[str]:
        """用 LLM 将复杂问题分解为子问题"""
        # 优先使用用户自定义提示词，否则使用默认
        base_prompt = self.config.prompts.question_decompose if self.config.prompts.question_decompose else DEFAULT_QUESTION_DECOMPOSE_PROMPT
        prompt = base_prompt.format(question=question)
        
        try:
            response = await self.chat_client.generate(prompt, temperature=0.3)
            
            # 清理可能的 markdown 代码块
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = re.sub(r'^```(?:json)?\s*', '', clean_response)
                clean_response = re.sub(r'\s*```$', '', clean_response)
            
            # 尝试找到 JSON 部分
            json_match = re.search(r'\{[^{}]*"sub_questions"\s*:\s*\[[^\]]*\][^{}]*\}', clean_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                sub_questions = data.get("sub_questions", [])
                # 过滤有效的子问题
                sub_questions = [q.strip() for q in sub_questions if isinstance(q, str) and len(q.strip()) > 5]
                if sub_questions:
                    return sub_questions[:4]
            
            # JSON 解析失败，返回空列表
            logger.warning(f"问题分解 JSON 解析失败，使用原始预处理结果")
            return []
        except Exception as e:
            logger.warning(f"问题分解失败: {e}")
            return []  # 统一返回空列表，让调用方使用 processed
    
    async def _reasoning_retrieve(
        self, 
        question: str, 
        processed: Dict,
        top_k: int = 5
    ) -> List[Dict]:
        """推理检索：分解问题后并行检索，整合结果"""
        if not self.embedding_client or not self.vector_store.is_available():
            return await self._fallback_retrieval()
        
        # 1. 分解问题
        sub_questions = await self._decompose_question(question)
        
        # 如果分解失败或只有一个子问题，使用原始预处理结果
        if not sub_questions or len(sub_questions) <= 1:
            logger.info(f"推理检索: 问题无需分解，使用混合检索")
            return await self._hybrid_retrieve(question, processed, top_k)
        
        logger.info(f"推理检索: 问题分解为 {len(sub_questions)} 个子问题: {sub_questions}")
        
        # 2. 并行检索每个子问题
        per_question_k = max(5, top_k)
        
        # 先对所有子问题进行预处理（CPU操作，很快）
        sub_processed_list = [self.preprocessor.preprocess(sub_q) for sub_q in sub_questions]
        
        # 创建并行检索任务
        async def retrieve_sub_question(sub_q: str, sub_processed: Dict) -> tuple:
            """单个子问题的检索任务，返回 (子问题, 结果列表)"""
            try:
                results = await self._hybrid_retrieve(sub_q, sub_processed, per_question_k)
                return (sub_q, results)
            except Exception as e:
                logger.warning(f"子问题检索失败: {sub_q} - {e}")
                return (sub_q, [])
        
        # 并行执行所有子问题检索
        tasks = [
            retrieve_sub_question(sub_q, sub_processed)
            for sub_q, sub_processed in zip(sub_questions, sub_processed_list)
        ]
        results_tuples = await asyncio.gather(*tasks)
        
        # 3. 整合并去重结果
        all_results = []
        seen_ids = set()
        
        for sub_q, results in results_tuples:
            for r in results:
                r_id = r.get("id", "")
                if r_id and r_id not in seen_ids:
                    seen_ids.add(r_id)
                    r["sub_question"] = sub_q  # 标记来源子问题
                    all_results.append(r)
        
        # 4. 按混合分数排序
        all_results.sort(key=lambda x: x.get("hybrid_score", x.get("score", 0)), reverse=True)
        
        logger.info(f"推理检索: 并行检索 {len(sub_questions)} 个子问题，共获得 {len(all_results)} 个结果")
        return all_results
    
    async def _fallback_retrieval(self) -> List[Dict]:
        """降级检索（当向量搜索不可用时）"""
        page_nums = await self.storage.list_pages()
        results = []
        
        for page_num in page_nums[:10]:  # 最多返回10页
            analysis = await self.storage.load_page_analysis(page_num)
            if analysis:
                results.append({
                    "id": f"page_{page_num}",
                    "document": analysis.get("page_summary", ""),
                    "metadata": {"page_num": page_num, "type": "page"},
                    "score": 1.0
                })
        
        return results
    
    async def _build_context(
        self,
        relevant_results: List[Dict],
        use_parent_child: bool = False
    ) -> str:
        """构建问答上下文
        
        Args:
            relevant_results: 检索到的相关结果（页面或事件）
            use_parent_child: 是否使用父子块模式
                - False: 只返回匹配的内容
                - True: 返回匹配内容所属批次的完整内容（父块）
        """
        if not relevant_results:
            return "未找到相关内容。"
        
        context_parts = []
        
        if use_parent_child:
            # 父子块模式：获取完整批次内容
            processed_batches = set()
            
            for result in relevant_results:
                metadata = result.get("metadata", {})
                result_type = metadata.get("type", "page")
                
                # 确定批次范围
                if result_type == "event":
                    # 事件 -> 从元数据获取批次信息
                    start_page = metadata.get("start_page", 0)
                    end_page = metadata.get("end_page", 0)
                    batch_key = f"{start_page}_{end_page}"
                else:
                    # 页面 -> 查找所属批次
                    page_num = self._extract_page_num(result)
                    if not page_num:
                        continue
                    batch = await self.storage.find_batch_for_page(page_num)
                    if batch:
                        page_range = batch.get("page_range", {})
                        start_page = page_range.get("start", 0)
                        end_page = page_range.get("end", 0)
                        batch_key = f"{start_page}_{end_page}"
                    else:
                        # 没有批次信息，添加单页
                        await self._add_single_page_context(page_num, context_parts)
                        continue
                
                # 避免重复添加同一批次
                if batch_key in processed_batches or start_page == 0:
                    continue
                processed_batches.add(batch_key)
                
                # 加载批次数据
                batch_data = await self.storage.load_batch_analysis(start_page, end_page)
                if not batch_data:
                    continue
                
                # 添加批次总结（父块核心）
                batch_summary = batch_data.get("batch_summary", "")
                if batch_summary:
                    context_parts.append(
                        f"【第{start_page}-{end_page}页 概述】\n{batch_summary}"
                    )
                
                # 添加关键事件
                key_events = batch_data.get("key_events", [])
                if key_events:
                    events_text = "、".join(str(e) for e in key_events[:5] if e)
                    if events_text:
                        context_parts.append(f"关键事件: {events_text}")
                
                # 添加批次内各页摘要
                for page_info in batch_data.get("pages", []):
                    p_num = page_info.get("page_number")
                    p_summary = page_info.get("page_summary", "")
                    if p_num and p_summary:
                        context_parts.append(f"  第{p_num}页: {p_summary}")
        else:
            # 普通模式：直接返回匹配内容
            seen_pages = set()
            for result in relevant_results[:10]:  # 限制数量
                metadata = result.get("metadata", {})
                result_type = metadata.get("type", "page")
                content = result.get("document", "")
                
                if result_type == "event":
                    # 事件类型
                    start_page = metadata.get("start_page", "?")
                    end_page = metadata.get("end_page", "?")
                    if content:
                        context_parts.append(f"【第{start_page}-{end_page}页 事件】{content}")
                else:
                    # 页面类型
                    page_num = self._extract_page_num(result)
                    if page_num and page_num not in seen_pages:
                        seen_pages.add(page_num)
                        if content:
                            context_parts.append(f"【第{page_num}页】{content}")
                        else:
                            await self._add_single_page_context(page_num, context_parts)
        
        return "\n\n".join(context_parts) if context_parts else "未找到相关内容。"
    
    def _extract_page_num(self, page_data: Dict) -> Optional[int]:
        """从检索结果中提取页码"""
        page_num = page_data.get("metadata", {}).get("page_num")
        if not page_num:
            doc_id = page_data.get("id", "")
            if doc_id.startswith("page_"):
                try:
                    page_num = int(doc_id.replace("page_", ""))
                except:
                    return None
        return page_num
    
    async def _add_single_page_context(self, page_num: int, context_parts: List[str]):
        """添加单页上下文"""
        page_analysis = await self.storage.load_page_analysis(page_num)
        if page_analysis:
            summary = page_analysis.get("page_summary", "")
            context_parts.append(f"【第{page_num}页】{summary}")
    
    def _build_qa_prompt(
        self,
        question: str,
        context: str,
        use_parent_child: bool = False
    ) -> str:
        """构建问答提示词（单轮对话模式）"""
        prompt_parts = []
        
        # 上下文
        if context:
            prompt_parts.append(f"【漫画内容】\n{context}\n")
            # 父子块模式提示
            if use_parent_child:
                prompt_parts.append("（注：以上内容按批次组织，包含批次概述和各页详情，请综合分析后回答）\n")
        
        # 当前问题
        prompt_parts.append(f"【用户问题】\n{question}")
        prompt_parts.append("\n请基于以上漫画内容回答问题，引用具体页码。如果内容中没有相关信息，请诚实说明。")
        
        return "\n".join(prompt_parts)
    
    def _extract_citations(self, relevant_results: List[Dict]) -> List[Citation]:
        """提取引用（支持页面和事件）"""
        citations = []
        seen_pages = set()
        
        for result in relevant_results[:8]:  # 最多8个引用
            metadata = result.get("metadata", {})
            result_type = metadata.get("type", "page")
            score = result.get("hybrid_score", result.get("score", 0))
            content = result.get("document", "")
            
            if result_type == "event":
                # 事件类型：使用起始页码
                page_num = metadata.get("start_page", 0)
                if page_num and page_num not in seen_pages:
                    seen_pages.add(page_num)
                    citations.append(Citation(
                        page=page_num,
                        score=score,
                        summary=f"[事件] {content[:80]}" if content else ""
                    ))
            else:
                # 页面类型
                page_num = metadata.get("page_num")
                if not page_num:
                    doc_id = result.get("id", "")
                    if doc_id.startswith("page_"):
                        try:
                            page_num = int(doc_id.replace("page_", ""))
                        except:
                            continue
                
                if page_num and page_num not in seen_pages:
                    seen_pages.add(page_num)
                    citations.append(Citation(
                        page=page_num,
                        score=score,
                        summary=content[:100] if content else ""
                    ))
        
        # 按页码排序
        citations.sort(key=lambda c: c.page)
        return citations[:5]  # 最终返回5个
    
    async def _generate_follow_up_questions(
        self,
        question: str,
        answer: str
    ) -> List[str]:
        """生成推荐问题"""
        # 简单的推荐问题
        suggestions = [
            "这个角色后来怎么样了？",
            "还有哪些重要角色？",
            "故事的主要冲突是什么？"
        ]
        
        return suggestions[:3]
