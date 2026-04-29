"""
网页漫画导入 - AI Agent 核心逻辑

使用 LLM + Firecrawl 工具实现智能漫画图片提取
"""

import logging
import json
import re
import time
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace

from openai import OpenAI

from src.shared.ai_providers import (
    CHAT_CAPABILITY,
    WEB_IMPORT_AGENT_CAPABILITY,
    normalize_provider_id,
    provider_supports_capability,
    resolve_provider_base_url_for_capability,
)
from src.shared.openai_helpers import create_openai_client

from .prompts import get_system_prompt
from .firecrawl_tools import FIRECRAWL_TOOLS, execute_firecrawl_tool_sync

logger = logging.getLogger("WebImport.Agent")


@dataclass
class AgentLog:
    """Agent 日志"""
    timestamp: str
    type: str  # 'info' | 'tool_call' | 'tool_result' | 'thinking' | 'error'
    message: str


@dataclass
class ExtractResult:
    """提取结果"""
    success: bool
    comic_title: str = ""
    chapter_title: str = ""
    pages: List[Dict[str, Any]] = field(default_factory=list)
    total_pages: int = 0
    source_url: str = ""
    error: Optional[str] = None


class StreamFallbackNeeded(Exception):
    """流式工具调用无法可靠解析，需要回退到非流式调用。"""
    pass


class MangaScraperAgent:
    """AI 驱动的漫画图片提取 Agent"""
    
    def __init__(self, config: dict):
        """
        初始化 Agent
        
        Args:
            config: 配置字典，包含：
                - firecrawl.apiKey: Firecrawl API Key
                - agent.provider: AI 服务商
                - agent.apiKey: AI API Key
                - agent.customBaseUrl: 自定义 API 地址
                - agent.modelName: 模型名称
                - agent.useStream: 是否流式调用
                - agent.forceJsonOutput: 是否强制 JSON 输出
                - agent.maxRetries: 最大重试次数
                - agent.timeout: 超时时间
                - extraction.prompt: 提取提示词
                - extraction.maxIterations: 最大迭代次数
        """
        self.config = config
        self.firecrawl_api_key = config.get('firecrawl', {}).get('apiKey', '')
        
        agent_config = config.get('agent', {})
        self.provider = normalize_provider_id(agent_config.get('provider', 'openai'))
        self.api_key = agent_config.get('apiKey', '')
        self.base_url = agent_config.get('customBaseUrl', '')
        self.model_name = agent_config.get('modelName', 'gpt-4o-mini')
        self.use_stream = agent_config.get('useStream', False)
        self.force_json = agent_config.get('forceJsonOutput', True)
        self.max_retries = agent_config.get('maxRetries', 3)
        self.timeout = agent_config.get('timeout', 120)
        
        extraction_config = config.get('extraction', {})
        self.custom_prompt = extraction_config.get('prompt', '')
        self.max_iterations = extraction_config.get('maxIterations', 10)

        if self.provider and not provider_supports_capability(self.provider, WEB_IMPORT_AGENT_CAPABILITY):
            raise ValueError(f"不支持的 AI Agent 服务商: {self.provider}")
        
        # 初始化 LLM 客户端
        self.client = self._init_llm_client()
    
    def _init_llm_client(self) -> OpenAI:
        """初始化 LLM 客户端"""
        # 获取 base_url
        base_url = self._get_base_url()

        return create_openai_client(
            api_key=self.api_key,
            base_url=base_url,
            timeout=self.timeout,
        )
    
    def _get_base_url(self) -> Optional[str]:
        """根据服务商获取 API 地址"""
        if self.provider == 'openai':
            return None
        return resolve_provider_base_url_for_capability(self.provider, CHAT_CAPABILITY, self.base_url)
    
    def _create_log(self, log_type: str, message: str) -> AgentLog:
        """创建日志对象"""
        return AgentLog(
            timestamp=datetime.now().strftime('%H:%M:%S'),
            type=log_type,
            message=message
        )
    
    def extract(
        self,
        url: str,
        on_log: Callable[[AgentLog], None] = None
    ) -> ExtractResult:
        """
        执行提取任务 (同步版本)
        
        Args:
            url: 漫画网页 URL
            on_log: 日志回调函数
        
        Returns:
            ExtractResult: 提取结果
        """
        def emit_log(log_type: str, message: str):
            if on_log:
                on_log(self._create_log(log_type, message))
            logger.info(f"[{log_type}] {message}")
        
        emit_log('info', f"开始提取: {url}")
        
        # 构建系统提示词
        system_prompt = get_system_prompt(
            custom_prompt=self.custom_prompt if self.custom_prompt else None,
            force_json=self.force_json
        )
        
        # 初始化消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请提取这个URL的漫画图片: {url}"}
        ]
        
        try:
            for iteration in range(self.max_iterations):
                emit_log('thinking', f"Agent 思考中... (迭代 {iteration + 1}/{self.max_iterations})")
                
                # 调用 LLM
                response = self._call_llm(messages)
                
                # 检查是否有工具调用
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        emit_log('tool_call', f"调用 {tool_name}: {json.dumps(tool_args, ensure_ascii=False)[:200]}...")
                        
                        # 执行工具 (同步)
                        tool_result = execute_firecrawl_tool_sync(
                            tool_name,
                            tool_args,
                            self.firecrawl_api_key,
                            timeout=self.timeout
                        )
                        
                        result_str = json.dumps(tool_result, ensure_ascii=False)
                        emit_log('tool_result', f"返回 {len(result_str)} 字符")
                        
                        # 添加工具调用和结果到消息
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments
                                }
                            }]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str[:50000]  # 限制长度
                        })
                else:
                    # 没有工具调用，解析最终结果
                    content = response.content
                    emit_log('info', "Agent 完成分析，正在解析结果...")
                    
                    result = self._parse_result(content, url)
                    
                    if result.success:
                        emit_log('info', f"提取成功: 《{result.comic_title}》- {result.chapter_title} - 共 {result.total_pages} 页")
                    else:
                        emit_log('error', f"解析结果失败: {result.error}")
                    
                    return result
            
            # 超过最大迭代次数
            emit_log('error', f"超过最大迭代次数 ({self.max_iterations})")
            return ExtractResult(
                success=False,
                source_url=url,
                error=f"超过最大迭代次数 ({self.max_iterations})"
            )
            
        except Exception as e:
            error_msg = str(e)
            emit_log('error', f"提取失败: {error_msg}")
            logger.exception("Agent 提取异常")
            return ExtractResult(
                success=False,
                source_url=url,
                error=error_msg
            )
    
    def _call_llm(self, messages: List[Dict]) -> Any:
        """
        调用 LLM (同步版本)
        
        Args:
            messages: 消息列表
        
        Returns:
            LLM 响应
        """
        max_attempts = max(1, int(self.max_retries or 1))

        for attempt in range(max_attempts):
            try:
                if self.use_stream:
                    try:
                        return self._call_llm_stream(messages)
                    except StreamFallbackNeeded as exc:
                        logger.warning("流式响应无法可靠解析，回退到非流式请求: %s", exc)
                        return self._call_llm_non_stream(messages)

                return self._call_llm_non_stream(messages)
            except Exception as e:
                logger.error(f"LLM 调用失败 (尝试 {attempt + 1}/{max_attempts}): {e}")
                if attempt >= max_attempts - 1 or not self._should_retry_llm_error(e):
                    raise
                time.sleep(2 ** attempt)

    def _call_llm_non_stream(self, messages: List[Dict]) -> Any:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=FIRECRAWL_TOOLS,
            tool_choice="auto",
            temperature=0.1
        )
        return response.choices[0].message

    def _call_llm_stream(self, messages: List[Dict]) -> Any:
        response_stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=FIRECRAWL_TOOLS,
            tool_choice="auto",
            temperature=0.1,
            stream=True,
        )

        content_parts: List[str] = []
        tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
        saw_tool_call_delta = False

        for chunk in response_stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
            if not delta:
                continue

            content = getattr(delta, "content", None)
            if content:
                content_parts.append(content)

            delta_tool_calls = getattr(delta, "tool_calls", None) or []
            for tool_call in delta_tool_calls:
                saw_tool_call_delta = True
                index = int(getattr(tool_call, "index", 0) or 0)
                state = tool_calls_by_index.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    },
                )

                tool_call_id = getattr(tool_call, "id", None)
                if tool_call_id:
                    state["id"] = tool_call_id

                tool_call_type = getattr(tool_call, "type", None)
                if tool_call_type:
                    state["type"] = tool_call_type

                function = getattr(tool_call, "function", None)
                if function:
                    function_name = getattr(function, "name", None)
                    if function_name:
                        state["function"]["name"] = function_name
                    function_arguments = getattr(function, "arguments", None)
                    if function_arguments:
                        state["function"]["arguments"] += function_arguments

        tool_calls = None
        if saw_tool_call_delta:
            tool_calls = self._finalize_stream_tool_calls(tool_calls_by_index)

        return SimpleNamespace(
            content="".join(content_parts),
            tool_calls=tool_calls,
        )

    def _finalize_stream_tool_calls(self, tool_calls_by_index: Dict[int, Dict[str, Any]]) -> List[Any]:
        tool_calls: List[Any] = []
        for index in sorted(tool_calls_by_index):
            state = tool_calls_by_index[index]
            tool_call_id = state.get("id") or ""
            function_name = state.get("function", {}).get("name") or ""
            function_arguments = state.get("function", {}).get("arguments") or ""

            if not tool_call_id or not function_name or not function_arguments:
                raise StreamFallbackNeeded("tool_call 缺少必要字段")

            try:
                json.loads(function_arguments)
            except json.JSONDecodeError as exc:
                raise StreamFallbackNeeded(f"tool_call 参数 JSON 不完整: {exc}") from exc

            tool_calls.append(
                SimpleNamespace(
                    id=tool_call_id,
                    type=state.get("type") or "function",
                    function=SimpleNamespace(
                        name=function_name,
                        arguments=function_arguments,
                    ),
                )
            )
        return tool_calls

    @staticmethod
    def _should_retry_llm_error(error: Exception) -> bool:
        error_text = str(error).lower()

        non_retryable_keywords = (
            "api key",
            "authentication",
            "unauthorized",
            "invalid_api_key",
            "base url",
            "not found",
            "permission",
            "401",
            "forbidden",
            "403",
            "404",
        )
        if any(keyword in error_text for keyword in non_retryable_keywords):
            return False

        retryable_keywords = (
            "timeout",
            "timed out",
            "connection",
            "connect",
            "reset",
            "temporarily unavailable",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
        if any(keyword in error_text for keyword in retryable_keywords):
            return True

        return bool(re.search(r"api 错误\s*(408|429|500|502|503|504)", str(error), re.IGNORECASE))
    
    def _parse_result(self, content: str, source_url: str) -> ExtractResult:
        """
        解析 LLM 返回的结果
        
        Args:
            content: LLM 返回的内容
            source_url: 原始 URL
        
        Returns:
            ExtractResult
        """
        if not content:
            return ExtractResult(
                success=False,
                source_url=source_url,
                error="LLM 返回内容为空"
            )
        
        try:
            # 清理 Markdown 代码块标记
            cleaned = self._clean_json_response(content)
            
            # 解析 JSON
            data = json.loads(cleaned)
            
            # 提取字段
            comic_title = data.get('comic_title', '') or data.get('comicTitle', '') or '未知漫画'
            chapter_title = data.get('chapter_title', '') or data.get('chapterTitle', '') or '未知章节'
            pages = data.get('pages', [])
            total_pages = data.get('total_pages', len(pages)) or data.get('totalPages', len(pages))
            
            # 标准化页面格式
            normalized_pages = []
            for i, page in enumerate(pages):
                normalized_pages.append({
                    'pageNumber': page.get('page_number', i + 1) or page.get('pageNumber', i + 1),
                    'imageUrl': page.get('image_url', '') or page.get('imageUrl', '')
                })
            
            return ExtractResult(
                success=True,
                comic_title=comic_title,
                chapter_title=chapter_title,
                pages=normalized_pages,
                total_pages=total_pages,
                source_url=source_url
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            logger.debug(f"原始内容: {content[:500]}")
            return ExtractResult(
                success=False,
                source_url=source_url,
                error=f"JSON 解析失败: {e}"
            )
        except Exception as e:
            logger.error(f"结果解析失败: {e}")
            return ExtractResult(
                success=False,
                source_url=source_url,
                error=f"结果解析失败: {e}"
            )
    
    def _clean_json_response(self, content: str) -> str:
        """
        清理 JSON 响应，移除 Markdown 代码块标记
        
        Args:
            content: 原始内容
        
        Returns:
            清理后的 JSON 字符串
        """
        content = content.strip()
        
        # 移除 ```json ... ``` 标记
        if content.startswith('```'):
            # 找到第一个换行
            first_newline = content.find('\n')
            if first_newline != -1:
                content = content[first_newline + 1:]
            
            # 移除结尾的 ```
            if content.endswith('```'):
                content = content[:-3]
        
        # 尝试找到 JSON 对象
        start = content.find('{')
        end = content.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            content = content[start:end + 1]
        
        return content.strip()
