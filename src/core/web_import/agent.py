"""
网页漫画导入 - AI Agent 核心逻辑

使用 LLM + Firecrawl 工具实现智能漫画图片提取
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlparse

from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.shared.ai_providers import (
    WEB_IMPORT_AGENT_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_supports_capability,
    resolve_provider_base_url_for_capability,
)
from src.shared.openai_helpers import create_openai_client
from src.shared.memory_errors import is_memory_allocation_error
from src.shared.ai_transport import RETRYABLE_STATUS_CODES

from .firecrawl_tools import FIRECRAWL_TOOLS, execute_firecrawl_tool_sync

logger = logging.getLogger("WebImport.Agent")

JSON_OUTPUT_SUFFIX = """

IMPORTANT: You must respond with valid JSON format only. Do not include any
markdown code block markers. Output the raw JSON object."""


@dataclass(slots=True)
class AgentLog:
    """Agent 日志"""
    timestamp: str
    type: str  # 'info' | 'tool_call' | 'tool_result' | 'thinking' | 'error'
    message: str


@dataclass(slots=True)
class ExtractResult:
    """提取结果"""
    success: bool
    comic_title: str = ""
    chapter_title: str = ""
    pages: list[dict[str, Any]] = field(default_factory=list)
    total_pages: int = 0
    source_url: str = ""
    error: str | None = None


class StreamFallbackNeeded(Exception):
    """流式工具调用无法可靠解析，需要回退到非流式调用。"""


class WebImportAgentControlRequested(RuntimeError):
    """The durable job requested pause or cancellation at a safe point."""


class MangaScraperAgent:
    """AI 驱动的漫画图片提取 Agent"""

    def __init__(
        self,
        *,
        firecrawl_api_key: str,
        provider: str,
        api_key: str,
        base_url: str,
        model_name: str,
        use_stream: bool,
        force_json: bool,
        max_retries: int,
        timeout: float,
        prompt: str,
        max_iterations: int,
        bypass_proxy: bool,
    ) -> None:
        if not isinstance(firecrawl_api_key, str) or not firecrawl_api_key.strip():
            raise ValueError("Firecrawl API Key 不能为空")
        if not isinstance(api_key, str):
            raise TypeError("AI Agent API Key 必须是字符串")
        if not isinstance(base_url, str):
            raise TypeError("AI Agent Base URL 必须是字符串")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("AI Agent 模型名称不能为空")
        if not isinstance(use_stream, bool) or not isinstance(force_json, bool):
            raise TypeError("AI Agent 流式与 JSON 开关必须是布尔值")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("AI Agent 重试次数必须是非负整数")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("AI Agent 超时时间必须是正数")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("AI Agent 提取提示词不能为空")
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("AI Agent 最大迭代次数必须是正整数")
        if not isinstance(bypass_proxy, bool):
            raise TypeError("AI Agent 代理开关必须是布尔值")
        self.firecrawl_api_key = firecrawl_api_key
        self.provider = normalize_provider_id(provider)
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.use_stream = use_stream
        self.force_json = force_json
        self.max_retries = max_retries
        self.timeout = timeout
        self.prompt = prompt
        self.max_iterations = max_iterations
        self.bypass_proxy = bypass_proxy

        if not provider_supports_capability(self.provider, WEB_IMPORT_AGENT_CAPABILITY):
            raise ValueError(f"不支持的 AI Agent 服务商: {self.provider}")
        manifest = get_provider_manifest(self.provider)
        if manifest.requires_api_key and not self.api_key.strip():
            raise ValueError(f"{manifest.display_name} 需要 API Key")
        if manifest.requires_base_url and not self.base_url.strip():
            raise ValueError(f"{manifest.display_name} 需要 Base URL")
        
        self.client = create_openai_client(
            api_key=self.api_key,
            base_url=resolve_provider_base_url_for_capability(
                self.provider,
                WEB_IMPORT_AGENT_CAPABILITY,
                self.base_url,
            ),
            timeout=self.timeout,
            bypass_proxy=self.bypass_proxy,
        )

    def close(self) -> None:
        self.client.close()
    
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
        on_log: Callable[[AgentLog], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> ExtractResult:
        """
        执行提取任务 (同步版本)
        
        Args:
            url: 漫画网页 URL
            on_log: 日志回调函数
        
        Returns:
            ExtractResult: 提取结果
        """
        if not isinstance(url, str) or not url.strip():
            raise ValueError("漫画网页 URL 不能为空")
        source_url = url.strip()
        parsed_source_url = urlparse(source_url)
        if (
            parsed_source_url.scheme not in {"http", "https"}
            or not parsed_source_url.netloc
        ):
            raise ValueError("漫画网页 URL 必须是绝对 HTTP(S) 地址")

        def emit_log(log_type: str, message: str):
            if on_log:
                on_log(self._create_log(log_type, message))
            logger.info(f"[{log_type}] {message}")
        
        emit_log('info', f"开始提取: {source_url}")
        
        self._check_control(should_stop)
        system_prompt = self.prompt
        if self.force_json:
            system_prompt += JSON_OUTPUT_SUFFIX
        
        # 初始化消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请提取这个URL的漫画图片: {source_url}"}
        ]
        
        for iteration in range(self.max_iterations):
            self._check_control(should_stop)
            emit_log('thinking', f"Agent 思考中... (迭代 {iteration + 1}/{self.max_iterations})")

            response = self._call_llm(messages, should_stop=should_stop)
            self._check_control(should_stop)

            if response.tool_calls:
                assistant_tool_calls: list[dict[str, Any]] = []
                tool_messages: list[dict[str, str]] = []
                for tool_call in response.tool_calls:
                    self._check_control(should_stop)
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError("AI Agent 工具参数不是合法 JSON") from exc
                    if not isinstance(tool_args, dict):
                        raise ValueError("AI Agent 工具参数必须是 JSON 对象")

                    emit_log('tool_call', f"调用 {tool_name}: {json.dumps(tool_args, ensure_ascii=False)[:200]}...")

                    tool_result = execute_firecrawl_tool_sync(
                        tool_name,
                        tool_args,
                        self.firecrawl_api_key,
                        timeout=self.timeout,
                        bypass_proxy=self.bypass_proxy,
                    )
                    self._check_control(should_stop)

                    result_str = json.dumps(tool_result, ensure_ascii=False)
                    emit_log('tool_result', f"返回 {len(result_str)} 字符")

                    assistant_tool_calls.append({
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_call.function.arguments
                            }
                    })
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": assistant_tool_calls,
                })
                messages.extend(tool_messages)
            else:
                self._check_control(should_stop)
                content = response.content
                emit_log('info', "Agent 完成分析，正在解析结果...")

                result = self._parse_result(content, source_url)

                if result.success:
                    emit_log('info', f"提取成功: 《{result.comic_title}》- {result.chapter_title} - 共 {result.total_pages} 页")
                else:
                    emit_log('error', f"解析结果失败: {result.error}")

                return result

        emit_log('error', f"超过最大迭代次数 ({self.max_iterations})")
        return ExtractResult(
            success=False,
            source_url=source_url,
            error=f"超过最大迭代次数 ({self.max_iterations})"
        )

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> Any:
        """
        调用 LLM (同步版本)
        
        Args:
            messages: 消息列表
        
        Returns:
            LLM 响应
        """
        max_attempts = self.max_retries + 1

        for attempt in range(max_attempts):
            self._check_control(should_stop)
            try:
                if self.use_stream:
                    try:
                        response = self._call_llm_stream(
                            messages,
                            should_stop=should_stop,
                        )
                    except StreamFallbackNeeded as exc:
                        logger.warning("流式响应无法可靠解析，回退到非流式请求: %s", exc)
                        self._check_control(should_stop)
                        response = self._call_llm_non_stream(messages)
                else:
                    response = self._call_llm_non_stream(messages)

                self._check_control(should_stop)
                return response
            except WebImportAgentControlRequested:
                raise
            except Exception as e:
                if is_memory_allocation_error(e):
                    raise
                logger.error(f"LLM 调用失败 (尝试 {attempt + 1}/{max_attempts}): {e}")
                if attempt >= max_attempts - 1 or not self._should_retry_llm_error(e):
                    raise
                self._wait_before_retry(2 ** attempt, should_stop)

    def _call_llm_non_stream(self, messages: list[dict[str, Any]]) -> Any:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=FIRECRAWL_TOOLS,
            tool_choice="auto",
            temperature=0.1
        )
        return response.choices[0].message

    def _call_llm_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> Any:
        response_stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=FIRECRAWL_TOOLS,
            tool_choice="auto",
            temperature=0.1,
            stream=True,
        )

        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        saw_tool_call_delta = False

        for chunk in response_stream:
            self._check_control(should_stop)
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

    @staticmethod
    def _check_control(
        should_stop: Callable[[], bool] | None,
    ) -> None:
        if should_stop is not None and should_stop():
            raise WebImportAgentControlRequested(
                "web import job control requested"
            )
    @classmethod
    def _wait_before_retry(
        cls,
        delay_seconds: float,
        should_stop: Callable[[], bool] | None,
    ) -> None:
        deadline = time.monotonic() + delay_seconds
        while True:
            cls._check_control(should_stop)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.25, remaining))

    def _finalize_stream_tool_calls(
        self,
        tool_calls_by_index: dict[int, dict[str, Any]],
    ) -> list[Any]:
        tool_calls: list[Any] = []
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
        if isinstance(error, (APIConnectionError, APITimeoutError)):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in RETRYABLE_STATUS_CODES
        return False
    
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
            data = json.loads(content.strip())

            required_fields = {
                'comic_title',
                'chapter_title',
                'pages',
                'total_pages',
            }
            if not isinstance(data, dict) or set(data) != required_fields:
                raise ValueError(
                    'Agent 结果必须且只能包含 comic_title、chapter_title、'
                    'pages、total_pages'
                )
            comic_title = data['comic_title']
            chapter_title = data['chapter_title']
            pages = data['pages']
            total_pages = data['total_pages']
            if (
                not isinstance(comic_title, str)
                or not isinstance(chapter_title, str)
                or not isinstance(pages, list)
                or not pages
                or isinstance(total_pages, bool)
                or not isinstance(total_pages, int)
                or total_pages != len(pages)
            ):
                raise ValueError('Agent 结果字段类型或 total_pages 不正确')
            
            # 标准化页面格式
            normalized_pages = []
            for i, page in enumerate(pages):
                if (
                    not isinstance(page, dict)
                    or set(page) != {'page_number', 'image_url'}
                    or isinstance(page['page_number'], bool)
                    or not isinstance(page['page_number'], int)
                    or page['page_number'] < 1
                    or page['page_number'] != i + 1
                    or not isinstance(page['image_url'], str)
                    or not page['image_url'].strip()
                ):
                    raise ValueError(f'Agent 第 {i + 1} 个页面字段不正确')
                parsed_url = urlparse(page['image_url'].strip())
                if parsed_url.scheme not in {'http', 'https'} or not parsed_url.netloc:
                    raise ValueError(f'Agent 第 {i + 1} 个页面 URL 不正确')
                normalized_pages.append({
                    'pageNumber': page['page_number'],
                    'imageUrl': page['image_url'].strip()
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
            if is_memory_allocation_error(e):
                raise
            logger.error(f"结果解析失败: {e}")
            return ExtractResult(
                success=False,
                source_url=source_url,
                error=f"结果解析失败: {e}"
            )
