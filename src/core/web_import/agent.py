"""
网页漫画导入 - AI Agent 核心逻辑

使用 LLM + Firecrawl 工具实现智能漫画图片提取
"""

import logging
import json
import re
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI

from .prompts import get_system_prompt, DEFAULT_EXTRACTION_PROMPT
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


class MaxIterationsExceeded(Exception):
    """超过最大迭代次数异常"""
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
        self.provider = agent_config.get('provider', 'openai')
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
        
        # 初始化 LLM 客户端
        self.client = self._init_llm_client()
    
    def _init_llm_client(self) -> OpenAI:
        """初始化 LLM 客户端"""
        # 获取 base_url
        base_url = self._get_base_url()
        
        return OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=self.timeout
        )
    
    def _get_base_url(self) -> Optional[str]:
        """根据服务商获取 API 地址"""
        if self.base_url:
            return self.base_url
        
        provider_urls = {
            'openai': None,  # 使用默认
            'siliconflow': 'https://api.siliconflow.cn/v1',
            'deepseek': 'https://api.deepseek.com/v1',
            'volcano': 'https://ark.cn-beijing.volces.com/api/v3',
            'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai/',
            'custom_openai': self.base_url or None
        }
        
        return provider_urls.get(self.provider)
    
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
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=FIRECRAWL_TOOLS,
                tool_choice="auto",
                temperature=0.1
            )
            return response.choices[0].message
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise
    
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
