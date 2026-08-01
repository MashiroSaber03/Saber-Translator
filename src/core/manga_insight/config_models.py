"""
Manga Insight 配置数据模型

使用 dataclass 定义配置对象，支持多种 VLM/Embedding 服务商。
通过 SerializableMixin 自动提供 to_dict/from_dict 方法。
"""

from dataclasses import dataclass, field
from typing import Optional

from .config.serialization import SerializableMixin
from src.shared.openai_options import (
    OpenAICompatibleExecutionOptions,
    OpenAICompatibleOptions,
    OpenAICompatibleRequestOptions,
)


@dataclass
class VLMConfig(SerializableMixin):
    """VLM 多模态模型配置"""
    provider: str = "gemini"
    api_key: str = ""
    model: str = "gemini-2.0-flash"
    base_url: Optional[str] = None
    openai_options: OpenAICompatibleOptions = field(default_factory=lambda: OpenAICompatibleOptions(
        request=OpenAICompatibleRequestOptions(
            force_json_output=False,
            temperature=0.3,
        ),
        execution=OpenAICompatibleExecutionOptions(
            use_stream=True,
            rpm_limit=0,
            transport_retries=10,
            business_retries=10,
        ),
    ))
    image_max_size: int = 1280  # 图片最大边长（像素），0 表示不压缩


@dataclass
class ChatLLMConfig(SerializableMixin):
    """对话模型配置"""
    use_same_as_vlm: bool = False
    provider: str = "gemini"
    api_key: str = ""
    model: str = "gemini-2.0-flash"
    base_url: Optional[str] = None
    openai_options: OpenAICompatibleOptions = field(default_factory=lambda: OpenAICompatibleOptions(
        request=OpenAICompatibleRequestOptions(),
        execution=OpenAICompatibleExecutionOptions(
            use_stream=True,
            rpm_limit=0,
            transport_retries=10,
            business_retries=10,
        ),
    ))


@dataclass
class EmbeddingConfig(SerializableMixin):
    """向量模型配置"""
    provider: str = "openai"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    base_url: Optional[str] = None
    rpm_limit: int = 0
    transport_retries: int = 10
    business_retries: int = 10
    timeout_seconds: float = 0


@dataclass
class ImageGenConfig(SerializableMixin):
    """生图模型配置"""
    provider: str = "gpt2api"
    api_key: str = ""
    model: str = "gpt-image-2"
    base_url: Optional[str] = None
    transport_retries: int = 10
    business_retries: int = 10
    timeout_seconds: float = 0

@dataclass
class PromptsConfig(SerializableMixin):
    """分析提示词配置"""
    batch_analysis: str = ""       # 批量分析提示词
    segment_summary: str = ""      # 段落总结提示词
    chapter_summary: str = ""      # 章节总结提示词
    book_overview: str = ""        # 全书概要提示词
    group_summary: str = ""        # 分组概要提示词（每N页生成一个）
    qa_response: str = ""          # 问答响应提示词
    question_decompose: str = ""   # 问题分解提示词
    analysis_system: str = ""      # 分析系统提示词
