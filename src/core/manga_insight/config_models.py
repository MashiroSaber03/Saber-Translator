"""
Manga Insight 配置数据模型

使用 dataclass 定义配置对象，支持多种 VLM/Embedding 服务商。
通过 SerializableMixin 自动提供 to_dict/from_dict 方法。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from .config.serialization import SerializableMixin


class APIProvider(Enum):
    """
    统一的 API 服务商枚举

    所有模型类型（VLM、Embedding、Reranker、生图）共用此枚举。
    各服务商支持的能力不同，通过 provider_registry 查询。
    """
    OPENAI = "openai"
    GEMINI = "gemini"
    QWEN = "qwen"
    SILICONFLOW = "siliconflow"
    DEEPSEEK = "deepseek"
    VOLCANO = "volcano"
    JINA = "jina"
    COHERE = "cohere"
    CUSTOM = "custom"


# 向后兼容的别名（避免破坏现有代码）
VLMProvider = APIProvider
EmbeddingProvider = APIProvider
RerankerProvider = APIProvider
ImageGenProvider = APIProvider


class AnalysisDepth(Enum):
    """分析深度枚举"""
    QUICK = "quick"        # 仅基础信息提取
    STANDARD = "standard"  # 标准分析
    DEEP = "deep"          # 深度分析（主题、情感等）


@dataclass
class VLMConfig(SerializableMixin):
    """VLM 多模态模型配置"""
    provider: str = "gemini"
    api_key: str = ""
    model: str = "gemini-2.0-flash"
    base_url: Optional[str] = None
    rpm_limit: int = 10
    max_retries: int = 3
    max_images_per_request: int = 10
    temperature: float = 0.3
    force_json: bool = False  # 强制 JSON 输出（OpenAI 兼容 API）
    use_stream: bool = True  # 使用流式请求（避免超时）
    image_max_size: int = 0  # 图片最大边长（像素），0 表示不压缩


@dataclass
class ChatLLMConfig(SerializableMixin):
    """对话模型配置"""
    use_same_as_vlm: bool = True
    provider: str = "gemini"
    api_key: str = ""
    model: str = "gemini-2.0-flash"
    base_url: Optional[str] = None
    rpm_limit: int = 10
    max_retries: int = 3
    use_stream: bool = True  # 使用流式请求（避免超时）


@dataclass
class EmbeddingConfig(SerializableMixin):
    """向量模型配置"""
    provider: str = "openai"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    base_url: Optional[str] = None
    dimension: int = 1536
    rpm_limit: int = 0
    max_retries: int = 3


@dataclass
class RerankerConfig(SerializableMixin):
    """重排序模型配置（默认启用，需配置 API Key 后生效）"""
    enabled: bool = True  # 默认启用
    provider: str = "jina"
    api_key: str = ""
    model: str = "jina-reranker-v2-base-multilingual"
    base_url: Optional[str] = None
    top_k: int = 5
    rpm_limit: int = 60
    max_retries: int = 3


@dataclass
class ImageGenConfig(SerializableMixin):
    """生图模型配置"""
    provider: str = "siliconflow"
    api_key: str = ""
    model: str = "stabilityai/stable-diffusion-3-5-large"
    base_url: Optional[str] = None
    max_retries: int = 3             # 每张图重试次数


# 预设架构模板
ARCHITECTURE_PRESETS = {
    "simple": {
        "name": "简洁模式",
        "description": "批量分析 → 全书总结（适合短篇，100页以内）",
        "layers": [
            {"name": "批量分析", "units_per_group": 5, "align_to_chapter": False},
            {"name": "全书总结", "units_per_group": 0, "align_to_chapter": False}
        ]
    },
    "standard": {
        "name": "标准模式",
        "description": "批量分析 → 段落总结 → 全书总结（通用）",
        "layers": [
            {"name": "批量分析", "units_per_group": 5, "align_to_chapter": False},
            {"name": "段落总结", "units_per_group": 5, "align_to_chapter": False},
            {"name": "全书总结", "units_per_group": 0, "align_to_chapter": False}
        ]
    },
    "chapter_based": {
        "name": "章节模式",
        "description": "批量分析 → 章节总结 → 全书总结（有明确章节的漫画）",
        "layers": [
            {"name": "批量分析", "units_per_group": 5, "align_to_chapter": True},
            {"name": "章节总结", "units_per_group": 0, "align_to_chapter": True},
            {"name": "全书总结", "units_per_group": 0, "align_to_chapter": False}
        ]
    },
    "full": {
        "name": "完整模式",
        "description": "批量分析 → 小总结 → 章节总结 → 全书总结（长篇连载）",
        "layers": [
            {"name": "批量分析", "units_per_group": 5, "align_to_chapter": False},
            {"name": "小总结", "units_per_group": 5, "align_to_chapter": False},
            {"name": "章节总结", "units_per_group": 0, "align_to_chapter": True},
            {"name": "全书总结", "units_per_group": 0, "align_to_chapter": False}
        ]
    }
}


@dataclass
class BatchAnalysisSettings(SerializableMixin):
    """批量分析设置"""
    pages_per_batch: int = 5                # 每批次分析的页数 (1-10)
    context_batch_count: int = 1            # 作为上文参考的前置批次数量 (0-5)

    # 层级架构配置
    architecture_preset: str = "standard"   # 预设架构: simple/standard/chapter_based/full
    custom_layers: List[Dict[str, Any]] = field(default_factory=list)  # 自定义层级

    def get_layers(self) -> List[Dict[str, Any]]:
        """获取当前架构的层级列表"""
        # 如果是自定义模式且有自定义层级，使用自定义
        if self.architecture_preset == "custom" and self.custom_layers and len(self.custom_layers) > 0:
            return self.custom_layers

        # 否则使用预设（custom 模式但没有自定义层级时回退到 standard）
        preset_key = self.architecture_preset if self.architecture_preset in ARCHITECTURE_PRESETS else "standard"
        preset = ARCHITECTURE_PRESETS.get(preset_key, ARCHITECTURE_PRESETS["standard"])
        return preset["layers"]

    def get_preset_info(self) -> Dict[str, Any]:
        """获取当前预设的信息"""
        return ARCHITECTURE_PRESETS.get(self.architecture_preset, ARCHITECTURE_PRESETS["standard"])


@dataclass
class AnalysisSettings(SerializableMixin):
    """分析设置"""
    depth: str = "standard"
    auto_analyze_new_chapters: bool = False
    save_intermediate_results: bool = True
    batch: BatchAnalysisSettings = field(default_factory=BatchAnalysisSettings)


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


@dataclass
class MangaInsightConfig(SerializableMixin):
    """Manga Insight 完整配置"""
    vlm: VLMConfig = field(default_factory=VLMConfig)
    chat_llm: ChatLLMConfig = field(default_factory=ChatLLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    image_gen: ImageGenConfig = field(default_factory=ImageGenConfig)  # 新增：生图模型配置
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    # 服务商配置缓存（用于切换服务商时保存/恢复配置）
    provider_settings: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """重写以保持向后兼容的键名"""
        result = super().to_dict()
        # 保持 providerSettings 键名向后兼容
        result["providerSettings"] = result.pop("provider_settings", {})
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MangaInsightConfig":
        """重写以处理 providerSettings 键名兼容"""
        if data is None:
            data = {}
        # 兼容 providerSettings 键名
        if "providerSettings" in data and "provider_settings" not in data:
            data["provider_settings"] = data.pop("providerSettings")
        return super().from_dict(data)


# ============================================================
# 提示词模板 - 从 prompts.py 导入（保持向后兼容）
# ============================================================

from .prompts import (
    DEFAULT_QA_SYSTEM_PROMPT,
    DEFAULT_ANALYSIS_SYSTEM_PROMPT,
    DEFAULT_QUESTION_DECOMPOSE_PROMPT,
    DEFAULT_BATCH_ANALYSIS_PROMPT,
    DEFAULT_SEGMENT_SUMMARY_PROMPT,
    DEFAULT_CHAPTER_FROM_SEGMENTS_PROMPT,
    DEFAULT_GROUP_SUMMARY_PROMPT,
    DEFAULT_BOOK_OVERVIEW_PROMPT
)

# ============================================================
# 概要模板 - 从 overview_templates.py 导入（保持向后兼容）
# ============================================================

from .overview_templates import (
    OVERVIEW_TEMPLATES,
    get_overview_templates,
    get_overview_template_prompt
)
