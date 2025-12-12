"""
Manga Insight 配置数据模型

使用 dataclass 定义配置对象，支持多种 VLM/Embedding 服务商。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum


class VLMProvider(Enum):
    """VLM 服务商枚举"""
    GEMINI = "gemini"
    OPENAI = "openai"
    QWEN = "qwen"
    SILICONFLOW = "siliconflow"
    DEEPSEEK = "deepseek"
    VOLCANO = "volcano"
    CUSTOM = "custom"


class EmbeddingProvider(Enum):
    """Embedding 服务商枚举"""
    OPENAI = "openai"
    SILICONFLOW = "siliconflow"
    LOCAL = "local"
    CUSTOM = "custom"


class RerankerProvider(Enum):
    """Reranker 服务商枚举"""
    JINA = "jina"
    COHERE = "cohere"
    SILICONFLOW = "siliconflow"
    BGE = "bge"
    CUSTOM = "custom"


class AnalysisDepth(Enum):
    """分析深度枚举"""
    QUICK = "quick"        # 仅基础信息提取
    STANDARD = "standard"  # 标准分析
    DEEP = "deep"          # 深度分析（主题、情感等）


@dataclass
class VLMConfig:
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "rpm_limit": self.rpm_limit,
            "max_retries": self.max_retries,
            "max_images_per_request": self.max_images_per_request,
            "temperature": self.temperature,
            "force_json": self.force_json,
            "use_stream": self.use_stream,
            "image_max_size": self.image_max_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLMConfig":
        return cls(
            provider=data.get("provider", "gemini"),
            api_key=data.get("api_key", ""),
            model=data.get("model", "gemini-2.0-flash"),
            base_url=data.get("base_url"),
            rpm_limit=data.get("rpm_limit", 10),
            max_retries=data.get("max_retries", 3),
            max_images_per_request=data.get("max_images_per_request", 10),
            temperature=data.get("temperature", 0.3),
            force_json=data.get("force_json", False),
            use_stream=data.get("use_stream", True),
            image_max_size=data.get("image_max_size", 0)
        )


@dataclass
class ChatLLMConfig:
    """对话模型配置"""
    use_same_as_vlm: bool = True
    provider: str = "gemini"
    api_key: str = ""
    model: str = "gemini-2.0-flash"
    base_url: Optional[str] = None
    rpm_limit: int = 10
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_same_as_vlm": self.use_same_as_vlm,
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "rpm_limit": self.rpm_limit,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatLLMConfig":
        return cls(
            use_same_as_vlm=data.get("use_same_as_vlm", True),
            provider=data.get("provider", "gemini"),
            api_key=data.get("api_key", ""),
            model=data.get("model", "gemini-2.0-flash"),
            base_url=data.get("base_url"),
            rpm_limit=data.get("rpm_limit", 10),
            max_retries=data.get("max_retries", 3)
        )


@dataclass
class EmbeddingConfig:
    """向量模型配置"""
    provider: str = "openai"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    base_url: Optional[str] = None
    dimension: int = 1536
    rpm_limit: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "dimension": self.dimension,
            "rpm_limit": self.rpm_limit,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingConfig":
        return cls(
            provider=data.get("provider", "openai"),
            api_key=data.get("api_key", ""),
            model=data.get("model", "text-embedding-3-small"),
            base_url=data.get("base_url"),
            dimension=data.get("dimension", 1536),
            rpm_limit=data.get("rpm_limit", 0),
            max_retries=data.get("max_retries", 3)
        )


@dataclass
class RerankerConfig:
    """重排序模型配置（默认启用，需配置 API Key 后生效）"""
    enabled: bool = True  # 默认启用
    provider: str = "jina"
    api_key: str = ""
    model: str = "jina-reranker-v2-base-multilingual"
    base_url: Optional[str] = None
    top_k: int = 5
    rpm_limit: int = 60
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "top_k": self.top_k,
            "rpm_limit": self.rpm_limit,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RerankerConfig":
        return cls(
            enabled=data.get("enabled", True),  # 默认启用
            provider=data.get("provider", "jina"),
            api_key=data.get("api_key", ""),
            model=data.get("model", "jina-reranker-v2-base-multilingual"),
            base_url=data.get("base_url"),
            top_k=data.get("top_k", 5),
            rpm_limit=data.get("rpm_limit", 60),
            max_retries=data.get("max_retries", 3)
        )


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
class BatchAnalysisSettings:
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages_per_batch": self.pages_per_batch,
            "context_batch_count": self.context_batch_count,
            "architecture_preset": self.architecture_preset,
            "custom_layers": self.custom_layers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchAnalysisSettings":
        return cls(
            pages_per_batch=data.get("pages_per_batch", 5),
            context_batch_count=data.get("context_batch_count", 1),
            architecture_preset=data.get("architecture_preset", "standard"),
            custom_layers=data.get("custom_layers", [])
        )


@dataclass
class AnalysisSettings:
    """分析设置"""
    depth: str = "standard"
    auto_analyze_new_chapters: bool = False
    save_intermediate_results: bool = True
    batch: BatchAnalysisSettings = field(default_factory=BatchAnalysisSettings)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "auto_analyze_new_chapters": self.auto_analyze_new_chapters,
            "save_intermediate_results": self.save_intermediate_results,
            "batch": self.batch.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisSettings":
        return cls(
            depth=data.get("depth", "standard"),
            auto_analyze_new_chapters=data.get("auto_analyze_new_chapters", False),
            save_intermediate_results=data.get("save_intermediate_results", True),
            batch=BatchAnalysisSettings.from_dict(data.get("batch", {}))
        )


@dataclass
class PromptsConfig:
    """分析提示词配置"""
    batch_analysis: str = ""       # 批量分析提示词
    segment_summary: str = ""      # 段落总结提示词
    chapter_summary: str = ""      # 章节总结提示词
    book_overview: str = ""        # 全书概要提示词
    group_summary: str = ""        # 分组概要提示词（每N页生成一个）
    qa_response: str = ""          # 问答响应提示词
    question_decompose: str = ""   # 问题分解提示词
    analysis_system: str = ""      # 分析系统提示词
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_analysis": self.batch_analysis,
            "segment_summary": self.segment_summary,
            "chapter_summary": self.chapter_summary,
            "book_overview": self.book_overview,
            "group_summary": self.group_summary,
            "qa_response": self.qa_response,
            "question_decompose": self.question_decompose,
            "analysis_system": self.analysis_system
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptsConfig":
        return cls(
            batch_analysis=data.get("batch_analysis", ""),
            segment_summary=data.get("segment_summary", ""),
            chapter_summary=data.get("chapter_summary", ""),
            book_overview=data.get("book_overview", ""),
            group_summary=data.get("group_summary", ""),
            qa_response=data.get("qa_response", ""),
            question_decompose=data.get("question_decompose", ""),
            analysis_system=data.get("analysis_system", "")
        )


@dataclass
class MangaInsightConfig:
    """Manga Insight 完整配置"""
    vlm: VLMConfig = field(default_factory=VLMConfig)
    chat_llm: ChatLLMConfig = field(default_factory=ChatLLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vlm": self.vlm.to_dict(),
            "chat_llm": self.chat_llm.to_dict(),
            "embedding": self.embedding.to_dict(),
            "reranker": self.reranker.to_dict(),
            "analysis": self.analysis.to_dict(),
            "prompts": self.prompts.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MangaInsightConfig":
        return cls(
            vlm=VLMConfig.from_dict(data.get("vlm", {})),
            chat_llm=ChatLLMConfig.from_dict(data.get("chat_llm", {})),
            embedding=EmbeddingConfig.from_dict(data.get("embedding", {})),
            reranker=RerankerConfig.from_dict(data.get("reranker", {})),
            analysis=AnalysisSettings.from_dict(data.get("analysis", {})),
            prompts=PromptsConfig.from_dict(data.get("prompts", {}))
        )


# ============================================================
# 默认提示词模板
# ============================================================

DEFAULT_QA_SYSTEM_PROMPT = """你是专业的漫画分析助手。请基于提供的漫画内容回答用户问题。

【回答要求】
1. 全程使用中文回答
2. 引用具体页码（如"在第5页中..."）
3. 回答要准确、有条理
4. 简单问题简短回答（1-3句），复杂问题可展开说明
5. 如果有多个相关内容，列举最相关的2-3个
6. 如果提供的内容无法回答问题，诚实说明"根据已分析的内容，暂未找到相关信息"
7. 不要编造漫画中没有的内容"""

DEFAULT_QUESTION_DECOMPOSE_PROMPT = """你是漫画内容检索助手。请将用户的复杂问题分解为2-4个独立的子问题，便于分别检索。

【分解原则】
- 每个子问题针对一个具体信息点
- 子问题综合起来能回答原问题
- 如果原问题已经足够简单，返回 {{"sub_questions": ["原问题"]}}

【用户问题】
{question}

【输出要求】
必须且只能输出以下JSON格式，不要有任何其他文字：
{{"sub_questions": ["子问题1", "子问题2"]}}"""

DEFAULT_ANALYSIS_SYSTEM_PROMPT = "你是一个漫画剧情分析师，请生成结构化的分析结果。"


# ============================================================
# 批量分析模式提示词
# ============================================================

DEFAULT_BATCH_ANALYSIS_PROMPT = """你是一个专业的漫画分析师。请分析这组连续的 {page_count} 张漫画页面（第 {start_page} 页至第 {end_page} 页）。

【重要说明】
- 这是漫画原图（未翻译版本），请直接阅读原文内容
- 无论漫画原文是什么语言，你的所有输出内容必须使用中文
- 请特别关注页面之间的剧情连续性

请按以下 JSON 格式返回结果：
{{
    "page_range": {{
        "start": {start_page},
        "end": {end_page}
    }},
    "pages": [
        {{
            "page_number": <页码>,
            "page_summary": "<该页详细内容概括，包含场景描述、角色行为、重要对话和情节发展。简单页面80-150字，复杂页面150-300字，根据内容丰富程度调整>"
        }}
    ],
    "batch_summary": "<这组页面的整体剧情概述，详细描述故事发展、角色互动和情感变化，200-400字>",
    "key_events": ["<这组页面中的3-5个关键事件，每个事件用一句话概括>"],
    "continuity_notes": "<与上文的衔接、场景转换、剧情走向说明>"
}}

注意：
1. 按正确的漫画阅读顺序分析
2. 重点关注剧情发展和角色互动
3. page_summary 要详细描述该页发生的事情
4. batch_summary 要完整概括这批页面的故事内容

【重要】请直接输出JSON，不要包含任何解释、markdown代码块或其他文字。"""


DEFAULT_SEGMENT_SUMMARY_PROMPT = """【输出中文】基于以下 {batch_count} 个批次的分析结果（第 {start_page} 页至第 {end_page} 页），生成一个连贯的小总结。

【批次分析结果】
{batch_summaries}

请生成结构化的小总结，JSON 格式：
{{
    "segment_id": "{segment_id}",
    "page_range": {{
        "start": {start_page},
        "end": {end_page}
    }},
    "summary": "<这段内容的主要剧情概括，详细描述故事发展、角色互动和关键事件，150-300字>"
}}

要求：
1. 整合各批次的信息，形成连贯叙述
2. 突出重要角色和关键事件
3. 注意剧情的因果关系

【重要】请直接输出JSON，不要包含任何解释、markdown代码块或其他文字。"""


DEFAULT_CHAPTER_FROM_SEGMENTS_PROMPT = """【输出中文】基于以下小总结，生成完整的章节总结。

【章节信息】
章节：{chapter_title}
页面范围：第 {start_page} 页至第 {end_page} 页

【小总结列表】
{segment_summaries}

请生成章节总结，JSON 格式：
{{
    "chapter_id": "{chapter_id}",
    "title": "{chapter_title}",
    "page_range": {{
        "start": {start_page},
        "end": {end_page}
    }},
    "summary": "<本章完整剧情概述，按时间顺序描述主要事件和角色行为，400-600字>",
    "main_plot": "<一句话概括本章核心剧情线，如'主角与敌人首次交锋并险胜'>",
    "key_events": ["<按顺序列出3-5个关键事件，每个事件一句话描述>"],
    "connections": {{
        "previous": "<本章开头与前文的衔接，如'承接上章的战斗结束后...'，首章可留空>",
        "foreshadowing": "<本章埋下的伏笔或未解决的悬念，如'神秘人物的身份仍未揭晓'>"
    }}
}}

要求：
1. 综合所有小总结，形成完整的章节叙述
2. 理清人物关系和剧情脉络
3. summary 要详细描述剧情发展，不要空泛概括

【重要】请直接输出JSON，不要包含任何解释、markdown代码块或其他文字。"""


# ============================================================
# 概要生成提示词（统一管理）
# ============================================================

DEFAULT_GROUP_SUMMARY_PROMPT = """【输出中文】请将以下第 {start_page} 页至第 {end_page} 页的漫画内容总结为一个连贯的段落。

【页面内容】
{page_contents}

要求：
1. 按时间顺序描述主要事件和角色行为
2. 不要遗漏关键剧情转折
3. 字数150-250字，根据内容复杂度调整"""


DEFAULT_BOOK_OVERVIEW_PROMPT = """【输出中文】请根据以下内容，生成一份**详细的剧情概述**。

【内容摘要】
{section_summaries}

【任务说明】
你需要像给朋友复述故事一样，详细讲述内容中发生的所有事情。这不是宣传简介，而是完整的剧情回顾，可以包含所有剧透。

【输出策略】
请先判断内容的完整性，然后选择合适的方式：

■ 如果内容包含完整故事（有明确的结局/收尾）：
  → 按「起因 → 发展 → 高潮 → 结局」的结构完整叙述
  → 详细描述故事如何开始、中间发生了什么、最后如何收尾
  
■ 如果内容只是故事的一部分（剧情戛然而止、没有结局）：
  → 按时间顺序详细叙述已有的剧情发展
  → 在结尾注明「（目前分析至第X页/第X章，故事仍在继续）」
  → 不要编造或猜测后续剧情

【写作要求】
1. 具体描述事件：谁做了什么、发生了什么、结果如何
2. 不要省略情节：每个重要转折都要提到
3. 避免空话套话：
   ❌ "展现了友情的可贵" "描绘了成长的主题"
   ✅ "小明帮助小红逃出了困境" "主角最终选择了..."
4. 字数根据内容调整：内容少则200-400字，内容多则400-800字

请直接输出概述正文，无需标题。"""
