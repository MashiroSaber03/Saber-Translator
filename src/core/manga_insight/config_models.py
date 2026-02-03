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


class ImageGenProvider(Enum):
    """生图服务商枚举"""
    OPENAI = "openai"           # DALL-E
    SILICONFLOW = "siliconflow" # SD3.5等
    QWEN = "qwen"               # 通义万相
    VOLCANO = "volcano"         # 火山引擎
    CUSTOM = "custom"           # 自定义


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
    use_stream: bool = True  # 使用流式请求（避免超时）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_same_as_vlm": self.use_same_as_vlm,
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "rpm_limit": self.rpm_limit,
            "max_retries": self.max_retries,
            "use_stream": self.use_stream
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
            max_retries=data.get("max_retries", 3),
            use_stream=data.get("use_stream", True)
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


@dataclass
class ImageGenConfig:
    """生图模型配置"""
    provider: str = "siliconflow"
    api_key: str = ""
    model: str = "stabilityai/stable-diffusion-3-5-large"
    base_url: Optional[str] = None
    max_retries: int = 3             # 每张图重试次数
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageGenConfig":
        return cls(
            provider=data.get("provider", "siliconflow"),
            api_key=data.get("api_key", ""),
            model=data.get("model", "stabilityai/stable-diffusion-3-5-large"),
            base_url=data.get("base_url"),
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
    image_gen: ImageGenConfig = field(default_factory=ImageGenConfig)  # 新增：生图模型配置
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    # 服务商配置缓存（用于切换服务商时保存/恢复配置）
    provider_settings: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vlm": self.vlm.to_dict(),
            "chat_llm": self.chat_llm.to_dict(),
            "embedding": self.embedding.to_dict(),
            "reranker": self.reranker.to_dict(),
            "image_gen": self.image_gen.to_dict(),
            "analysis": self.analysis.to_dict(),
            "prompts": self.prompts.to_dict(),
            # 保存服务商配置缓存
            "providerSettings": self.provider_settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MangaInsightConfig":
        return cls(
            vlm=VLMConfig.from_dict(data.get("vlm", {})),
            chat_llm=ChatLLMConfig.from_dict(data.get("chat_llm", {})),
            embedding=EmbeddingConfig.from_dict(data.get("embedding", {})),
            reranker=RerankerConfig.from_dict(data.get("reranker", {})),
            image_gen=ImageGenConfig.from_dict(data.get("image_gen", {})),
            analysis=AnalysisSettings.from_dict(data.get("analysis", {})),
            prompts=PromptsConfig.from_dict(data.get("prompts", {})),
            # 恢复服务商配置缓存
            provider_settings=data.get("providerSettings", {})
        )



# ============================================================
# 默认提示词模板
# ============================================================

DEFAULT_QA_SYSTEM_PROMPT = """你是专业的漫画分析助手。请基于提供的漫画内容回答用户问题。

【回答要求】
1. 全程使用中文回答
2. 引用具体页码时使用加粗格式（如"在**第5页**中..."）
3. 回答要准确、有条理
4. 简单问题简短回答（1-3句），复杂问题可展开说明
5. 如果有多个相关内容，使用列表形式列举最相关的2-3个
6. 如果提供的内容无法回答问题，诚实说明"根据已分析的内容，暂未找到相关信息"
7. 不要编造漫画中没有的内容

【格式要求】
- 使用 Markdown 格式输出
- 重点内容使用 **加粗**
- 多个要点使用列表（- 或 1. 2. 3.）
- 引用对话使用 > 引用格式
- 不需要使用标题（#）"""

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


DEFAULT_BOOK_OVERVIEW_PROMPT = """【输出中文】请根据以下内容，生成一份**结构化的剧情概述**，使用 Markdown 格式输出。

【内容摘要】
{section_summaries}

【任务说明】
你需要像给朋友复述故事一样，详细讲述内容中发生的所有事情。这不是宣传简介，而是完整的剧情回顾，可以包含所有剧透。

【输出格式要求 - 必须使用 Markdown】
请使用以下结构组织内容（根据实际情况调整小节）：

## 📖 故事背景
简要介绍故事的世界观、主要角色和初始设定。

## 🎬 剧情发展
按时间线描述主要事件，可以用多个小节：
### 开端
...
### 发展
...
### 转折/高潮
...

## 👥 主要角色
- **角色名**：角色简介和在故事中的作用

## 📌 关键事件
用列表形式列出重要转折点：
- 事件1
- 事件2

## 💭 当前进度
（如果故事未完结）注明分析进度和悬念

【输出策略】
■ 如果内容包含完整故事：完整叙述起因→发展→高潮→结局
■ 如果内容只是故事的一部分：按时间顺序详细叙述，结尾注明进度

【写作要求】
1. **必须使用 Markdown 格式**：标题用 ##，列表用 -，重点用 **加粗**
2. 具体描述事件：谁做了什么、发生了什么、结果如何
3. 不要省略情节：每个重要转折都要提到
4. 避免空话套话，要有具体内容
5. 字数根据内容调整：内容少则300-500字，内容多则500-1000字

请直接输出 Markdown 格式的概述，无需代码块包裹。"""


# ============================================================
# 概要模板系统（多种输出风格）
# ============================================================

OVERVIEW_TEMPLATES = {
    "no_spoiler": {
        "name": "无剧透简介",
        "icon": "🎁",
        "description": "不含关键剧透的故事简介，适合推荐给朋友",
        "prompt": """【输出中文】请根据以下内容，生成一份**无剧透的故事简介**，使用 Markdown 格式输出。

【内容摘要】
{section_summaries}

【任务说明】
这份简介将用于向没看过这部漫画的朋友推荐。请介绍故事的设定和开头，吸引他们的兴趣，但**绝对不能剧透任何重要转折、结局或惊喜**。

【输出格式要求 - 必须使用 Markdown】

## 🎁 故事简介

### 🌍 故事背景
介绍世界观设定、时代背景等基础信息。

### 👤 主角介绍
介绍主要角色的基本信息和初始状态（不透露后续发展）。

### 📖 故事开端
只描述故事的起点和初始冲突，用悬念吸引读者。

### ✨ 推荐理由
用2-3点说明这部漫画的亮点（画风、剧情特色、情感等）。

### 🏷️ 标签
列出适合的标签，如：#热血 #悬疑 #恋爱 #奇幻 等

【写作要求】
1. **严禁剧透**：不透露任何转折、真相、结局
2. **制造悬念**：让读者想知道"后来怎么样了"
3. **突出亮点**：强调作品的独特魅力
4. 总字数控制在 200-400 字

请直接输出 Markdown 格式的简介，无需代码块包裹。"""
    },
    
    "story_summary": {
        "name": "故事概要",
        "icon": "📖",
        "description": "完整的剧情回顾，包含所有剧透，适合回顾整个故事",
        "prompt": """【输出中文】请根据以下内容，生成一份**完整的故事概要**，使用 Markdown 格式输出。

【内容摘要】
{section_summaries}

【任务说明】
你需要像给朋友复述故事一样，详细讲述内容中发生的所有事情。这不是宣传简介，而是完整的剧情回顾，可以包含所有剧透。

【输出格式要求 - 必须使用 Markdown】
请使用以下结构组织内容（根据实际情况调整小节）：

## 📖 故事背景
简要介绍故事的世界观、主要角色和初始设定。

## 🎬 剧情发展
按时间线描述主要事件，可以用多个小节：
### 开端
...
### 发展
...
### 转折/高潮
...

## 👥 主要角色
- **角色名**：角色简介和在故事中的作用

## 📌 关键事件
用列表形式列出重要转折点：
- 事件1
- 事件2

## 💭 当前进度
（如果故事未完结）注明分析进度和悬念

【写作要求】
1. **必须使用 Markdown 格式**：标题用 ##，列表用 -，重点用 **加粗**
2. 具体描述事件：谁做了什么、发生了什么、结果如何
3. 不要省略情节：每个重要转折都要提到
4. 避免空话套话，要有具体内容
5. 字数根据内容调整：内容少则300-500字，内容多则800-1200字

请直接输出 Markdown 格式的概述，无需代码块包裹。"""
    },
    
    "recap": {
        "name": "前情回顾",
        "icon": "⏪",
        "description": "精炼版剧情回顾，适合接续阅读前快速回忆",
        "prompt": """【输出中文】请根据以下内容，生成一份**精炼的前情回顾**，使用 Markdown 格式输出。

【内容摘要】
{section_summaries}

【任务说明】
读者即将继续阅读这部漫画，需要快速回忆之前发生了什么。请生成一份简洁但信息完整的前情回顾。

【输出格式要求 - 必须使用 Markdown】

## ⏪ 前情回顾

### 📍 故事进展到哪了
用1-2句话说明当前剧情进度。

### 🔑 你需要记住的关键信息
用简洁的列表形式列出：
- **重要人物**：谁是谁，他们的关系
- **核心冲突**：主要矛盾是什么
- **最新进展**：最近发生了什么重要的事

### ⚡ 上次的关键场景
简述最近1-2个重要场景，帮助读者快速进入状态。

### ❓ 待解决的悬念
列出尚未揭晓的谜团或未完成的事件。

【写作要求】
1. **简洁为主**：每个要点控制在1-2句话
2. **突出重点**：只保留对后续剧情有影响的信息
3. **便于快速阅读**：多用列表，少用长段落
4. 总字数控制在 300-500 字

请直接输出 Markdown 格式的前情回顾，无需代码块包裹。"""
    },
    
    "character_guide": {
        "name": "角色图鉴",
        "icon": "👥",
        "description": "详细的人物介绍和关系梳理",
        "prompt": """【输出中文】请根据以下内容，生成一份**角色图鉴**，使用 Markdown 格式输出。

【内容摘要】
{section_summaries}

【任务说明】
整理这部漫画中出现的所有重要角色，包括他们的特点、关系和在故事中的作用。

【输出格式要求 - 必须使用 Markdown】

## 👥 角色图鉴

### 🌟 主要角色

#### [角色名]
- **身份**：角色的身份/职业
- **性格**：性格特点描述
- **特点**：外貌特征或标志性元素
- **故事作用**：在剧情中扮演的角色
- **关键行为**：做过的重要事情

（为每个主要角色创建类似条目）

### 📋 次要角色
用简短的列表介绍配角：
- **角色名**：一句话介绍

### 🔗 人物关系
用清晰的方式描述角色之间的关系：
- A 与 B：关系描述（如：师徒、对手、恋人等）
- ...

### ⚔️ 阵营/势力（如适用）
如果故事中有不同阵营，列出各阵营及其成员。

【写作要求】
1. **信息准确**：基于分析内容，不要编造
2. **条理清晰**：按重要程度排序
3. **关系明确**：人物关系要写清楚
4. 根据角色数量调整篇幅

请直接输出 Markdown 格式的角色图鉴，无需代码块包裹。"""
    },
    
    "world_setting": {
        "name": "世界观设定",
        "icon": "🌍",
        "description": "故事的世界观、势力、规则等背景设定",
        "prompt": """【输出中文】请根据以下内容，生成一份**世界观设定集**，使用 Markdown 格式输出。

【内容摘要】
{section_summaries}

【任务说明】
整理这部漫画的世界观设定，包括背景、势力、规则等，帮助读者理解故事发生的世界。

【输出格式要求 - 必须使用 Markdown】

## 🌍 世界观设定

### 🗺️ 世界背景
描述故事发生的世界/时代/地点的基本情况。

### ⚡ 力量体系（如适用）
如果故事中有特殊能力/魔法/科技等设定：
- **体系名称**：基本原理
- **等级划分**：如果有的话
- **代表能力**：主要角色使用的能力

### 🏛️ 势力与组织
列出故事中的重要势力/组织/国家：
- **势力名**：简介、立场、代表人物

### 📜 重要规则/设定
故事中的特殊规则或设定：
- 规则1：说明
- 规则2：说明

### 📍 重要地点
故事中出现的关键场所：
- **地点名**：简介及其重要性

### 📚 术语表（如适用）
故事中的专有名词解释：
- **术语**：解释

【写作要求】
1. **基于内容**：只整理漫画中明确出现的设定
2. **条理分明**：分类清晰，便于查阅
3. **简洁准确**：每个条目简明扼要
4. 如果某些类别在漫画中没有涉及，可以省略

请直接输出 Markdown 格式的设定集，无需代码块包裹。"""
    },
    
    "highlights": {
        "name": "名场面盘点",
        "icon": "✨",
        "description": "精彩场景和高光时刻回顾，附页码定位",
        "prompt": """【输出中文】请根据以下内容，生成一份**名场面盘点**，使用 Markdown 格式输出。

【内容摘要】
{section_summaries}

【任务说明】
盘点这部漫画中最精彩、最令人印象深刻的场景和时刻，帮助读者回顾或定位想重温的片段。

【输出格式要求 - 必须使用 Markdown】

## ✨ 名场面盘点

### 🔥 高燃时刻
最热血、最激动人心的场景：

#### 1. [场景标题]
- **页码**：第 X 页
- **场景描述**：发生了什么
- **精彩之处**：为什么这个场景令人印象深刻

（列出 3-5 个高燃场景）

### 💕 感人瞬间
最触动人心的情感场景：

#### 1. [场景标题]
- **页码**：第 X 页
- **场景描述**：发生了什么
- **感人之处**：为什么这个场景令人感动

（列出 2-4 个感人场景）

### 😂 趣味时刻（如适用）
轻松搞笑的场景：
- **第 X 页**：简述场景

### 🎨 视觉名场面
画面特别精美或有冲击力的场景：
- **第 X 页**：简述场景及视觉亮点

### 💬 经典台词
令人印象深刻的对白：
> "台词内容" —— 角色名（第 X 页）

【写作要求】
1. **标注页码**：每个场景都要注明大致页码范围
2. **生动描述**：让没看过的人也能感受到精彩
3. **分类清晰**：按场景类型分类
4. 根据内容丰富程度调整数量

请直接输出 Markdown 格式的名场面盘点，无需代码块包裹。"""
    },
    
    "reading_notes": {
        "name": "阅读笔记",
        "icon": "📝",
        "description": "结构化的阅读笔记，包含要点和思考",
        "prompt": """【输出中文】请根据以下内容，生成一份**阅读笔记**，使用 Markdown 格式输出。

【内容摘要】
{section_summaries}

【任务说明】
生成一份结构化的阅读笔记，帮助读者整理和记忆这部漫画的内容。

【输出格式要求 - 必须使用 Markdown】

## 📝 阅读笔记

### 📋 基本信息
- **当前进度**：已分析到第 X 页
- **主要类型**：（如：热血/悬疑/恋爱等）
- **核心主题**：一句话概括

### 📖 剧情脉络
用简洁的时间线形式梳理主要剧情：
1. **开端**：...
2. **发展**：...
3. **当前**：...

### 🔑 关键要点
需要记住的重要信息：
- [ ] 要点1
- [ ] 要点2
- [ ] 要点3

### ❓ 未解之谜
目前尚未揭晓的悬念：
1. ...
2. ...

### 💡 个人观察
基于分析内容的一些观察和推测：
- 观察1
- 观察2

### 🔖 值得重温的部分
推荐回看的页码和原因：
- **第 X-Y 页**：原因

### ⭐ 评价要素
- **剧情**：简评
- **角色**：简评
- **节奏**：简评

【写作要求】
1. **结构清晰**：便于快速查阅
2. **要点突出**：抓住最重要的信息
3. **客观分析**：基于内容进行合理推断
4. 总字数 400-700 字

请直接输出 Markdown 格式的阅读笔记，无需代码块包裹。"""
    }
}

# 获取模板列表（供 API 使用）
def get_overview_templates() -> dict:
    """获取所有概要模板的元信息"""
    return {
        key: {
            "name": template["name"],
            "icon": template["icon"],
            "description": template["description"]
        }
        for key, template in OVERVIEW_TEMPLATES.items()
    }

# 获取指定模板的提示词
def get_overview_template_prompt(template_key: str) -> str:
    """获取指定模板的提示词"""
    template = OVERVIEW_TEMPLATES.get(template_key)
    if template:
        return template["prompt"]
    # 默认返回故事概要模板
    return OVERVIEW_TEMPLATES["story_summary"]["prompt"]
