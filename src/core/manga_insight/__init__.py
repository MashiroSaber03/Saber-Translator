"""
Manga Insight 漫画智能分析系统

核心模块，提供漫画深度分析、智能问答等功能。
"""

import logging

logger = logging.getLogger("MangaInsight")

# 导出主要组件
from .config_models import (
    MangaInsightConfig,
    VLMConfig,
    ChatLLMConfig,
    EmbeddingConfig,
    RerankerConfig,
    ImageGenConfig,
    AnalysisSettings,
    VLMProvider,
    EmbeddingProvider,
    RerankerProvider,
    ImageGenProvider,
    AnalysisDepth
)

from .task_models import (
    AnalysisTask,
    AnalysisProgress,
    TaskStatus,
    TaskType
)

from .task_manager import AnalysisTaskManager
from .storage import AnalysisStorage
from .config_utils import load_insight_config, save_insight_config

# 版本号
__version__ = "1.0.0"
