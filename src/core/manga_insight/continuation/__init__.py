"""
Manga Insight 漫画续写模块

提供基于已分析漫画数据的AI续写功能。
"""

import logging

logger = logging.getLogger("MangaInsight.Continuation")

from .models import (
    ChapterScript,
    PageContent,
    CharacterForm,
    CharacterProfile,
    ContinuationCharacters
)

from .story_generator import StoryGenerator
from .image_generator import ImageGenerator
from .character_manager import CharacterManager

__all__ = [
    'ChapterScript',
    'PageContent',
    'CharacterForm',
    'CharacterProfile',
    'ContinuationCharacters',
    'StoryGenerator',
    'ImageGenerator',
    'CharacterManager'
]
