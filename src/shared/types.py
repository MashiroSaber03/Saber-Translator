"""
类型定义模块

定义项目中常用的类型别名，提升代码可读性和类型安全性
"""

from typing import TypeAlias, Tuple, List, Dict, Any, Optional, Union
from PIL import Image

# 图像相关类型
ImageType: TypeAlias = Image.Image
BubbleCoord: TypeAlias = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
BubbleCoords: TypeAlias = List[BubbleCoord]

# 文本相关类型
TextList: TypeAlias = List[str]
TranslationResult: TypeAlias = Tuple[TextList, TextList]  # (bubble_texts, textbox_texts)

# 状态相关类型
Position: TypeAlias = Dict[str, int]  # {'x': int, 'y': int}
BubbleState: TypeAlias = Dict[str, Any]  # 气泡状态字典
BubbleStatesList: TypeAlias = List[BubbleState]  # 气泡状态列表

# 颜色类型
ColorHex: TypeAlias = str  # '#RRGGBB' 格式
ColorRGB: TypeAlias = Tuple[int, int, int]  # (R, G, B)
ColorType: TypeAlias = Union[ColorHex, ColorRGB]

# 处理结果类型
ProcessingResult: TypeAlias = Tuple[
    ImageType,      # processed_image
    TextList,       # original_texts
    TextList,       # translated_bubble_texts
    TextList,       # translated_textbox_texts
    BubbleCoords,   # bubble_coords
    BubbleStatesList  # bubble_states
]

# OCR相关类型
OCREngine: TypeAlias = str  # 'auto' | 'manga_ocr' | 'paddle_ocr' | 'baidu_ocr' | 'ai_vision'
LanguageCode: TypeAlias = str  # 'zh' | 'en' | 'japan' | 'korean' ...

# 翻译相关类型
ModelProvider: TypeAlias = str  # 'siliconflow' | 'deepseek' | 'ollama' ...
APIKey: TypeAlias = Optional[str]

# 配置相关类型
ConfigDict: TypeAlias = Dict[str, Any]
PromptContent: TypeAlias = str

# 插件相关类型
HookName: TypeAlias = str
PluginName: TypeAlias = str
PluginMetadata: TypeAlias = Dict[str, str]  # {'name', 'version', 'author', 'description'}
